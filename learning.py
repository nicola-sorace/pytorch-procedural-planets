"""
Neural network which tries to retrieve Perlin gradients from a final planet image
"""
import os
import time
import traceback
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import read_image
import torchsummary
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt
from matplotlib import cm
from planet_generation_perlin import grids_to_planet, split_planet_into_faces, apply_offset
from conf import device, data_path, img_size, layers


#%% Setup data
class PlanetDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_names = [
            file[:-4]
            for file in os.listdir(path)
            if file.endswith('.png')
        ]

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        name = self.all_names[idx]
        img = read_image(os.path.join(self.path, name + '.png')).to(device).float() / 255.0
        grids = [x.to(device) for x in torch.load(os.path.join(self.path, name + '.pt'), map_location=device)]
        return img, grids


model_bkup_path = "model_bkup.pt"
colormap = cm.turbo

all_data = PlanetDataset(data_path)

train_size = int(len(all_data) * 0.7)
test_size = len(all_data) - train_size
train_data, test_data = random_split(all_data, [train_size, test_size])

batch_size = 64
num_batches = 500
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True, num_workers=2 if device == 'cpu' else 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2 if device == 'cpu' else 0)
print(f"Dataset: {len(train_data)} train, {len(test_data)} test, {len(all_data)} total")


#%% Setup network
class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, cross_convs, activ, pool):
        super(DownLayer, self).__init__()
        self.activ = activ
        self.pool = pool
        self.downConv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchNorms = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in range(1 + cross_convs)
        ])
        self.crossConvs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
            for _ in range(cross_convs)
        ])

    def forward(self, x):
        x = self.pool(self.activ(self.batchNorms[0](self.downConv(x))))
        for i in range(len(self.crossConvs)):
            x = self.activ(self.batchNorms[i + 1](self.crossConvs[i](x)))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_channels, concat_channels, out_channels, kernel_size, scale, padding, cross_convs, activ):
        super(UpLayer, self).__init__()
        self.activ = activ
        self.partial_out_channels = out_channels - concat_channels
        self.upConv = nn.ConvTranspose2d(in_channels, self.partial_out_channels, scale, stride=scale)
        self.batchNorm0 = nn.BatchNorm2d(self.partial_out_channels)
        self.batchNorms = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in range(cross_convs)
        ])
        self.crossConvs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
            for _ in range(cross_convs)
        ])

    def forward(self, x, c):
        x = torch.concat((c, self.activ(self.batchNorm0(self.upConv(x)))), dim=1)
        for i in range(len(self.crossConvs)):
            x = self.activ(self.batchNorms[i](self.crossConvs[i](x)))
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.output_images = True
        self.process_width = 32  # Width of planet faces during processing
        self.activ = nn.ReLU()
        self.outActiv = nn.Tanh()  # Output needs to be in [-1, 1]
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # Down-path
        in_channels = 1 + 4 * len(layers)
        self.downLayer0 = DownLayer(in_channels, 32, 3, 'same', 2, self.activ, self.pool)
        self.downLayer1 = DownLayer(32, 64, 3, 'same', 2, self.activ, self.pool)
        self.downLayer2 = DownLayer(64, 128, 3, 'same', 2, self.activ, self.pool)
        self.downLayer3 = DownLayer(128, 256, 3, 'same', 2, self.activ, self.pool)

        # Up-path
        self.upLayer0 = UpLayer(256, 128, 256, 3, 2, 'same', 2, self.activ)
        self.upLayer1 = UpLayer(256, 64, 128, 3, 2, 'same', 2, self.activ)
        self.upLayer2 = UpLayer(128, 32, 64, 3, 2, 'same', 2, self.activ)
        self.upLayer3 = UpLayer(64, in_channels, 32 + in_channels, 3, 2, 'same', 2, self.activ)

        # Out
        self.outConv0 = nn.Conv2d(32 + in_channels, len(layers) * 3, 3, padding=(1, 1))

        pool_sizes = [
            int((self.process_width - 2) / (layer + 1))
            for layer in layers
        ]
        self.grid_pools = [
            nn.AvgPool3d((1, pool_sizes[i], pool_sizes[i]))
            for i, layer in enumerate(layers)
        ]

    def set_output_images(self, output_images):
        self.output_images = output_images
        return self

    def forward(self, x):
        # Super-impose the location of nodes onto the region they interpolate to
        x = torch.concat([x] + [
            torch.concat([
                apply_offset(
                    x,
                    torch.FloatTensor([i * torch.pi / 4 / l]).to(device),
                    torch.FloatTensor([j * torch.pi / 4 / l]).to(device))
                for i, j in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            ], 1)
            for l in layers
        ], 1)

        # Split into faces (peeking beyond edges), and flatten faces dimension into batch dimension
        l0 = split_planet_into_faces(x, self.process_width).flatten(0, 1)

        # Down-path
        l1 = self.downLayer0(l0)             # -> 32x32x32
        l2 = self.downLayer1(l1)             # -> 16x16x64
        l3 = self.downLayer2(l2)             # -> 8x8x128
        x = self.downLayer3(l3)              # -> 4x4x256
        # Up-path
        x = self.upLayer0(x, l3)             # -> 8x8x512
        x = self.upLayer1(x, l2)             # -> 16x16x256
        x = self.upLayer2(x, l1)             # -> 32x32x128
        x = self.upLayer3(x, l0)             # -> 64x64x65

        # Out
        x = self.outActiv(self.outConv0(x))  # -> 62x62x64

        # Unflatten to retrieve faces dimension
        x = x.unflatten(0, (batch_size, 6))
        # Pool outputs into required sizes
        grids = [
            self.grid_pools[i](x[:, :, i*3:i*3+3, :, :])
            for i in range(len(layers))
        ]
        if self.output_images:
            # Generate planet images from grids
            x = grids_to_planet(grids, batch_size, img_size)
            return grids, x[:, None, :, :]
        else:
            return grids, None


net = Network().to(device)
# torchsummary.summary(net, (1, img_size[1], img_size[0]))
print(f"Model parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

criterion = nn.MSELoss()

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.1)
# optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.9)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda epoch: 0.8 ** epoch
)

losses = []
val_losses = []


#%% Load saved weights
if os.path.isfile(model_bkup_path):
    print("Continuing from checkpoint")
    weights, losses, val_losses = torch.load(model_bkup_path, map_location=device)
    net.load_state_dict(weights)

#%% Train

def save():
    torch.save((net.state_dict(), losses, val_losses), model_bkup_path)


class EpochBar(IncrementalBar):
    def __init__(self, name, max: int):
        super().__init__(name, max=max)
        self.loss = 0
        self.val_loss = None

    @property
    def suffix(self):
        if self.val_loss is None:
            return "[%(index)d/%(max)d] Loss: %(loss).8f (%(runtime).0f imgs/sec)"
        else:
            return "[%(index)d/%(max)d] Loss: %(loss).8f (%(runtime).0f imgs/sec) | Val Loss: %(val_loss).8f"

    def next(self, loss, runtime, val_loss=None):
        self.loss = loss
        self.runtime = runtime
        self.val_loss = val_loss
        super().next()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    net.train()
    print("Training")
    try:
        for epoch in range(num_batches):
            running_loss = 0.0
            start_time = time.time()
            bar = EpochBar(f"Epoch {epoch}", max=len(train_loader))
            for i, (inputs, labels) in enumerate(train_loader):
                output_grids, output_imgs = net(inputs)
                loss = criterion(output_imgs, inputs)

                # # Alternatively train by grid loss instead of image loss
                # loss = criterion(
                #     torch.concat([x.flatten() for x in output_grids]),
                #     torch.concat([x.flatten() for x in labels]),
                # )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss = running_loss / (i + 1)
                imgs_per_sec = batch_size / ((time.time() - start_time) / (i + 1))
                bar.next(loss, imgs_per_sec)

            running_val_loss = 0.0
            for i, (inputs, labels) in enumerate(test_loader):
                output_grids, output_imgs = net(inputs)
                loss = criterion(output_imgs, inputs)
                running_val_loss += loss.item()
            val_loss = running_val_loss / len(test_loader)
            bar.next(loss, imgs_per_sec, val_loss)
            losses.append(loss)
            val_losses.append(val_loss)

            bar.finish()
            scheduler.step()
        save()
        print("Done")
    except KeyboardInterrupt:
        print('\n')
        while True:
            try:
                choice = input("Save? (Y/n) ")
                if choice in ['y', 'Y']:
                    save()
                    break
                if choice in ['n', 'N']:
                    break
            except KeyboardInterrupt:
                pass
            print("Invalid choice")
    except Exception as e:
        traceback.print_exc()
        save()

#%% Result
if __name__ == "__main__":
    # Plot loss trace
    fig, ax = plt.subplots()
    ax.plot(
        range(len(losses)),
        losses,
        label="Training Loss",
        color='blue'
    )
    ax.plot(
        range(len(val_losses)),
        val_losses,
        label="Validation Loss",
        color='orange'
    )
    ax.legend()
    fig.show()
    fig.savefig("losses.png")

    net.eval()
    imgs_truth, vals_truth = next(iter(test_loader))
    grids_pred, imgs_pred = net(imgs_truth)

    samples = 5
    imgs = zip(
        imgs_truth[:samples].permute(0, 2, 3, 1).to('cpu').detach(),
        imgs_pred[:samples].permute(0, 2, 3, 1).to('cpu').detach()
    )

    grids = zip(
        [torch.concat([vals_truth[j][i].flatten().to('cpu').detach() for j in range(3)]) for i in range(samples)],
        [torch.concat([grids_pred[j][i].flatten().to('cpu').detach() for j in range(3)]) for i in range(samples)],
    )

    # Plot images
    fig, axs = plt.subplots(samples, 2)
    for ax, img in zip(axs, imgs):
        for i in range(2):
            ax[i].imshow(img[i], cmap=colormap, vmin=0, vmax=1)
            ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        print(f"Loss: {criterion(img[0], img[1])}")
    fig.show()
    fig.savefig("images.png")

    # Plot grid value distributions
    fig, axs = plt.subplots(samples, 2, sharex=True)
    for ax, img in zip(axs, grids):
        for i in range(2):
            ax[i].hist(img[i].detach(), 100)
            ax[i].set(xlim=(-1, 1))
    fig.show()
    fig.savefig("grid_distributions.png")

