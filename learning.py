"""
Neural network which tries to retrieve Perlin gradients from a final planet image
"""
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import read_image
import torchsummary
import matplotlib.pyplot as plt
from matplotlib import cm
from planet_generation_perlin import grids_to_planet, split_planet_into_faces
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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True, num_workers=2 if device == 'cpu' else 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2 if device == 'cpu' else 0)
print(f"Dataset: {len(train_data)} train, {len(test_data)} test, {len(all_data)} total")


#%% Setup network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_images = True
        self.process_width = 64  # Width of planet faces during processing
        self.activ = nn.ReLU()
        self.outActiv = nn.Tanh()  # Output needs to be in [-1, 1]
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # Down-path
        self.downConv0 = nn.Conv2d(1, 32, 3, padding=(1, 1))
        self.batchNorm0 = nn.BatchNorm2d(32)
        self.acrossConv0a = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.batchNorm0a = nn.BatchNorm2d(32)
        self.acrossConv0b = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.batchNorm0b = nn.BatchNorm2d(32)
        self.downConv1 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.acrossConv1a = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.batchNorm1a = nn.BatchNorm2d(64)
        self.acrossConv1b = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.batchNorm1b = nn.BatchNorm2d(64)
        self.downConv2 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.acrossConv2a = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.batchNorm2a = nn.BatchNorm2d(128)
        self.acrossConv2b = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.batchNorm2b = nn.BatchNorm2d(128)
        self.downConv3 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        # Across
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.acrossConv3a = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.batchNorm3a = nn.BatchNorm2d(256)
        self.acrossConv3b = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.batchNorm3b = nn.BatchNorm2d(256)
        # Up
        self.upConv0 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.batchNorm4 = nn.BatchNorm2d(128)
        self.acrossConv4a = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.batchNorm4a = nn.BatchNorm2d(256)
        self.acrossConv4b = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.batchNorm4b = nn.BatchNorm2d(256)
        self.upConv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.batchNorm5 = nn.BatchNorm2d(64)
        self.acrossConv5a = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.batchNorm5a = nn.BatchNorm2d(128)
        self.acrossConv5b = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.batchNorm5b = nn.BatchNorm2d(128)
        self.upConv2 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        self.batchNorm6 = nn.BatchNorm2d(32)
        self.acrossConv6a = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.batchNorm6a = nn.BatchNorm2d(64)
        self.acrossConv6b = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.batchNorm6b = nn.BatchNorm2d(64)
        self.upConv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.batchNorm7 = nn.BatchNorm2d(32)
        self.acrossConv7a = nn.Conv2d(33, 33, 3, padding=(1, 1))
        self.batchNorm7a = nn.BatchNorm2d(33)
        self.acrossConv7b = nn.Conv2d(33, 33, 3, padding=(1, 1))
        self.batchNorm7b = nn.BatchNorm2d(33)
        # Out
        # This last layer removes the padding that was introduced when planets were split into faces
        self.outConv0 = nn.Conv2d(33, len(layers) * 3, 3)

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
        batch_size = x.shape[0]
        # Split into faces, and flatten faces dimension into batch dimension
        x = split_planet_into_faces(x, self.process_width - 2, padding=1).flatten(0, 1)

        l0 = x
        # 64x64x1
        l1 = self.pool(self.activ(self.batchNorm0(self.downConv0(l0))))
        l1 = self.activ(self.batchNorm0a(self.acrossConv0a(l1)))
        l1 = self.activ(self.batchNorm0b(self.acrossConv0b(l1)))
        # 32x32x64
        l2 = self.pool(self.activ(self.batchNorm1(self.downConv1(l1))))
        l2 = self.activ(self.batchNorm1a(self.acrossConv1a(l2)))
        l2 = self.activ(self.batchNorm1b(self.acrossConv1b(l2)))
        # 16x16x128
        l3 = self.pool(self.activ(self.batchNorm2(self.downConv2(l2))))
        l3 = self.activ(self.batchNorm2a(self.acrossConv2a(l3)))
        l3 = self.activ(self.batchNorm2b(self.acrossConv2b(l3)))
        # 8x8x256
        x = self.pool(self.activ(self.batchNorm3(self.downConv3(l3))))
        x = self.activ(self.batchNorm3a(self.acrossConv3a(x)))
        x = self.activ(self.batchNorm3b(self.acrossConv3b(x)))
        # 4x4x512
        x = torch.concat((l3, self.activ(self.batchNorm4(self.upConv0(x)))), dim=1)
        x = self.activ(self.batchNorm4a(self.acrossConv4a(x)))
        x = self.activ(self.batchNorm4b(self.acrossConv4b(x)))
        # 8x8x512
        x = torch.concat((l2, self.activ(self.batchNorm5(self.upConv1(x)))), dim=1)
        x = self.activ(self.batchNorm5a(self.acrossConv5a(x)))
        x = self.activ(self.batchNorm5b(self.acrossConv5b(x)))
        # 16x16x256
        x = torch.concat((l1, self.activ(self.batchNorm6(self.upConv2(x)))), dim=1)
        x = self.activ(self.batchNorm6a(self.acrossConv6a(x)))
        x = self.activ(self.batchNorm6b(self.acrossConv6b(x)))
        # 32x32x128
        x = torch.concat((l0, self.activ(self.batchNorm7(self.upConv3(x)))), dim=1)
        x = self.activ(self.batchNorm7a(self.acrossConv7a(x)))
        x = self.activ(self.batchNorm7b(self.acrossConv7b(x)))
        # 64x64x65
        x = self.outActiv(self.outConv0(x))
        # 62x62x64

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
torchsummary.summary(net, (1, img_size[1], img_size[0]))

#%% Load saved weights
if os.path.isfile(model_bkup_path):
    net.load_state_dict(torch.load(model_bkup_path, map_location=device))

#%% Train
# torch.autograd.set_detect_anomaly(True)
print("Setup")
net.train()
criterion = nn.MSELoss()

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.1)
# optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.9)

try:
    print("Training")
    for epoch in range(2):  # loop over the dataset multiple times
        print(f"Epoch {epoch}")
        running_loss = 0.0
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
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            if i % 20 == 19:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.8f}')
                running_loss = 0.0
    torch.save(net.state_dict(), model_bkup_path)
    print("Done")
except KeyboardInterrupt:
    print("\n")
    while True:
        try:
            choice = input("Save? (Y/n) ")
            if choice in ['y', 'Y']:
                torch.save(net.state_dict(), model_bkup_path)
                break
            if choice in ['n', 'N']:
                break
        except KeyboardInterrupt:
            pass
        print("Invalid choice")

#%% Result
net.eval()
imgs_truth, vals_truth = next(iter(test_loader))
grids_pred, imgs_pred = net(imgs_truth)

print([(grid.min(), grid.max()) for grid in vals_truth])
print([(grid.min().item(), grid.max().item()) for grid in grids_pred])

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
fig.show()

# Plot grid value distributions
fig, axs = plt.subplots(samples, 2)
for ax, img in zip(axs, grids):
    for i in range(2):
        ax[i].hist(img[i].detach(), 100)
        ax[i].set(xlim=(-1, 1))
fig.show()
