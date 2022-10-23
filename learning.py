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
        img = read_image(os.path.join(self.path, name + '.png')).to(device).float()
        grids = [x.to(device) for x in torch.load(os.path.join(self.path, name + '.pt'))]
        return img, grids


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
        self.process_width = 64  # Width of planet faces during processing
        self.activ = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # Down-path
        self.downConv0 = nn.Conv2d(1, 64, 3, padding=(1, 1))
        self.downConv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.downConv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.downConv3 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        # Across
        self.acrossConv0 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        # Up
        self.upConv0 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.upConv1 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.upConv2 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.upConv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # Out
        self.outConv0 = nn.Conv2d(65, len(layers) * 3, 3, padding=(1, 1))

        pool_sizes = [
            int(self.process_width / (layer + 1))
            for layer in layers
        ]
        self.grid_pools = [
            nn.AvgPool3d((1, pool_sizes[i], pool_sizes[i]))
            for i, layer in enumerate(layers)
        ]

    def forward(self, x):
        batch_size = x.shape[0]
        # Split into faces, and flatten faces dimension into batch dimension
        x = split_planet_into_faces(x, self.process_width - 2, padding=1).flatten(0, 1)

        l0 = x
        # 64x64x1
        l1 = self.pool(self.activ(self.downConv0(l0)))
        # 32x32x64
        l2 = self.pool(self.activ(self.downConv1(l1)))
        # 16x16x128
        l3 = self.pool(self.activ(self.downConv2(l2)))
        # 8x8x256
        l4 = self.pool(self.activ(self.downConv3(l3)))
        # 4x4x256
        x = self.activ(self.acrossConv0(l4))
        # 4x4x256
        x = torch.concat((l3, self.activ(self.upConv0(x))), dim=1)
        # 8x8x512
        x = torch.concat((l2, self.activ(self.upConv1(x))), dim=1)
        # 16x16x256
        x = torch.concat((l1, self.activ(self.upConv2(x))), dim=1)
        # 32x32x128
        x = torch.concat((l0, self.activ(self.upConv3(x))), dim=1)
        # 64x64x65
        x = self.activ(self.outConv0(x))
        # 64x64x64

        # Unflatten to retrieve faces dimension
        x = x.unflatten(0, (batch_size, 6))
        # Pool outputs into required sizes
        grids = [
            self.grid_pools[i](x[:, :, i*3:i*3+3, :, :])
            for i, layer in enumerate(layers)
        ]
        # Generate planet images from grids
        x = grids_to_planet(grids, batch_size, img_size)

        return grids, x[:, None, :, :]


net = Network().to(device)
torchsummary.summary(net, (1, img_size[1], img_size[0]))

#%% Train
# torch.autograd.set_detect_anomaly(True)
print("Setup")
net.train()
criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.1)

print("Training")
for epoch in range(2):  # loop over the dataset multiple times
    print(f"Epoch {epoch}")
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # optimizer.zero_grad()
        output_grids, output_imgs = net(inputs)
        loss = criterion(output_imgs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 2000 == 1999:  # print every 2000 mini-batches
        if i % 20 == 19:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            running_loss = 0.0
print("Done")

#%% Save model weights
torch.save(net.state_dict(), f"model_bkup.pt")

#%% Result
net.eval()
imgs_truth, vals_truth = next(iter(test_loader))
grids_pred, imgs_pred = net(imgs_truth)

samples = 5
imgs = zip(
    imgs_truth[:samples].permute(0, 2, 3, 1).detach(),
    imgs_pred[:samples].permute(0, 2, 3, 1).detach()
)

fig, axs = plt.subplots(samples, 2)
for ax, img in zip(axs, imgs):
    for i in range(2):
        ax[i].imshow(img[i])
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
fig.show()
