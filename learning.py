"""
Neural network which tries to retrieve Perlin gradients from a final planet image
"""
#%% Imports
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

    def forward(self, x):
        l0 = x
        # 128x64
        l1 = self.pool(self.activ(self.downConv0(l0)))
        # 64x32x64
        l2 = self.pool(self.activ(self.downConv1(l1)))
        # 32x16x128
        l3 = self.pool(self.activ(self.downConv2(l2)))
        # 16x8x128
        l4 = self.pool(self.activ(self.downConv3(l3)))
        # 8x4x128
        x = self.acrossConv0(l4)
        # 8x4x128
        x = torch.concat((l3, self.activ(self.upConv0(x))), dim=1)
        # 16x8x256
        x = torch.concat((l2, self.activ(self.upConv1(x))), dim=1)
        # 32x16x256
        x = torch.concat((l1, self.activ(self.upConv2(x))), dim=1)
        # 64x32x128
        x = torch.concat((l0, self.activ(self.upConv3(x))), dim=1)
        # 128x64x65
        x = self.activ(self.outConv0(x))
        # 128x64x64

        #TODO This all needs to be vectorized for GPU performance
        batch_grids = []
        imgs = []
        for i in range(x.shape[0]):
            grids = []
            for j, width in enumerate(layers):
                grids.append(
                    torch.stack([
                        split_planet_into_faces(x[i, j*3 + k], width + 1)
                        for k in range(3)
                    ], dim=-1)
                )
            batch_grids.append(grids)
            imgs.append(grids_to_planet(grids, img_size)[None, :, :])
        imgs = torch.stack(imgs)

        return batch_grids, imgs


net = Network().to(device)
torchsummary.summary(net, (1, img_size[1], img_size[0]))

#%% Train
# torch.autograd.set_detect_anomaly(True)
print("Setup")
net.train()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
