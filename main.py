import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# # Open test images
# with h5py.File('camelyonpatch_level_2_split_test_x.h5', 'r') as f:
#     print(list(f.keys()))   # Should show one key, like ['x']
#     x_test = f['x'][:]      # Load the image array
#
# print(f"x_test shape: {x_test.shape}")  # Expected: (num_images, 96, 96, 3)
#
# plt.imshow(x_test[150])  # RGB image
# plt.axis("off")
# plt.show()


class PCamDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_file = h5py.File(h5_path, 'r')
        self.images = self.h5_file['x']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape: (H, W, C) usually

        if self.transform:
            img = self.transform(img)

        return img


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = PCamDataset("camelyonpatch_level_2_split_test_x.h5", transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8*24*24, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        print(x.shape)
        x = x.flatten()
        print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        print(x.shape)
        x = F.softmax(x, dim=0)

        return x





model = Discriminator()
x = dataset.__getitem__(10)
model.forward(x)