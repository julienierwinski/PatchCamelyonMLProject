import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
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


