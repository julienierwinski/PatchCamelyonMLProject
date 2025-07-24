import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision.utils as vutils

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
device = 'cuda'
print(torch.__version__)
print(torch.cuda.is_available())              # should be True
print(torch.version.cuda)                     # should be '11.8'
print(torch.cuda.get_device_name(0))          # GPU name

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
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8*24*24, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_dim=100, g_channels=64, out_channels=3):
            super().__init__()

            # input Z: (z_dim, 1, 1)
            self.convT1 = nn.ConvTranspose2d(z_dim, g_channels * 8, 6, 1, 0, bias=False)  # 6x6
            self.bn1 = nn.BatchNorm2d(g_channels * 8)

            self.convT2 = nn.ConvTranspose2d(g_channels * 8, g_channels * 4, 4, 2, 1, bias=False)  # 6x6
            self.bn2 = nn.BatchNorm2d(g_channels * 4)

            self.convT3 = nn.ConvTranspose2d(g_channels * 4, g_channels * 2, 4, 2, 1, bias=False)  # 6x6
            self.bn3 = nn.BatchNorm2d(g_channels * 2)

            self.convT4 = nn.ConvTranspose2d(g_channels * 2, g_channels, 4, 2, 1, bias=False)  # 6x6
            self.bn4 = nn.BatchNorm2d(g_channels)

            self.convT5 = nn.ConvTranspose2d(g_channels, out_channels, 4, 2, 1, bias=False)  # 6x6
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()

    def forward(self, z):
            z = self.convT1(z)
            z = self.bn1(z)
            z = self.relu(z)
            z = self.convT2(z)
            z = self.bn2(z)
            z = self.relu(z)
            z = self.convT3(z)
            z = self.bn3(z)
            z = self.relu(z)
            z = self.convT4(z)
            z = self.bn4(z)
            z = self.relu(z)
            z = self.convT5(z)
            z = self.tanh(z)
            return z


generator = Generator().to(device)
discriminator = Discriminator().to(device)
model = Discriminator()

criterion = nn.BCELoss()
fixed_noise = torch.randn(32, 100, 1, 1, device=device) #(batch, channels, h, w)

real_label = 1.
fake_label = 0.

generator.apply(weights_init)
discriminator.apply(weights_init)

optimizerG = optim.Adam(generator.parameters(), lr=0.1, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.1, betas=(0.5, 0.999))


img_list = []
G_losses = []
D_losses = []
epochs = 10
iters = 0
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        ## Train using real batch
        discriminator.zero_grad()
        real_cpu = data.to(device)
        batch = real_cpu.size(0)
        label = torch.full((batch,), real_label, dtype=torch.float, device=device)
        output = discriminator(real_cpu).view(-1)
        err_real = criterion(output, label)
        err_real.backward()
        D_x = output.mean().item()

        ## Train using fake batch
        noise = torch.randn(batch, 100, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake.detach()).view(-1)
        err_fake = criterion(output, label)
        err_fake.backward()
        D_G_z1 = output.mean().item()
        errD = err_real + err_fake
        optimizerD.step()

        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1