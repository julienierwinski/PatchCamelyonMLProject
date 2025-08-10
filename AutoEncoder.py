import torch
import h5py
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import random
import os

batch_size = 512


#Data Loader
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

# 1. Define the Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        # Encoder: 96×96 → 48×48 → 24×24 → 12×12 → 6×6
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),  # → 32×48×48
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),              # → 64×24×24
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),              # →128×12×12
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),             # →256×6×6
            nn.ReLU(True),
        )
        # Decoder: 6×6 → 12×12 → 24×24 → 48×48 → 96×96
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # →128×12×12
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # → 64×24×24
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # → 32×48×48
            nn.ReLU(True),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # → 3×96×96
            nn.Tanh(),                               # for [-1,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

# 2. Setup device, model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 3. Data loading

dataset = PCamDataset("E:/PCAM/camelyonpatch_level_2_split_train_x.h5", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. Training loop
num_epochs = 60
# for epoch in range(1, num_epochs+1):
#     model.train()
#     running_loss = 0.0
#
#     for imgs in dataloader:
#         imgs = imgs.to(device)                  # (B, 3, 96, 96)
#         recon = model(imgs)                     # (B, 3, 96, 96)
#         loss  = criterion(recon, imgs)          # MSE between recon & input
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * imgs.size(0)
#
#     avg_loss = running_loss / len(dataloader.dataset)
#     print(f"Epoch {epoch:2d}/{num_epochs}, Loss: {avg_loss:.6f}")
#     torch.save(model.state_dict(), "autoencoder_weights.pth")
#     model.eval()
#     with torch.no_grad():
#         sample_imgs = next(iter(dataloader)).to(device)
#         recon_imgs = model(sample_imgs)
#
#         idx = random.sample(range(sample_imgs.size(0)), 6)
#
#         fig, axes = plt.subplots(2, 6, figsize=(18, 6), tight_layout=True)
#         for col, i in enumerate(idx):
#             orig = (sample_imgs[i].cpu().permute(1, 2, 0) + 1) / 2  # unnormalize
#             recon = (recon_imgs[i].cpu().permute(1, 2, 0) + 1) / 2
#
#             axes[0, col].imshow(orig)
#             axes[0, col].set_title(f"Orig {i}")
#             axes[0, col].axis('off')
#
#             axes[1, col].imshow(recon)
#             axes[1, col].set_title(f"Recon {i}")
#             axes[1, col].axis('off')
#
#         plt.suptitle(f"Epoch {epoch}")
#         plt.tight_layout()
#
#         os.makedirs("reconstructions", exist_ok=True)
#         out_path = f"reconstructions/recon_epoch{epoch:02d}.png"
#         fig.savefig(out_path, dpi=150, bbox_inches='tight')
#         plt.close(fig)

