device = 'cuda'
z_dim = 128
batch_size = 512

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

dataset = PCamDataset("/content/camelyonpatch_level_2_split_train_x.h5", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=6, stride=1, padding=0, bias=False)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = F.sigmoid(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_dim=100, g_channels=128, out_channels=3):
            super().__init__()

            # input Z: (z_dim, 1, 1)
            self.convT1 = nn.ConvTranspose2d(z_dim, g_channels * 8, 6, 1, 0, bias=False)  # 6x6
            self.bn1 = nn.BatchNorm2d(g_channels * 8)

            self.convT2 = nn.ConvTranspose2d(g_channels * 8, g_channels * 4, 4, 2, 1, bias=False)  # 12x12
            self.bn2 = nn.BatchNorm2d(g_channels * 4)

            self.convT3 = nn.ConvTranspose2d(g_channels * 4, g_channels * 2, 4, 2, 1, bias=False)  # 24x24
            self.bn3 = nn.BatchNorm2d(g_channels * 2)

            self.convT4 = nn.ConvTranspose2d(g_channels * 2, g_channels, 4, 2, 1, bias=False)  # 48x48
            self.bn4 = nn.BatchNorm2d(g_channels)

            self.convT5 = nn.ConvTranspose2d(g_channels, out_channels, 4, 2, 1, bias=False)  # 96x96
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


generator = Generator(z_dim= z_dim).to(device)
discriminator = Discriminator().to(device)
model = Discriminator()

criterion = nn.BCELoss()
fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device) #(batch, channels, h, w)
fixed_noise = torch.empty(batch_size, z_dim, 1, 1, device=device).uniform_(-1, 1)

real_label = 1.
fake_label = 0.

generator.apply(weights_init)
discriminator.apply(weights_init)

optimizerG = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))


img_list = []
G_losses = []
D_losses = []
best_g_loss = float('inf')
epochs = 60
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
        # noise = torch.randn(batch, z_dim, 1, 1, device=device)
        noise = torch.empty(batch, z_dim, 1, 1, device=device).uniform_(-1, 1)

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

        current_g_loss = errG.item()

        # Check if this is the best generator loss so far

        best_g_loss = current_g_loss

            # Save generator & discriminator state_dicts
        torch.save(generator.state_dict(), 'best_generator.pth')
        torch.save(discriminator.state_dict(), 'best_discriminator.pth')


        iters += 1


gen = Generator(z_dim=z_dim).to(device)
disc = Discriminator().to(device)

# 2. Load saved weights
gen.load_state_dict(torch.load('/content/best_generator.pth', map_location=device))
disc.load_state_dict(torch.load('/content/best_discriminator.pth', map_location=device))

# 3. Set to eval mode
gen.eval()
disc.eval()
fixed_noise = torch.empty(batch_size, z_dim, 1, 1, device=device).uniform_(-1, 1)
with torch.no_grad():
    fake_batch = generator(fixed_noise).detach().cpu()  # shape: (32, 3, 96, 96)
for i in range(20):
  # 2. Pick one image (e.g. index 0)
  single_img = fake_batch[i]           # shape: (3, 96, 96)

  # 3. Un-normalize from [-1,1] back to [0,1] for display
  single_img = (single_img + 1) / 2    # now in [0,1]

  # 4. Convert to H×W×C and plot
  plt.figure(figsize=(4,4))
  plt.axis("off")
  plt.title("Fake Image #0")
  plt.imshow(single_img.permute(1, 2, 0).numpy())
  plt.show()