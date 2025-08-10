
class PCamTestDataset(Dataset):
    def __init__(self,
                 img_h5_path: str,
                 lbl_h5_path: str,
                 transform=None):
        # open both H5 files
        self.img_h5 = h5py.File(img_h5_path, 'r')
        self.lbl_h5 = h5py.File(lbl_h5_path, 'r')

        # assume datasets are named "x" and "y"
        self.images = self.img_h5['x']    # shape: (N, H, W, C)
        self.labels = self.lbl_h5['y']    # shape: (N,)

        assert len(self.images) == len(self.labels), \
            "Image and label counts must match"

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]            # e.g. numpy array H×W×C
        if self.transform:
            img = self.transform(img)

        lbl = float(self.labels[idx])     # 0.0 or 1.0
        return img, torch.tensor(lbl)


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_ds = PCamTestDataset(
    img_h5_path = "/content/camelyonpatch_level_2_split_train_x.h5",
    lbl_h5_path = "/content/camelyonpatch_level_2_split_train_y.h5",
    transform     = test_transform
)

valid_ds = PCamTestDataset(
    img_h5_path = "/content/camelyonpatch_level_2_split_valid_x.h5",
    lbl_h5_path = "/content/camelyonpatch_level_2_split_valid_y.h5",
    transform     = test_transform
    )

train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)

class DiscriminatorClassifier(nn.Module):
    def __init__(self, pretrained_disc: nn.Module):
        super().__init__()
        # --- encoder = all layers except the final conv5+sigmoid ---
        self.encoder = nn.Sequential(
            pretrained_disc.conv1,
            pretrained_disc.relu,
            pretrained_disc.conv2,
            pretrained_disc.bn2,
            pretrained_disc.relu,
            pretrained_disc.conv3,
            pretrained_disc.bn3,
            pretrained_disc.relu,
            pretrained_disc.conv4,
            pretrained_disc.bn4,
            pretrained_disc.relu,
        )
        # --- new two‑class head: 1×1 conv → global pool → flatten ---
        self.class_head = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        features = self.encoder(x)           # B×512×6×6
        logits   = self.class_head(features) # B×2
        return logits

# 2) Instantiate and freeze the backbone
base_disc = Discriminator().to(device)
base_disc.load_state_dict(torch.load('/content/best_discriminator_healthy.pth', map_location=device))
model = DiscriminatorClassifier(base_disc).to(device)

# Freeze encoder
for p in model.encoder.parameters():
    p.requires_grad = False

# 3) Set up optimizer & loss for head only
optimizer_head = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-3,
    betas=(0.5, 0.999)
)
criterion = nn.CrossEntropyLoss()
n_epochs = 20
# 4) Train just the head for N_head_epochs
model.train()
for epoch in range(n_epochs):
    epoch_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device).long()
        optimizer_head.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer_head.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs} - Head Training Loss: {avg_loss:.4f}")

    valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False)

    test_ds = PCamTestDataset(
        img_h5_path="/content/camelyonpatch_level_2_split_test_x.h5",
        lbl_h5_path="/content/camelyonpatch_level_2_split_test_y.h5",
        transform=test_transform
    )

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model_1.eval()
    model_2.eval()
    correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            logits = model_1(imgs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Val Accuracy: {100 * correct / total:.2f}%")

    # Compute and print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Optional: detailed classification report (precision, recall, F1)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Healthy", "Cancerous"]))

    correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            logits = model_2(imgs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Val Accuracy: {100 * correct / total:.2f}%")

    # Compute and print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Optional: detailed classification report (precision, recall, F1)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Healthy", "Cancerous"]))
    from sklearn.metrics import confusion_matrix, classification_report

    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Val Accuracy: {100 * correct / total:.2f}%")
    torch.save(model.state_dict(), 'finetuned_discriminator_healthy.pth')


test_ds = PCamTestDataset(
    img_h5_path = "/content/camelyonpatch_level_2_split_test_x.h5",
    lbl_h5_path = "/content/camelyonpatch_level_2_split_test_y.h5",
    transform     = test_transform
    )

test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
model_1.eval()
model_2.eval()
correct = total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device).long()
        logits_1 = model_1(imgs)
        logits_2 = model_2(imgs)
        probs1 = torch.softmax(logits_1, dim=1)
        probs2 = torch.softmax(logits_2, dim=1)
        probs  = (probs1 + probs2) / 2
        preds  = probs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Val Accuracy: {100 * correct / total:.2f}%")

# Compute and print confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Optional: detailed classification report (precision, recall, F1)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Healthy", "Cancerous"]))