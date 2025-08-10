from AutoEncoder import ConvAutoencoder
import torch
import h5py
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

class PCamDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_path    = x_path
        self.y_path    = y_path
        self.transform = transform
        with h5py.File(self.y_path, 'r') as f:
            self.length = len(f['y'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.x_path, 'r') as fx, h5py.File(self.y_path, 'r') as fy:
            img   = fx['x'][idx]               # shape: (96,96,3)
            label = fy['y'][idx]               # 0-d array

        if self.transform:
            img = self.transform(img)

        # extract a Python int from the 0-dim array
        label = int(label.item())

        return img, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),                # → [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3) # → [–1,1]
])
if __name__ == "__main__":
    dataset = PCamDataset(
        x_path="E:/PCAM/camelyonpatch_level_2_split_train_x.h5",
        y_path="E:/PCAM/camelyonpatch_level_2_split_train_y.h5",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=512)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Re-instantiate your autoencoder so you get its encoder architecture back
    ae = ConvAutoencoder(img_channels=3).to(device)

    full_sd = torch.load("E:/PCAM/autoencoder_weights.pth", map_location=device)

    # 3. Extract only the encoder parameters, stripping the "encoder." prefix
    encoder_sd = OrderedDict()
    for k, v in full_sd.items():
        if k.startswith("encoder."):
            new_key = k.replace("encoder.", "", 1)
            encoder_sd[new_key] = v

    # 4. Load into your Sequential encoder
    ae.encoder.load_state_dict(encoder_sd)


    # 3. Freeze the encoder
    for p in ae.encoder.parameters():
        p.requires_grad = False


    class EncoderClassifier(nn.Module):
        def __init__(self, encoder, num_classes=2):
            super().__init__()
            self.encoder = encoder
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256*6*6, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            with torch.no_grad():
                feats = self.encoder(x)
            feats = self.flatten(feats)
            feats = self.fc1(feats)
            feats = self.fc2(feats)
            return feats


    classifier = EncoderClassifier(ae.encoder, num_classes=2).to(device)

    optimizer = optim.Adam(
        list(classifier.fc1.parameters()) + list(classifier.fc2.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    test_ds = PCamDataset(
        x_path="E:/PCAM/camelyonpatch_level_2_split_test_x.h5",
        y_path="E:/PCAM/camelyonpatch_level_2_split_test_y.h5",
        transform=transform
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    num_epochs = 50
    recall = 0.0
    for epoch in range(1, num_epochs+1):
        # ——— Training ———
        classifier.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds  = []
        all_labels = []

        for imgs, labels in dataloader:
            imgs   = imgs.to(device)    # (B, 3, 96, 96)
            labels = labels.to(device)  # (B,)

            logits = classifier(imgs)          # (B, 2)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            running_loss += loss.item() * imgs.size(0)

            # accumulate correct predictions
            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc  = running_corrects / len(dataset)

        print(f"[Train] Epoch {epoch:2d}/{num_epochs}  "
              f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = classifier(imgs)  # (B,2)
                preds = torch.argmax(logits, dim=1)  # (B,)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # concatenate batches
        all_preds = np.concatenate(all_preds)  # (N_test,)
        all_labels = np.concatenate(all_labels)  # (N_test,)
        recall_epoch = recall_score(all_labels, all_preds, average='binary')
        print(f"Recall: {recall_epoch:.4f}")
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy:.4f}")
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix (rows=true class, cols=predicted class):")
        print(cm)
        if recall_epoch > recall:
            recall = recall_epoch
            torch.save(classifier.state_dict(), "fully_connected_autoencoder_weights.pth")
            print('model saved!')






    # classifier = EncoderClassifier(ae.encoder, num_classes=2).to(device)
    # classifier.load_state_dict(torch.load("E:/PCAM/fully_connected_autoencoder_weights.pth", map_location=device))
    # classifier.eval()
    #
    # all_preds  = []
    # all_labels = []
    #
    # with torch.no_grad():
    #     for imgs, labels in test_loader:
    #         imgs   = imgs.to(device)
    #         labels = labels.to(device)
    #
    #         logits = classifier(imgs)              # (B,2)
    #         preds  = torch.argmax(logits, dim=1)   # (B,)
    #
    #         all_preds.append(preds.cpu().numpy())
    #         all_labels.append(labels.cpu().numpy())
    #
    # # concatenate batches
    # all_preds  = np.concatenate(all_preds)    # (N_test,)
    # all_labels = np.concatenate(all_labels)   # (N_test,)
    #
    # from sklearn.metrics import accuracy_score
    # accuracy = accuracy_score(all_labels, all_preds)
    # print(f"Test Accuracy: {accuracy:.4f}")
    #
    # # compute confusion matrix
    # cm = confusion_matrix(all_labels, all_preds)
    # print("Confusion Matrix (rows=true class, cols=predicted class):")
    # print(cm)
    #
    # from sklearn.metrics import recall_score
    #
    # # after you’ve built all_preds and all_labels:
    # recall = recall_score(all_labels, all_preds, average='binary')
    # print(f"Recall: {recall:.4f}")