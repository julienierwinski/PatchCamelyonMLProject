from AutoEncoder import ConvAutoencoder
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from AutoEncoderFC import PCamDataset
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvAutoencoder(img_channels=3).to(device)

model.load_state_dict(torch.load(r"E:\PCAM\autoencoder_weights.pth", map_location=device))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),                # → [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3) # → [–1,1]
])

dataset = PCamDataset(
    x_path="E:/PCAM/camelyonpatch_level_2_split_test_x.h5",
    y_path="E:/PCAM/camelyonpatch_level_2_split_test_y.h5",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=512)

# 1. Switch to eval mode & disable grads
model.eval()
features = []
labels   = []

with torch.no_grad():
    for imgs, lbls in dataloader:       # assumes PCamDataset returns (img, label)
        imgs = imgs.to(device)
        # 2. Extract encoder features: (B, 256, 6, 6)
        z = model.encoder(imgs)
        # 3. Flatten: (B, 256*6*6)
        z = z.view(z.size(0), -1).cpu().numpy()
        features.append(z)
        labels.append(lbls.numpy())
# 4. Stack into big arrays
features = np.vstack(features)          # shape: (N, 9216)
labels   = np.concatenate(labels)       # shape: (N,)
df = pd.DataFrame(features)
df['label'] = labels

# Save to CSV
df.to_csv("pcam_test_encoder_features.csv", index=False)

# 4) Fit logistic regression
# clf = LogisticRegression(
#     penalty='l2',
#     C=1.0,
#     solver='saga',    # handles large data, sparse penalties
#     max_iter=1000,
#     n_jobs=-1
# )
# clf.fit(features, labels)
#
#
#
#
# ##test##
# test_ds = PCamDataset(
#     x_path="E:/PCAM/camelyonpatch_level_2_split_test_x.h5",
#     y_path="E:/PCAM/camelyonpatch_level_2_split_test_y.h5",
#     transform=transform
# )
#
# test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
# model.eval()
# features_test = []
# labels_test   = []
#
# with torch.no_grad():
#     for imgs, lbls in test_loader:       # assumes PCamDataset returns (img, label)
#         imgs = imgs.to(device)
#         # 2. Extract encoder features: (B, 256, 6, 6)
#         z = model.encoder(imgs)
#         # 3. Flatten: (B, 256*6*6)
#         z = z.view(z.size(0), -1).cpu().numpy()
#         features_test.append(z)
#         labels_test.append(lbls.numpy())
#
# # 4. Stack into big arrays
# features_test = np.vstack(features_test)          # shape: (N, 9216)
# labels_test   = np.concatenate(labels_test)       # shape: (N,)
#
# # 5) Evaluate
# y_pred = clf.predict(features_test)
# acc    = accuracy_score(labels_test, y_pred)
# cm     = confusion_matrix(labels_test, y_pred)
#
# print(f"Test Accuracy: {acc:.4f}")
# print("Confusion Matrix:\n", cm)

# # 5. PCA → 2D
# pca = PCA(n_components=2)
# feat2 = pca.fit_transform(features)     # shape: (N, 2)
#
# # 6. Scatter plot
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     feat2[:, 0], feat2[:, 1],
#     c=labels,        # color by class (0 vs. 1)
#     cmap='coolwarm',
#     alpha=0.7,
#     s=10
# )
# plt.legend(*scatter.legend_elements(), title="Class")
# plt.title("PCA of Encoder Features")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.show()
