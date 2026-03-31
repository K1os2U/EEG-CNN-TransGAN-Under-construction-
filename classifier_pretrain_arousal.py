import pickle as pkl
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
TRAIN_RATIO = 0.8
LABEL_SMOOTHING = 0.1

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_norm(x: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Robust Z-score normalization with quantile clipping."""
    ql = np.quantile(x, lower)
    qu = np.quantile(x, upper)
    x_clipped = np.clip(x, ql, qu)
    mu = x_clipped.mean()
    sigma = x_clipped.std() + 1e-6
    return (x_clipped - mu) / sigma


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """Compute accuracy, precision, recall, and F1 score."""
    preds = preds.cpu()
    labels = labels.cpu()

    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return acc, precision, recall, f1


def split_dataset(dataset: Dataset, train_ratio: float = 0.8, seed: int = 42):
    """Split dataset into train and test subsets."""
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, test_len], generator=generator)


class EEGDataset4D(Dataset):
    """Load preprocessed DEAP EEG data (raw + DE/PSD features)."""

    def __init__(self, preprocessors_results: dict, de_dir: str = './DE&PSD_Arousal_feature/'):
        de_all, raw_all, lbl_all = [], [], []

        for subject in sorted(preprocessors_results.keys()):
            raw = preprocessors_results[subject]['feature']  # (2400, 128, 9, 9)
            lbl = preprocessors_results[subject]['label']    # (2400,)
            de = loadmat(os.path.join(de_dir, f'{subject}.mat'))['data']  # (2400, 8, 9, 9)

            assert raw.shape[0] == de.shape[0] == lbl.shape[0]

            # Reshape to (trial=40, second=60, ...)
            raw = raw.reshape(40, 60, 128, 9, 9)
            de = de.reshape(40, 60, 8, 9, 9)
            lbl = lbl.reshape(40, 60)

            # Trial-level robust normalization
            for t in range(40):
                raw[t] = robust_norm(raw[t])
                de[t][:, :4] = robust_norm(de[t][:, :4])  # DE
                de[t][:, 4:] = robust_norm(de[t][:, 4:])  # PSD

            # Flatten to per-second samples
            de_all.append(de.reshape(-1, 8, 9, 9))
            raw_all.append(raw.reshape(-1, 128, 9, 9))
            lbl_all.append(lbl.reshape(-1))

        self.de_seg = np.concatenate(de_all, axis=0)
        self.raw_seg = np.concatenate(raw_all, axis=0)
        self.lbl_seg = np.concatenate(lbl_all, axis=0)

        print("[Dataset Info]")
        print(f"DE shape:   {self.de_seg.shape}")
        print(f"Raw shape:  {self.raw_seg.shape}")
        print(f"Label shape:{self.lbl_seg.shape}")
        assert len(self.de_seg) == len(self.raw_seg) == len(self.lbl_seg)

    def __len__(self):
        return len(self.de_seg)

    def __getitem__(self, idx):
        de = torch.from_numpy(self.de_seg[idx]).float()
        raw = torch.from_numpy(self.raw_seg[idx]).float()
        lbl = torch.tensor(self.lbl_seg[idx]).long()
        return de, raw, lbl

class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        b = out_channels // 3
        self.branch1 = nn.Conv2d(in_channels, b, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, b, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, b, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.branch1(x))
        x3 = self.relu(self.branch3(x))
        x5 = self.relu(self.branch5(x))
        return torch.cat([x1, x3, x5], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_map, max_map], dim=1)
        return x * self.sigmoid(self.conv(att))

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SharedBackboneClassifier(nn.Module):
    """Classifier for raw EEG signals using multi-scale CNN + attention + transformer."""
    def __init__(self, in_channels: int = 128, d_model: int = 256, num_classes: int = 2):
        super().__init__()
        self.inception = InceptionBlock(in_channels, 192)
        self.se = SEBlock(192)
        self.spatial = SpatialAttention()
        self.conv_reduce = nn.Conv2d(192, d_model, kernel_size=1)
        self.transformer = TransformerEncoder(d_model=d_model)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.inception(eeg)
        x = self.se(x)
        x = self.spatial(x)
        x = self.conv_reduce(x)  # (B, d_model, 9, 9)
        x = x.flatten(2).transpose(1, 2)  # (B, 81, d_model)
        x = self.transformer(x)
        feat = x.mean(dim=1)  # Global average pooling
        return self.classifier_head(feat)


def classifier_pretrain(
    model,
    dataset,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    train_ratio: float = TRAIN_RATIO,
    label_smoothing: float = LABEL_SMOOTHING,
    seed: int = SEED,
    device: str = DEVICE
):
    """Train and evaluate the pre-classifier with train/test split."""
    set_seed(seed)
    train_set, test_set = split_dataset(dataset, train_ratio, seed)

    # Use num_workers=0 for Windows compatibility; set >0 on Linux/macOS if needed
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for _, (_, raw_eeg, labels) in enumerate(train_loader):
            raw_eeg, labels = raw_eeg.to(device), labels.to(device)
            logits = model(raw_eeg)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for _, (_, raw_eeg, labels) in enumerate(test_loader):
                raw_eeg, labels = raw_eeg.to(device), labels.to(device)
                preds = model(raw_eeg).argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)

        print(
            f"[Epoch {epoch+1:03d}/{epochs}] "
            f"Loss={train_loss/len(train_loader):.4f} | "
            f"Acc={acc*100:.2f}% | "
            f"Precision={prec*100:.2f}% | "
            f"Recall={rec*100:.2f}% | "
            f"F1={f1*100:.2f}%"
        )

    torch.save(model.state_dict(), "Classifier_Pretrain_BinaryClass_Arousal.pth")
    print(" Pre-classifier training finished.")

if __name__ == '__main__':
    # Disable Flash Attention for compatibility
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Load preprocessed data
    with open('./dataset/deap_binary_Arousal_dataset.pkl', 'rb') as f:
        preprocessors_results = pkl.load(f)
    # Build dataset and model
    dataset = EEGDataset4D(preprocessors_results)
    model = SharedBackboneClassifier(in_channels=128, d_model=256, num_classes=2)

    # Train
    classifier_pretrain(
        model=model,
        dataset=dataset,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        seed=SEED,
        device=DEVICE
    )
