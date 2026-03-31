import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation
from scipy.io import loadmat
from pathlib import Path
import os
from torch.utils.data import random_split
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    NUM_EPOCHS = 100
    BATCH_SIZE  = 64
    C_LR        = 1e-5
    WD          = 5e-4
    TEST_SIZE   = 0.2
    FOLD = 4



DATASET_BASE_DIR = Path('./eeg_dataset')
DATASET_FOLD_DIR = DATASET_BASE_DIR / 'DEAP'
PREPROCESSED_EEG_DIR = DATASET_FOLD_DIR / 'data_preprocessed_python'

label_preprocessors = {'label': Sequence([BinaryLabel()])}
feature_preprocessors = {
    'feature':
    Sequence([Raw2TNCF(),
              RemoveBaseline(),
              TNCF2NCF(),
              ChannelToLocation()])
}

preprocessors_results = DEAPDataset(
    PREPROCESSED_EEG_DIR, label_preprocessors,
    feature_preprocessors)('./dataset/deap_binary_valence_dataset.pkl')

def robust_norm(x, lower=0.01, upper=0.99):
    """
    Robust Z-score normalization for numpy arrays
    x: np.ndarray (any shape)
    """
    ql = np.quantile(x, lower)
    qu = np.quantile(x, upper)

    x = np.clip(x, ql, qu)

    mu = x.mean()
    sigma = x.std() + 1e-6

    return (x - mu) / sigma

class EEGDataset4D(Dataset):
    def __init__(self, preprocessors_results, de_dir='./DE&PSD_feature/'):
        de_all, raw_all, lbl_all = [], [], []

        for subject in sorted(preprocessors_results.keys()):
            # raw: (2400, 128, 9, 9)
            raw = preprocessors_results[subject]['feature']
            lbl = preprocessors_results[subject]['label']

            de_path = os.path.join(de_dir, f'{subject}.mat')
            de = loadmat(de_path)['data']   # (2400, 8, 9, 9)

            assert raw.shape[0] == de.shape[0] == lbl.shape[0]

            raw = raw.reshape(40, 60, 128, 9, 9)
            de  = de.reshape(40, 60, 8, 9, 9)
            lbl = lbl.reshape(40, 60)

            for t in range(40):
                # ===== raw EEG =====
                raw[t] = robust_norm(raw[t])

                # ===== DE / PSD =====
                de_trial = de[t]

                de_part  = de_trial[:, :4]   # (60,4,9,9)
                psd_part = de_trial[:, 4:]   # (60,4,9,9)

                de_trial[:, :4] = robust_norm(de_part)
                de_trial[:, 4:] = robust_norm(psd_part)

                de[t] = de_trial

            de_all.append(de.reshape(-1, 8, 9, 9))
            raw_all.append(raw.reshape(-1, 128, 9, 9))
            lbl_all.append(lbl.reshape(-1))

        self.de_seg  = np.concatenate(de_all, axis=0)
        self.raw_seg = np.concatenate(raw_all, axis=0)
        self.lbl_seg = np.concatenate(lbl_all, axis=0)

        print('[Dataset]')
        print('DE:', self.de_seg.shape)
        print('RAW:', self.raw_seg.shape)
        print('LBL:', self.lbl_seg.shape)

        assert len(self.de_seg) == len(self.raw_seg) == len(self.lbl_seg)

    def __len__(self):
        return len(self.de_seg)

    def __getitem__(self, idx):
        de  = torch.from_numpy(self.de_seg[idx]).float()
        raw = torch.from_numpy(self.raw_seg[idx]).float()
        lbl = torch.tensor(self.lbl_seg[idx]).long()
        return de, raw, lbl

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        b = out_channels // 3

        self.branch1 = nn.Conv2d(in_channels, b, kernel_size=1, padding=0)

        self.branch3 = nn.Conv2d(in_channels, b, kernel_size=3, padding=1)

        self.branch5 = nn.Conv2d(in_channels, b, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.branch1(x))
        x3 = self.relu(self.branch3(x))
        x5 = self.relu(self.branch5(x))
        return torch.cat([x1, x3, x5], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_map, max_map], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class SharedBackboneClassifier(nn.Module):
    def __init__(self, in_channels=128, d_model=256, num_classes=2):
        super().__init__()

        self.inception = InceptionBlock(
            in_channels=in_channels,
            out_channels=192
        )

        self.se = SEBlock(192)
        self.spatial = SpatialAttention()

        self.conv_reduce = nn.Conv2d(192, d_model, kernel_size=1)

        self.transformer = TransformerEncoder(d_model=d_model)

        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg):
        """
        eeg: (B, 128, 9, 9)
        """
        x = self.inception(eeg)
        x = self.se(x)
        x = self.spatial(x)

        x = self.conv_reduce(x)          # (B, d_model, 9, 9)
        x = x.flatten(2).transpose(1, 2) # (B, 81, d_model)

        x = self.transformer(x)
        feat = x.mean(dim=1)             # Global token pooling

        cls_out = self.classifier_head(feat)
        return cls_out

def split_dataset(dataset, train_ratio=0.8, seed=42):
    length = len(dataset)
    train_len = int(length * train_ratio)
    test_len = length - train_len

    generator = torch.Generator().manual_seed(seed)

    train_set, test_set = random_split(
        dataset,
        [train_len, test_len],
        generator=generator
    )

    return train_set, test_set

def compute_metrics(preds, labels):
    """
    preds, labels: torch.Tensor (N,)
    """
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

    return acc, precision, f1

def train_preclassifier_with_eval(
    model,
    dataset,
    epochs=80,
    batch_size=64,
    lr=3e-4,
    seed=42,
    device='cuda'
):
    set_seed(seed)
    train_set, test_set = split_dataset(dataset, 0.8, seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0    
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for _, (_, raw_eeg, labels) in enumerate(train_loader):
            raw_eeg = raw_eeg.to(device)
            labels = labels.to(device)

            logits = model(raw_eeg)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Test / Evaluation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for _, (_, raw_eeg, labels) in enumerate(test_loader):
                raw_eeg = raw_eeg.to(device)
                labels = labels.to(device)

                logits = model(raw_eeg)
                preds = logits.argmax(dim=1)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc, precision, f1 = compute_metrics(all_preds, all_labels)

        print(
            f"[Epoch {epoch+1:03d}/{epochs}] "
            f"TrainLoss={train_loss/len(train_loader):.4f} | "
            f"Acc={acc*100:.2f}% | "
            f"Prec={precision*100:.2f}% | "
            f"F1={f1*100:.2f}%"
        )

    torch.save(model.state_dict(),"preclassifier_raw_eeg_with_split.pth")

    print("Pre-classifier training + evaluation finished.")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
set_seed(SEED)

dataset = EEGDataset4D(preprocessors_results)

model = SharedBackboneClassifier(
    in_channels=128,
    d_model=256,
    num_classes=2
)

train_preclassifier_with_eval(
    model,
    dataset,
    epochs=100,
    batch_size=64,
    seed=SEED,
    device=DEVICE
)
