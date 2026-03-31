import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation
import logging
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import random
from scipy.io import loadmat
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, precision_score, f1_score

RECEIVED_PARAMS = {
    "c_lr": 0.00001,
    "weight_ssl": 0.5,
}

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

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

            # ---------- reshape to (trial, second, ...) ----------
            raw = raw.reshape(40, 60, 128, 9, 9)
            de  = de.reshape(40, 60, 8, 9, 9)
            lbl = lbl.reshape(40, 60)

            # ---------- trial-level normalization ----------
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

            # ---------- flatten back to second ----------
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
# -------------------------
# Positional Encoding
# -------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, E)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L]


# -------------------------
# 2D Conv Block
# -------------------------
class Conv2DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act=nn.GELU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            act()
        )

    def forward(self, x):
        return self.block(x)


# -------------------------
# 2D CNN Encoder → Transformer → 2D Decoder
# -------------------------
class EEGGenerator2DTransformer(nn.Module):
    def __init__(
        self,
        in_channels=8,      # 输入 8 个频段通道
        base_channels=(32, 64, 128),
        transformer_d_model=512,
        transformer_layers=4,
        nhead=8,
        ffn_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        # -------------------------
        # 2D Encoder
        # -------------------------
        self.enc1 = Conv2DBlock(in_channels, base_channels[0])  # 8 → 32
        self.enc2 = Conv2DBlock(base_channels[0], base_channels[1])  # 32 → 64
        self.enc3 = Conv2DBlock(base_channels[1], base_channels[2])  # 64 → 128

        # 最终 feature map: (B,128,3,3)
        self.pool = nn.MaxPool2d(3)  # 9x9 → 3x3

        self.token_dim = base_channels[2] * 3 * 3  # 128*3*3 = 1152

        # -------------------------
        # Projection to Transformer dims: 1152 → 512
        # -------------------------
        self.proj_in = nn.Linear(self.token_dim, transformer_d_model)
        self.pos_enc = SinusoidalPositionalEncoding(transformer_d_model)

        # -------------------------
        # Transformer
        # -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)

        # output projection: 512 → 1152
        self.proj_out = nn.Linear(transformer_d_model, self.token_dim)

        # -------------------------
        # Decoder
        # -------------------------
        # reshape → (B,128,3,3)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 3→6
        self.dec1 = Conv2DBlock(128, 64)

        self.up2 = nn.Upsample(size=(9, 9), mode="bilinear", align_corners=False)  # 6→9
        self.dec2 = Conv2DBlock(64, 32)

        self.final = nn.Conv2d(32, 128, kernel_size=3, padding=1)  # (B,128,9,9)

    def forward(self, x):
        mask = (x.abs().sum(dim=(1, 2), keepdim=True) > 0).float()
        # -------------------------
        # Encoder: (B, 8, 9, 9)
        # -------------------------
        x = self.enc1(x)   # → (B,32,9,9)
        x = self.enc2(x)   # → (B,64,9,9)
        x = self.enc3(x)   # → (B,128,9,9)

        x = self.pool(x)   # → (B,128,3,3)

        # -------------------------
        # Flatten → transformer tokens
        # -------------------------
        B = x.size(0)
        x = x.reshape(B, -1)           # → (B, 1152)
        x = self.proj_in(x)            # → (B, 512)
        x = x.unsqueeze(1)             # → (B, 1, 512)

        x = self.pos_enc(x)
        x = self.transformer(x)        # (B,1,512)

        x = self.proj_out(x)           # → (B,1,1152)
        x = x.squeeze(1)               # → (B,1152)

        # -------------------------
        # reshape → feature map
        # -------------------------
        x = x.view(B, 128, 3, 3)

        # -------------------------
        # Decoder: upsample → conv
        # -------------------------
        x = self.up1(x)  # (B,128,6,6)
        x = self.dec1(x)  # (B,64,6,6)

        x = self.up2(x)  # (B,64,9,9)
        x = self.dec2(x)  # (B,32,9,9)

        x = self.final(x)  # (B,128,9,9)
        x = torch.tanh(x)
        return x * mask

# ---------------------------------------------------------
#  Inception 多尺度 2D 卷积模块
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# SE 通道注意力
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 空间注意力模块 (CBAM 风格)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Transformer Encoder (3 层)
# 输入序列长度 = 81 (9×9)
# 每个 token 的维度 = d_model
# ---------------------------------------------------------
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


# ---------------------------------------------------------
#   判别 + 分类共享 Backbone
# ---------------------------------------------------------
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = EEGDataset4D(
    preprocessors_results=preprocessors_results,
    de_dir='./DE&PSD_feature/'
)

indices = np.arange(len(dataset))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=dataset.lbl_seg
)

train_set = Subset(dataset, train_idx)
test_set  = Subset(dataset, test_idx)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
generator = EEGGenerator2DTransformer().to(DEVICE)
classifier = SharedBackboneClassifier().to(DEVICE)

generator.load_state_dict(torch.load("stage2_generator.pth"))
ckpt = torch.load("preclassifier_raw_eeg_with_split.pth", map_location=DEVICE)
classifier.load_state_dict(ckpt["model"])

generator.eval()
for p in generator.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

epochs = 100

for epoch in range(epochs):
    classifier.train()
    total_loss = 0.0

    for de, raw, lbl in train_loader:
        de  = de.to(DEVICE)           # (B, 8, 9, 9)
        raw = raw.to(DEVICE)          # (B, 128, 9, 9)
        lbl = lbl.to(DEVICE)          # (B,)

        with torch.no_grad():
            fake = generator(de)      # (B, 128, 9, 9)

        # ========== 1 : 1 混合 ==========
        mix_eeg = torch.cat([raw, fake], dim=0)     # (2B, 128, 9, 9)
        mix_lbl = torch.cat([lbl, lbl], dim=0)      # (2B,)

        pred = classifier(mix_eeg)
        loss = criterion(pred, mix_lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"[Stage3][Epoch {epoch+1}/{epochs}] "
        f"Loss = {total_loss / len(train_loader):.4f}"
    )

classifier.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for _, raw, lbl in test_loader:
        raw = raw.to(DEVICE)
        lbl = lbl.to(DEVICE)

        logits = classifier(raw)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)

print("===== Stage3 Fine-tune Results =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score : {f1:.4f}")

torch.save(
    classifier.state_dict(),
    "classifier_finetuned_with_gan_eeg.pth"
)