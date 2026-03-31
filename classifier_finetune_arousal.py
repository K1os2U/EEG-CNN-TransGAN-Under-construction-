import os
import math
import random
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from scipy.io import loadmat


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
batch_size = 64
epochs = 100
classifier_lr = 1e-4
test_size = 0.2

def set_seed(seed: int = 42):
    """Ensure reproducibility."""
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


class EEGDataset4D(Dataset):
    def __init__(self, preprocessors_results: dict, de_dir: str = './DE&PSD_Arousal_feature/'):
        de_all, raw_all, lbl_all = [], [], []

        for subject in sorted(preprocessors_results.keys()):
            raw = preprocessors_results[subject]['feature']  # (2400, 128, 9, 9)
            lbl = preprocessors_results[subject]['label']  # (2400,)
            de = loadmat(os.path.join(de_dir, f'{subject}.mat'))['data']  # (2400, 8, 9, 9)

            assert raw.shape[0] == de.shape[0] == lbl.shape[0]

            raw = raw.reshape(40, 60, 128, 9, 9)
            de = de.reshape(40, 60, 8, 9, 9)
            lbl = lbl.reshape(40, 60)

            for t in range(40):
                raw[t] = robust_norm(raw[t])
                de[t][:, :4] = robust_norm(de[t][:, :4])  # DE
                de[t][:, 4:] = robust_norm(de[t][:, 4:])  # PSD

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

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class Conv2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act=nn.GELU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            act()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EEGGenerator2DTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int = 8,
            base_channels=(32, 64, 128),
            transformer_d_model: int = 512,
            transformer_layers: int = 4,
            nhead: int = 8,
            ffn_dim: int = 1024,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.enc1 = Conv2DBlock(in_channels, base_channels[0])
        self.enc2 = Conv2DBlock(base_channels[0], base_channels[1])
        self.enc3 = Conv2DBlock(base_channels[1], base_channels[2])
        self.pool = nn.MaxPool2d(3)  # 9x9 → 3x3

        self.token_dim = base_channels[2] * 3 * 3  # 1152
        self.proj_in = nn.Linear(self.token_dim, transformer_d_model)
        self.pos_enc = SinusoidalPositionalEncoding(transformer_d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        self.proj_out = nn.Linear(transformer_d_model, self.token_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = Conv2DBlock(128, 64)
        self.up2 = nn.Upsample(size=(9, 9), mode="bilinear", align_corners=False)
        self.dec2 = Conv2DBlock(64, 32)
        self.final = nn.Conv2d(32, 128, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x.abs().sum(dim=(1, 2), keepdim=True) > 0).float()

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.pool(x)  # (B,128,3,3)

        B = x.size(0)
        x = x.view(B, -1)  # (B,1152)
        x = self.proj_in(x).unsqueeze(1)  # (B,1,512)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.proj_out(x.squeeze(1)).view(B, 128, 3, 3)

        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = torch.tanh(self.final(x))
        return x * mask


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
        feat = x.mean(dim=1)
        return self.classifier_head(feat)

def classifier_finetune(dataset, generator, classifier, TEST_SIZE, SEED, BATCH_SIZE, NUM_EPOCHS, DEVICE, CLASSIFIER_LR):
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=dataset.lbl_seg
    )
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize models
    generator = generator.to(DEVICE)
    classifier = classifier.to(DEVICE)

    # Load checkpoints
    GENERATOR_CKPT_PATH = "Generator_BinaryClass_Arousal.pth"
    PRECLASSIFIER_CKPT_PATH = "Classifier_Pretrain_BinaryClass_Arousal.pth"
    SAVE_PATH = "Classifier_Finetune_BinaryClass_Arousal.pth"
    generator.load_state_dict(torch.load(GENERATOR_CKPT_PATH, map_location=DEVICE))
    classifier.load_state_dict(torch.load(PRECLASSIFIER_CKPT_PATH, map_location=DEVICE))

    # Freeze generator
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=CLASSIFIER_LR)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        classifier.train()
        total_loss = 0.0

        for de, raw, lbl in train_loader:
            de, raw, lbl = de.to(DEVICE), raw.to(DEVICE), lbl.to(DEVICE)

            with torch.no_grad():
                fake = generator(de)  # (B, 128, 9, 9)

            # Mix real and fake
            mix_eeg = torch.cat([raw, fake], dim=0)
            mix_lbl = torch.cat([lbl, lbl], dim=0)

            pred = classifier(mix_eeg)
            loss = criterion(pred, mix_lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Stage3][Epoch {epoch + 1}/{NUM_EPOCHS}] Loss = {total_loss / len(train_loader):.4f}")

    # Final evaluation
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, raw, lbl in test_loader:
            raw, lbl = raw.to(DEVICE), lbl.to(DEVICE)
            preds = classifier(raw).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n===== Stage3 Fine-tune Results =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Save final model
    torch.save(classifier.state_dict(), SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

if __name__ == '__main__':
    # Disable Flash Attention for compatibility
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    set_seed(seed)

    # Load preprocessed data
    with open('./dataset/deap_binary_Arousal_dataset.pkl', 'rb') as f:
        preprocessors_results = pkl.load(f)

    # Build dataset
    dataset = EEGDataset4D(preprocessors_results)
    generator = EEGGenerator2DTransformer()
    classifier = SharedBackboneClassifier()

    classifier_finetune(dataset, generator, classifier, test_size, seed, batch_size, epochs, device, classifier_lr)
