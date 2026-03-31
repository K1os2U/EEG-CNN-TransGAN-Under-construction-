import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl
from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation
import logging
from scipy.io import loadmat
from pathlib import Path
import os

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
        in_channels=8,
        base_channels=(32, 64, 128),
        transformer_d_model=512,
        transformer_layers=4,
        nhead=8,
        ffn_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.enc1 = Conv2DBlock(in_channels, base_channels[0])  # 8 → 32
        self.enc2 = Conv2DBlock(base_channels[0], base_channels[1])  # 32 → 64
        self.enc3 = Conv2DBlock(base_channels[1], base_channels[2])  # 64 → 128
        self.pool = nn.MaxPool2d(3)  # 9x9 → 3x3

        self.token_dim = base_channels[2] * 3 * 3  # 128*3*3 = 1152

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

    def forward(self, x):
        mask = (x.abs().sum(dim=(1, 2), keepdim=True) > 0).float()

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.pool(x)

        B = x.size(0)
        x = x.reshape(B, -1)
        x = self.proj_in(x)
        x = x.unsqueeze(1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.proj_out(x)
        x = x.squeeze(1)
        x = self.up1(x)
        x = self.dec1(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.final(x)
        x = torch.tanh(x)
        return x * mask


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
class SharedBackboneGANClassifier(nn.Module):
    def __init__(self, in_channels=128, cond_channels=8, d_model=256, num_classes=2):
        super().__init__()
        self.cond_embed = nn.Sequential(
            nn.Conv2d(cond_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1)
        )

        # ---- 多尺度卷积 ----
        self.inception = InceptionBlock(
            in_channels=in_channels + 32,
            out_channels=192
        )

        # ---- 注意力 ----
        self.se = SEBlock(192)
        self.spatial = SpatialAttention()

        # ---- 维度压缩到 Transformer 的输入维度 ----
        self.conv_reduce = nn.Conv2d(192, d_model, kernel_size=1)

        # ---- Transformer ----
        self.transformer = TransformerEncoder(d_model=d_model)

        # ---- 判别器头 ----
        self.discriminator_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 真/假
        )

        # ---- 分类器头 ----
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)   # 二分类输出
        )

    def forward(self, eeg, cond):
        """
        eeg : (B, 128, 9, 9)
        cond: (B, 8,   9, 9)
        """

        # 1️⃣ 条件嵌入
        cond_feat = self.cond_embed(cond)   # (B, 32, 9, 9)

        # 2️⃣ 条件融合（通道级）
        x = torch.cat([eeg, cond_feat], dim=1)  # (B, 160, 9, 9

        x = self.inception(x)

        # 2. 通道注意力
        x = self.se(x)

        # 3. 空间注意力
        x = self.spatial(x)

        # 4. Conv1×1 压缩通道 → Transformer d_model
        x = self.conv_reduce(x)   # (B, d_model, 9, 9)

        # 5. Flatten → 序列化
        x = x.flatten(2).transpose(1, 2)   # (B, 81, d_model)

        # 6. Transformer
        x = self.transformer(x)

        # 7. 取 CLS = 全局平均
        feat = x.mean(dim=1)  # (B, d_model)

        # --- 输出 ---
        disc_out = self.discriminator_head(feat)  # (B, 1)

        return disc_out


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_gradient_penalty(D, real_samples, fake_samples, cond_feat, device='cuda'):
    """
    WGAN-GP gradient penalty for conditional GAN
    real_samples: (B, C, H, W) 真实 EEG
    fake_samples: (B, C, H, W) 生成 EEG
    cond_feat: (B, 8, 9, 9) DE&PSD 特征
    """
    batch_size = real_samples.size(0)

    # alpha shape: (B, 1, 1, 1)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # 插值
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # 判别器输出
    d_interpolates, _ = D(interpolates, cond_feat)

    # fake tensor: ones_like
    fake = torch.ones_like(d_interpolates, device=device)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # reshape -> (B, C*H*W)
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
def generator_train(
    generator,
    discriminator,
    dataset,
    epochs=140,
    lambda_gp=10,
    batch_size = 64,
    device='cuda'
):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # optimizers
    g_opt = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=1e-4,
        betas=(0.5, 0.999)
    )

    for epoch in range(epochs):
        for i, (de_feat, real_eeg, _) in enumerate(loader):

            de_feat  = de_feat.to(device)      # (B, 8, 9, 9)
            real_eeg = real_eeg.to(device)     # (B, 128, 9, 9)

            discriminator.train()

            fake_eeg = generator(de_feat)

            d_real, _ = discriminator(real_eeg, de_feat)
            d_fake, _ = discriminator(fake_eeg.detach(), de_feat)


            gp = compute_gradient_penalty(
                discriminator,
                real_eeg,
                fake_eeg.detach(),
                de_feat
            )
            print(
                f"D_real={d_real.mean().item():.2f} | "
                f"D_fake={d_fake.mean().item():.2f} | "
                f"GP={gp.item():.2f}"
            )

            d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # =========================
            # Train Generator
            # =========================
            if i % 7 == 0:
                fake_eeg = generator(de_feat)
                d_fake2, _ = discriminator(fake_eeg, de_feat)

                g_loss = -d_fake2.mean()

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

        print(
            f"[Stage2][{epoch+1}/{epochs}] "
            f"D_loss={d_loss.item():.4f}, "
            f"G_loss={g_loss.item():.4f}"
        )

    torch.save(generator.state_dict(), "stage2_generator.pth")
    print("Stage2 GAN training finished.")

if __name__ == '__main__':
    with open('./dataset/deap_binary_valence_dataset.pkl', 'rb') as f:
        preprocessors_results = pkl.load(f)
    data = EEGDataset4D(preprocessors_results)
    generator_train(EEGGenerator2DTransformer(),SharedBackboneGANClassifier(),data)
