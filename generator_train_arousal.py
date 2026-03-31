import os
import math
import pickle as pkl
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1e-4
LAMBDA_GP = 10  # Gradient penalty weight for WGAN-GP



def robust_norm(x: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
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
            # Load raw EEG and labels from preprocessed dict
            raw = preprocessors_results[subject]['feature']  # (2400, 128, 9, 9)
            lbl = preprocessors_results[subject]['label']  # (2400,)

            # Load DE+PSD features from .mat file
            de_path = os.path.join(de_dir, f'{subject}.mat')
            de = loadmat(de_path)['data']  # (2400, 8, 9, 9)

            assert raw.shape[0] == de.shape[0] == lbl.shape[0], \
                f"Shape mismatch for {subject}"

            raw = raw.reshape(40, 60, 128, 9, 9)
            de = de.reshape(40, 60, 8, 9, 9)
            lbl = lbl.reshape(40, 60)

            for t in range(40):
                # Normalize raw EEG
                raw[t] = robust_norm(raw[t])

                de_trial = de[t]
                de_trial[:, :4] = robust_norm(de_trial[:, :4])  # DE bands
                de_trial[:, 4:] = robust_norm(de_trial[:, 4:])  # PSD bands
                de[t] = de_trial

            de_all.append(de.reshape(-1, 8, 9, 9))
            raw_all.append(raw.reshape(-1, 128, 9, 9))
            lbl_all.append(lbl.reshape(-1))

        self.de_seg = np.concatenate(de_all, axis=0)
        self.raw_seg = np.concatenate(raw_all, axis=0)
        self.lbl_seg = np.concatenate(lbl_all, axis=0)

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
        pe = pe.unsqueeze(0)  # (1, L, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input sequence."""
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

class EEGGenerator2DTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int = 8,
            base_channels: tuple = (32, 64, 128),
            transformer_d_model: int = 512,
            transformer_layers: int = 4,
            nhead: int = 8,
            ffn_dim: int = 1024,
            dropout: float = 0.1,
    ):
        super().__init__()
        # Encoder
        self.enc1 = Conv2DBlock(in_channels, base_channels[0])
        self.enc2 = Conv2DBlock(base_channels[0], base_channels[1])
        self.enc3 = Conv2DBlock(base_channels[1], base_channels[2])
        self.pool = nn.MaxPool2d(3)  # 9x9 → 3x3

        token_dim = base_channels[2] * 3 * 3  # 1152
        self.proj_in = nn.Linear(token_dim, transformer_d_model)
        self.pos_enc = SinusoidalPositionalEncoding(transformer_d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        self.proj_out = nn.Linear(transformer_d_model, token_dim)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # 3→6
        self.dec1 = Conv2DBlock(128, 64)
        self.up2 = nn.Upsample(size=(9, 9), mode="bilinear", align_corners=False)  # 6→9
        self.dec2 = Conv2DBlock(64, 32)
        self.final = nn.Conv2d(32, 128, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x.abs().sum(dim=(1, 2), keepdim=True) > 0).float()

        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.pool(x)  # (B, 128, 3, 3)

        # Transformer
        B = x.size(0)
        x = x.view(B, -1)  # (B, 1152)
        x = self.proj_in(x)  # (B, 512)
        x = x.unsqueeze(1)  # (B, 1, 512)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, 1, 512)
        x = self.proj_out(x).squeeze(1)  # (B, 1152)

        # Decoder
        x = x.view(B, 128, 3, 3)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.final(x)
        return torch.tanh(x) * mask


class SharedBackboneGANClassifier(nn.Module):

    def __init__(self, in_channels: int = 128, cond_channels: int = 8, d_model: int = 256):
        super().__init__()
        self.cond_embed = nn.Sequential(
            nn.Conv2d(cond_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1)
        )
        self.inception = InceptionBlock(in_channels + 32, 192)
        self.se = SEBlock(192)
        self.spatial = SpatialAttention()
        self.conv_reduce = nn.Conv2d(192, d_model, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.discriminator_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, eeg: torch.Tensor, cond: torch.Tensor):
        cond_feat = self.cond_embed(cond)
        x = torch.cat([eeg, cond_feat], dim=1)
        x = self.inception(x)
        x = self.se(x)
        x = self.spatial(x)
        x = self.conv_reduce(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        feat = x.mean(dim=1)
        disc_out = self.discriminator_head(feat)
        return disc_out


# Loss & Training Utilities
def compute_gradient_penalty(
        D: nn.Module,
        real_samples,
        fake_samples,
        cond_feat,
        device: str = 'cuda'
) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP (conditional version)."""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates, cond_feat)  # Only discriminator output needed
    fake = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generator_train(
        generator,
        discriminator,
        dataset,
        epochs: int = NUM_EPOCHS,
        lambda_gp: float = LAMBDA_GP,
        batch_size: int = BATCH_SIZE,
        device: str = DEVICE
):
    generator.to(device)
    discriminator.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    g_opt = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (de_feat, real_eeg, _) in enumerate(loader):
            de_feat = de_feat.to(device)
            real_eeg = real_eeg.to(device)

            # --- Train Discriminator ---
            fake_eeg = generator(de_feat).detach()
            d_real = discriminator(real_eeg, de_feat)
            d_fake = discriminator(fake_eeg, de_feat)

            gp = compute_gradient_penalty(discriminator, real_eeg, fake_eeg, de_feat, device)
            d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # --- Train Generator every few steps ---
            if i % 7 == 0:
                fake_eeg = generator(de_feat)
                d_fake2 = discriminator(fake_eeg, de_feat)
                g_loss = -d_fake2.mean()

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

        print(f"[Epoch {epoch + 1}/{epochs}] D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

    torch.save(generator.state_dict(), "Generator_BinaryClass_Arousal.pth")
    print("Training finished. Generator saved.")

if __name__ == '__main__':
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Load preprocessed data
    with open('./dataset/deap_binary_Arousal_dataset.pkl', 'rb') as f:
        preprocessors_results = pkl.load(f)

    dataset = EEGDataset4D(preprocessors_results)
    generator = EEGGenerator2DTransformer()
    discriminator = SharedBackboneGANClassifier()

    generator_train(generator, discriminator, dataset)
