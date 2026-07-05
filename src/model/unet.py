import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        return emb
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.act = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        t_emb = self.time_proj(t)[:, :, None, None]
        h = h + t_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h
    
class UNet(nn.Module):
    def __init__(self, in_channels=4, base=64, time_dim=64):
        super().__init__()

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # encoder
        self.conv1 = Block(in_channels, base, time_dim)
        self.conv2 = Block(base, base * 2, time_dim)

        # bottleneck
        self.bottleneck = Block(base * 2, base * 2, time_dim)

        # decoder
        self.up = nn.Upsample(scale_factor=2)

        self.conv3 = Block(base * 2 + base * 2, base, time_dim)
        self.conv4 = Block(base + base, base, time_dim)

        # output
        self.out = nn.Conv2d(base, in_channels, 1)

        # pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        t = t.long()
        t = self.time_mlp(t)

        # encoder
        x1 = self.conv1(x, t)          # 64x64
        x2 = self.conv2(self.pool(x1), t)  # 32x32

        # bottleneck
        x3 = self.bottleneck(self.pool(x2), t)  # 16x16

        # decoder stage 1
        x = self.up(x3)  # 32x32
        # x = F.interpolate(x, size=x2.shape[2:]) # BANDAID FIX - uncomment if shape mismatch
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x, t)

        # decoder stage 2
        x = self.up(x)  # 64x64
        # x = F.interpolate(x, size=x1.shape[2:]) # BANDAID FIX - uncomment if shape mismatch
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x, t) 

        return self.out(x)