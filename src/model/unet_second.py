import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.act = nn.SiLU()

        # maps time embedding → feature channels
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x, t):
        """
        x: [B, C, H, W]
        t: [B, time_dim]
        """

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # inject time here
        t_emb = self.time_proj(t)[:, :, None, None]  # [B, C, 1, 1]
        h = h + t_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, base=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base),
            nn.Linear(base, base),
            nn.SiLU(),
            nn.Linear(base, base),
        )

        self.conv1 = Block(in_channels, base, base)
        self.conv2 = Block(base, base * 2, base)
        self.conv3 = Block(base * 2 + base * 2, base, base)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = Block(base * 2, base * 2, base)
        self.up = nn.Upsample(scale_factor=2)

        # self.conv3 = Block(base * 2 + base * 2, base)
        self.out = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, t):
        t = t.long()
        t_emb = self.time_mlp(t)

        # encoder
        x1 = self.conv1(x, t_emb)
        x2 = self.conv2(self.pool(x1), t_emb)

        # bottleneck
        x_mid = self.bottleneck(self.pool(x2), t_emb)

        # decoder
        x = self.up(x_mid)
        x = F.interpolate(x, size=x2.shape[2:])

        x = torch.cat([x, x2], dim=1)

        x = self.conv3(x, t_emb)
        x = F.interpolate(x, size=x1.shape[2:]) #quick fix for shapes

        return self.out(x)