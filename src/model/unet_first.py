import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)
    
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

        self.conv1 = Block(in_channels, base)
        self.conv2 = Block(base, base * 2)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = Block(base * 2, base * 2)

        self.up = nn.Upsample(scale_factor=2)

        self.conv3 = Block(base * 2 + base * 2, base)
        self.out = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, t):
        # time embedding
        t_emb = self.time_mlp(t).view(t.shape[0], -1, 1, 1)

        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))

        # bottleneck
        x_mid = self.bottleneck(self.pool(x2))

        # decoder
        x = self.up(x_mid)

        # match shape exactly to skip connection
        x = F.interpolate(x, size=x2.shape[2:])

        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        # final output must match original input size
        x = F.interpolate(x, size=x1.shape[2:])

        return self.out(x)