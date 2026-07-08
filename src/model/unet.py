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

        # residuals Identiy if out=in else 1c1 convolution to match shapes
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        identity = self.res_conv(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # time embedding injection
        t_emb = self.time_proj(t)[:, :, None, None]
        h = h + t_emb

        h = self.conv2(h)
        h = self.norm2(h)

        # residuals
        h = h + identity
        h = self.act(h)

        return h
    
class SelfAttention(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels

        self.norm = nn.GroupNorm(8,channels)

        self.q = nn.Conv2d(channels,channels,1)
        self.k = nn.Conv2d(channels,channels,1)
        self.v = nn.Conv2d(channels,channels,1)

        self.proj = nn.Conv2d(channels,channels,1)

    def forward(self, x):
        B,C,H,W = x.shape

        h = self.norm(x)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = q.view(B,C,H*W).transpose(1,2) # B,C,H,W -> B,C,HW -> B,HW,C
        k = k.view(B,C,H*W)                #                   -> B,C,HW
        v = v.view(B,C,H*W).transpose(1,2) #                   -> B,HW,C

        # attn = torch.bmm(q,k) #B,HW,C * B,C,HW -> B,HW,HW
        attn = torch.bmm(q,k) / (C**0.5)    #makes softmax "softer" amd follows original formula
        # attn = F.softmax(attn,dim=1)
        attn = F.softmax(attn,dim=-1)

        out = torch.bmm(attn,v) ##B,HW,HW * B,HW,C -> B,HW,C
        out = out.transpose(1,2).view(B,C,H,W) #B,HW,C->B,C,HW->B,C,H,W

        out = self.proj(out) # 1d conv

        return out + x #residual
class UNet(nn.Module):
    ##############
    # v1 values base = 64, time_dim = 64
    ##############
    def __init__(self, in_channels=4, base=96, time_dim=128):
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
        self.bottleneck1 = Block(base * 2, base * 3, time_dim)
        self.bottleneck2 = Block(base * 3, base * 2, time_dim)

        #attention
        self.attn = SelfAttention(base*2)

        # decoder
        self.up = nn.Upsample(scale_factor=2)

        self.conv3 = Block(base * 2 + base * 2, base, time_dim)
        self.conv4 = Block(base * 2, base, time_dim)

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
        x3 = self.pool(x2)
        x3 = self.bottleneck1(x3, t)  # 16x16
        x3 = self.bottleneck2(x3, t)  # 16x16

        #attention
        x3 = self.attn(x3)

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
    