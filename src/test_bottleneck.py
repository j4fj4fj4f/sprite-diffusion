import torch
from src.model.unet_second import SimpleUNet,Block
from src.model.unet_second import SinusoidalPositionEmbeddings

def main():
    base = 64
    B = 1

    x = torch.randn(B, base * 2, 32, 32)  # matches bottleneck input

    t_emb = torch.randn(B, base)  # fake time embedding

    bottleneck = Block(base * 2, base * 2, base)

    out = bottleneck(x, t_emb)

    print(out.shape)

    dim = 64
    t = torch.arange(10).float()  # [T]

    emb_layer = SinusoidalPositionEmbeddings(dim)

    out = emb_layer(t)

    print(out.shape)  # [10, 64]

if __name__ == "__main__":
    main()