import torch
from src.model.unet_first import SimpleUNet,Block
from src.model.unet_first import SinusoidalPositionEmbeddings

def main():
    block = Block
    base = 64
    bottleneck = Block(base * 2, base * 2)
    
    logtest = torch.log(torch.tensor(10000.0)) / (32 - 1)
    print(logtest)


    #random data
    x = torch.randn(1, 128, 64, 64)  # fake activation maps
    t = torch.tensor([10, 20, 30, 40])     # timesteps
    t = torch.arange(999)
    print(t)
    dim = 64
    print(f"dim  ",dim)
    Emb = SinusoidalPositionEmbeddings(dim)
    embedding = Emb(t)
    # print(f"embedding   ",embedding)
    print(embedding.shape)
    out = bottleneck(x)
    print(out.shape)

    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()