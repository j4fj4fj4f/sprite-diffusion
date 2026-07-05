import torch
from src.model.unet_first import SimpleUNet


def main():
    model = SimpleUNet()

    x = torch.randn(2, 4, 64, 64)  # fake sprites
    t = torch.tensor([10, 20])     # timesteps

    out = model(x, t)

    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()