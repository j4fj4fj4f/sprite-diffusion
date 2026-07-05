import torch
from src.model.unet_second import SimpleUNet
import time

def main():
    model = SimpleUNet()
    B = 32
    x = torch.randn(B, 4, 64, 64)  # fake sprites b = 32
    t = torch.randint(0, 1000, (B,))
    print(f"t  " ,t)
    print(f"t.shape  " ,t.shape)

    before = time.time()
    out = model(x, t)
    after = time.time()

    print(f"forward pass took ", (after-before), "s")
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()