import torch
import matplotlib.pyplot as plt

from src.model.diffusion import Diffusion
from src.utils.config import load_config
from src.data.data_loader import create_dataloader


def show(img):
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def main():
    cfg = load_config("configs/base.yaml")

    loader = create_dataloader(cfg)

    batch = next(iter(loader))

    x0 = batch[0]  # one sprite

    diffusion = Diffusion(timesteps=1000)

    steps = [0, 50, 100, 300, 600, 999]

    plt.figure(figsize=(10, 2))

    for i, t in enumerate(steps):
        xt = diffusion.q_sample(
            x0.unsqueeze(0),
            torch.tensor([t]),
        )[0]

        img = xt.permute(1, 2, 0).cpu()

        plt.subplot(1, len(steps), i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"t={t}")

    plt.show()


if __name__ == "__main__":
    main()