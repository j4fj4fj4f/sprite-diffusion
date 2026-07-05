import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.data.data_loader import create_dataloader


def show_batch(batch):
    batch = batch[:16]  # take first 16 images

    batch = batch.permute(0, 2, 3, 1)  # CHW → HWC

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(batch[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    cfg = load_config("configs/base.yaml")

    loader = create_dataloader(cfg)

    batch = next(iter(loader))

    print("Batch shape:", batch.shape)

    show_batch(batch)


if __name__ == "__main__":
    main()