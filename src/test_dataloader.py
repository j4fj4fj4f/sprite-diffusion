import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.data.data_loader import create_dataloader

CLASS_NAMES = {
    0: "Human",
    1: "Night Elf",
    2: "Orc"
}

def show_batch(batch,labels = None):
    batch = batch[:16]  # take first 16 images

    batch = batch.permute(0, 2, 3, 1)  # CHW → HWC

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(batch[i])

        if labels is not None:
            ax.set_title(CLASS_NAMES[labels[i].item()])

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    cfg = load_config("configs/base.yaml")

    loader = create_dataloader(cfg)

    from collections import Counter             # count labels in entire Dataloader
    all_labels = []
    for images,labels in loader:
        all_labels.extend(labels.tolist())

    print(Counter(all_labels))

    images, labels = next(iter(loader))

    print("images shape:", images.shape)
    print("labels",labels)
    
    show_batch(images,labels)

    
if __name__ == "__main__":
    main()