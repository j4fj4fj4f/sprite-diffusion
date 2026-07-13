from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class SpriteDataset(Dataset):
    def __init__(self, image_dirs: dict):
        """
        image_dirs:

        {
            "sprites_human": 0,
            "sprites_nightelves": 1,
            "sprites_orcs": 2
        }
        """
        self.image_paths = []
        self.labels = []

        for folder,label in image_dirs.items():
            folder = Path(folder)

            for img_path in sorted(folder.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGBA")

        image = self.transform(image)
        label = self.labels[idx]

        return image,label
    
