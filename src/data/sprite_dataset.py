from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class SpriteDataset(Dataset):
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)

        self.image_paths = sorted(
            list(self.image_dir.glob("*.png"))
        )

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGBA")

        image = self.transform(image)

        return image
    
