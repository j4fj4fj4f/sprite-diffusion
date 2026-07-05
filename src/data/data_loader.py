from torch.utils.data import DataLoader
from src.data.sprite_dataset import SpriteDataset
from src.utils.paths import get_project_root


def create_dataloader(cfg):
    root = get_project_root()

    dataset_path = root / cfg["data"]["raw_dir"]

    dataset = SpriteDataset(dataset_path)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["training"]["shuffle"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return dataloader