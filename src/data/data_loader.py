from torch.utils.data import DataLoader
from src.data.sprite_dataset import SpriteDataset
from src.utils.paths import get_project_root
from src.utils.config import load_config

def create_dataloader(cfg):
    root = get_project_root()
    class_dirs = {}
    
    for folder,label in cfg["data"]["classes"].items():
        class_dirs[root/ cfg["data"]["root_dir"] / folder] = label

    dataset = SpriteDataset(class_dirs)

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

if __name__ == "__main__":
    root = get_project_root()
    print(root)
    cfg = load_config(root/"configs"/"base.yaml")
    dataloader = create_dataloader(cfg)
    print(type(dataloader))