from utils.config import load_config
from utils.paths import get_project_root
from data.sprite_dataset import SpriteDataset
from pathlib import Path

cfg = load_config("../configs/base.yaml")
root = get_project_root()

# path_to_data = Path(cfg["root_dir"]) / Path(cfg["data"]["raw_dir"])
path_to_data = root / Path(cfg["data"]["raw_dir"])
dataset = SpriteDataset(path_to_data)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("Shape:", sample.shape)

import matplotlib.pyplot as plt

img = sample.permute(1, 2, 0)  # CHW → HWC

plt.imshow(img)
plt.show()