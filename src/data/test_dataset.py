from src.data.sprite_dataset import SpriteDataset
import os
from src.utils.config import load_config
from pathlib import Path



CDIR = Path(__file__).parent
ROOT_DIR = CDIR.parent.parent             #abspath converts relative path into absolute full PATHHHHHHH
cfg = load_config(ROOT_DIR/"configs"/"base.yaml")

class_dirs = {}
    
for folder,label in cfg["data"]["classes"].items():
    class_dirs[ROOT_DIR/ cfg["data"]["root_dir"] / folder] = label

dataset = SpriteDataset(class_dirs)
print(len(dataset))

sample = dataset[0][0]
label = dataset[0][1]
print(sample.shape)
print(label)