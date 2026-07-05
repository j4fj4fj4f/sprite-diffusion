from sprite_dataset import SpriteDataset
import os


CDIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CDIR,"../../"))             #abspath converts relative path into absolute full PATHHHHHHH
SPRITE_DIR = os.path.join(ROOT_DIR,"data","raw","sprites")

dataset = SpriteDataset(SPRITE_DIR)
print(len(dataset))

sample = dataset[0]
print(sample.shape)