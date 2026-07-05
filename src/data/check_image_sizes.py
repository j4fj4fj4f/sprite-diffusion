from PIL import Image
from pathlib import Path
from collections import defaultdict
import os

TARGET_SIZE = (64, 64)

CDIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CDIR,"../../"))             #abspath converts relative path into absolute full PATHHHHHHH
SPRITE_DIR = os.path.join(ROOT_DIR,"data","raw","sprites")

def check_images():
    path = Path(SPRITE_DIR)

    if not path.exists():
        print(f"Folder not found: {SPRITE_DIR}")
        return

    size_counts = defaultdict(int)
    wrong_size_files = []

    images = list(path.rglob("*.png")) + list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg"))

    if len(images) == 0:
        print("No images found.")
        return

    print(f"Found {len(images)} images\n")

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                size = img.size  # (width, height)
                size_counts[size] += 1

                if size != TARGET_SIZE:
                    wrong_size_files.append((img_path.name, size))

        except Exception as e:
            print(f"Failed to load {img_path}: {e}")

    print("\n--- SIZE SUMMARY ---")
    for size, count in size_counts.items():
        print(f"{size}: {count}")

    print("\n--- NON-64x64 IMAGES ---")
    if not wrong_size_files:
        print("All images are 64x64")
    else:
        for name, size in wrong_size_files[:20]:
            print(f"{name}: {size}")

        if len(wrong_size_files) > 20:
            print(f"...and {len(wrong_size_files) - 20} more")

if __name__ == "__main__":
    check_images()