from pathlib import Path
from src.utils.config import load_config

cfg = load_config("configs/base.yaml")
version = cfg["experiment"]["name"]
epoch = "best"

def get_next_index(folder):
    existing = list(folder.glob("i_*generated.png"))

    if not existing:
        return 0

    indices = []
    for file in existing:
        # file name example: i_12generated.png
        name = file.stem  # "i_12generated"
        idx = name.replace("i_", "").replace("generated", "")
        try:
            indices.append(int(idx))
        except ValueError:
            pass

    return max(indices) + 1 if indices else 0


sprite_dir = Path("checkpoints") / version / f"sprites_{epoch}_EMA"

index = get_next_index(sprite_dir)
print(index)