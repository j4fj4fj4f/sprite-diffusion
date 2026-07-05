from pathlib import Path

# This file is: Sprite-Diffusion/src/utils/paths.py
# So project root is 2 levels up

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_project_root():
    return PROJECT_ROOT