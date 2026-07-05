import kagglehub
import os

DATA_DIR = "data/raw"

def download():
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) >0:
        print("Dataset already exists, not downloading anything")
        return
    print("downloading Dataset...")
    path = kagglehub.dataset_download("volodymyrpivoshenko/pixel-characters-dataset")

    print("Downloaded to:", path)
    #move/copy?
    print("done")

if __name__ == "__main__":
    download()

