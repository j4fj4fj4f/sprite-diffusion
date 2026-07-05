import yaml

def load_config(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    
    path = "../../configs/base.yaml"
    config = load_config(path)
    from pathlib import Path
    
    rootdir = Path(config["root_dir"])
    print(rootdir)

    import os
    # print(os.path.exists(rootdir))
    # print(os.listdir(rootdir))

    # rawdir = rootdir / Path(config["data"]["raw_dir"])
    # print(os.listdir(rawdir))
    print(config["versions"]["simpleUNet"])