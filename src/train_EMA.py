import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import load_config
from src.data.data_loader import create_dataloader
from src.model.unet import UNet
from src.model.diffusion import Diffusion
from src.model.ema import EMA

from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import shutil

def train():
    cfg = load_config("configs/base.yaml")
    version = cfg['experiment']['name']
    checkpoint_dir = Path("checkpoints") / version
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on device:  " , device)

    # data
    loader = create_dataloader(cfg)
    print("Data Loader created successfully with length: ", len(loader))
    # model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])

    #EMA
    ema = EMA(model,decay = 0.98)

    #### try loading an earlier version
    if cfg["checkpoint"]["resume"]:
        # version = cfg['experiment']['name']
        checkpoint_path = checkpoint_dir / "best.pth"   #checkpoint_path is the loading path
        if checkpoint_path.exists():
            # create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = checkpoint_dir / f"best_backup_{timestamp}.pth"
            shutil.copy2(checkpoint_path,backup_path)
            print(f"Backed up checkpoint to: {backup_path}")

            checkpoint = torch.load(checkpoint_path,map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print(f"resumed checkpoint from: {checkpoint_path}")
        else:
            print(f"checkpoint not found: {checkpoint_path}, starting from scratch")
    else:
        print("Training from scratch")

    # diffusion
    diffusion = Diffusion(timesteps=1000, device=device)

    mse = nn.MSELoss()

    model.train()
    best = 10
    best_test = 10
    for epoch in range(cfg["training"]["epochs"]):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        
        for i, (x0, labels) in enumerate(pbar):
            x0 = x0.to(device)
            labels = labels.to(device)

            # # remove alpha -> using RGB instead of RGBalpa
            # if x0.shape[1] == 4:
            #     x0 = x0[:, :3, :, :]

            # sample timestep
            t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)

            # sample noise
            noise = torch.randn_like(x0)

            # create noisy image
            xt = diffusion.q_sample(x0, t, noise)

            # predict noise
            pred_noise = model(xt, t, labels)

            loss = mse(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update EMA
            ema.update(model)
            loss_value = loss.item()

            if i % 2 ==0:
                pbar.set_postfix(loss=f"{loss_value:.4f}")

            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        pbar.set_postfix(avg_loss=f"{epoch_loss:.4f}")
        if epoch_loss < best and epoch > (cfg["training"]["save_min_epoch"]):           #saves the model based on the best cumulated loss in an epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema":ema.ema_model.state_dict()
                    },checkpoint_dir / f"best.pth")
            print(f"Saved checkpoint for epoch {epoch} as best")
            best = epoch_loss

        if epoch_loss < 0.003 and epoch > (cfg["training"]["save_min_epoch"]): #conditioned on loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema":ema.ema_model.state_dict()
                    },checkpoint_dir / f"{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch} loss = {loss.item():.4f}")
        
        if (epoch % cfg["training"]["save_interval"]) == 0 and epoch > (cfg["training"]["save_min_epoch"]):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema":ema.ema_model.state_dict()
                    },checkpoint_dir / f"{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch} loss = {loss.item():.4f}")

    torch.save(                                                                         # save last run
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema":ema.ema_model.state_dict()
                },checkpoint_dir / f"last.pth")
    print(f"Saved checkpoint for epoch {epoch}")

        # if ((epoch + 1) % cfg["training"]["save_interval"]) == 0: # uncomment this block if you want to use a fixed save_interval
        #     torch.save(
        #         model.state_dict(),
        #         checkpoint_dir / f"epoch_{epoch}.pth"
        #         )
        #     print(f"Saved checkpoint for epoch {epoch}")

if __name__ == "__main__":
    train()