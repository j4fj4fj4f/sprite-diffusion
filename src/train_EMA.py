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
from pathlib import Path

def train():
    cfg = load_config("configs/base.yaml")
    version = cfg['experiment']['name']
    checkpoint_dir = Path("checkpoints") / version
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on device:  " , device)

    # data
    loader = create_dataloader(cfg)

    # model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])

    ##### try loading an earlier version
    version = cfg['experiment']['name']
    checkpoint_path = checkpoint_dir / "best.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    #EMA
    ema = EMA(model,decay = 0.98)

    # diffusion
    diffusion = Diffusion(timesteps=1000, device=device)

    mse = nn.MSELoss()

    model.train()
    best = 10
    for epoch in range(cfg["training"]["epochs"]):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for i, x0 in enumerate(pbar):
            x0 = x0.to(device)

            # # remove alpha if needed
            # if x0.shape[1] == 4:
            #     x0 = x0[:, :3, :, :]

            # sample timestep
            t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)

            # sample noise
            noise = torch.randn_like(x0)

            # create noisy image
            xt = diffusion.q_sample(x0, t, noise)

            # predict noise
            pred_noise = model(xt, t)

            loss = mse(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update EMA
            ema.update(model)
            loss_value = loss.item()

            if i % 10 ==0:
                pbar.set_postfix(loss=loss_value)
            if i % 25 ==0:
                pbar.set_postfix(loss=loss_value)
            if i % 50 == 0:
                print(f"{version} Epoch {epoch} Step {i} Loss {loss.item():.4f}")
        if loss.item() < best and epoch > 85:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema":ema.ema_model.state_dict()
                    },checkpoint_dir / f"best.pth")
            print(f"Saved checkpoint for epoch {epoch} as best")
            best = loss.item()
        if loss.item() < 0.005: #conditioned on loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema":ema.ema_model.state_dict()
                    },checkpoint_dir / f"epoch_{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch} loss = {loss.item():.4f}")
    torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema":ema.ema_model.state_dict()
                },checkpoint_dir / f"last.pth")
    print(f"Saved checkpoint for epoch {epoch}")

        # if ((epoch + 1) % cfg["training"]["save_interval"]) == 0:
        #     torch.save(
        #         model.state_dict(),
        #         checkpoint_dir / f"epoch_{epoch}.pth"
        #         )
        #     print(f"Saved checkpoint for epoch {epoch}")

if __name__ == "__main__":
    train()