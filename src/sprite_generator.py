import torch
from torchvision.utils import save_image

from src.model.unet import UNet
from src.model.diffusion import Diffusion
from src.utils.config import load_config

from pathlib import Path


def sample(model, diffusion, device, n=4, img_size=64):
    model.eval()

    with torch.no_grad():

        # start from random noise
        x = torch.randn(n, 4, img_size, img_size, device=device)

        # reverse diffusion
        for t in reversed(range(diffusion.timesteps)):

            t_batch = torch.full((n,), t, device=device, dtype=torch.long)

            pred_noise = model(x, t_batch)

            x = diffusion.p_sample(x, t_batch, pred_noise)

        return x
    
if __name__ == "__main__":
    cfg = load_config("configs/base.yaml")
    version = cfg['experiment']['name'] 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)

    # epoch = cfg["training"]["epochs"] - 1
    epoch = "best_t"
    checkpoint_path = Path("checkpoints") / version / f"{epoch}.pth"
    # checkpoint_path = Path("checkpoints") / version / f"epoch_{epoch}.pth"
    # checkpoint_path = Path("checkpoints") / version / "last.pth"
    # checkpoint_path = Path("checkpoints") / version / "best_t.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    EMA = True
    if EMA == True :
        model.load_state_dict(checkpoint["ema"])
        sprite_dir = Path("checkpoints") / version / f"sprites_{epoch}_EMA"
    else:
        model.load_state_dict(checkpoint["model"])
        sprite_dir = Path("checkpoints") / version / f"sprites_{epoch}"
    
    print(sprite_dir)
    sprite_dir.mkdir(parents=True, exist_ok=True)

    diffusion = Diffusion(timesteps=1000, device=device)

    def ddpm_sample(n):
        for i in range(n):
            samples = sample(model, diffusion, device, n=4)

            save_image(samples, Path(sprite_dir / f"i_{i}generated.png"), nrow=2)

            print(f"Saved generated image {i}.png into: {sprite_dir}")
    
    def ddim_sample(n):
        for i in range(n):
            samples = diffusion.ddim_sample(
            model=model,
            shape=(4, 4, 64, 64),
            device=device,
            steps=200,
            eta=1.0,
            )
            save_image(samples, Path(sprite_dir / "ddim" / f"i_{i}generated.png"), nrow=2)

            print(f"Saved generated image {i}.png into: {sprite_dir}")
    

    # ddim_sample(5)
    ddpm_sample(5) 