import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from pathlib import Path

from src.model.unet import UNet
from src.model.diffusion import Diffusion
from src.utils.config import load_config


@torch.no_grad()
def sample_with_steps(model, diffusion, device, n=4, img_size=64, save_steps=None):
    model.eval()

    # start from pure noise
    x = torch.randn(n, 4, img_size, img_size, device=device)

    results = []

    save_steps = set(save_steps or [])

    for t in reversed(range(diffusion.timesteps)):

        t_batch = torch.full((n,), t, device=device, dtype=torch.long)

        pred_noise = model(x, t_batch)

        x = diffusion.p_sample(x, t_batch, pred_noise)

        # store snapshots at selected timesteps
        if t in save_steps:
            results.append((t, x.detach().cpu()))

    return results


def save_plot(results, out_path, n=4):
    """
    results: list of (timestep, tensor)
    each tensor shape: [n, 4, H, W]
    """

    results = sorted(results, key=lambda x: -x[0])

    fig, axes = plt.subplots(len(results), n, figsize=(2 * n, 2 * len(results)))

    # if only one timestep
    if len(results) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (t, batch) in zip(range(len(results)), results):

        for col in range(n):

            ax = axes[row_idx, col]

            img = batch[col][:3]  # RGBA → RGB

            img = img.permute(1, 2, 0).numpy()

            # normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            ax.imshow(img)
            ax.axis("off")

            if col == 0:
                ax.set_title(f"t={t}")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    cfg = load_config("configs/base.yaml")

    version = cfg["experiment"]["name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # output folder
    out_dir = Path("checkpoints") / version / "timesteps_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = UNet().to(device)
    epoch = 116
    checkpoint_path = Path("checkpoints") / version / f"epoch_{epoch}.pth"
    checkpoint_path = Path("checkpoints") / version / "last.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    diffusion = Diffusion(timesteps=1000, device=device)

    # choose timesteps to inspect
    save_steps = [
        999, 900, 800, 700,
        500, 300, 200, 100, 50, 10, 0
    ]

    results = sample_with_steps(
        model,
        diffusion,
        device,
        n=4,
        img_size=64,
        save_steps=save_steps
    )

    out_path = out_dir / "trajectory.png"
    save_plot(results, out_path)

    print(f"Saved timestep visualization to {out_path}")