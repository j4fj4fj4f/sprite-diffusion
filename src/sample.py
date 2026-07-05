import torch
from src.model.unet import UNet


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"sampling on device" , device)
model = UNet().to(device)

model.load_state_dict(
    torch.load("checkpoints/unet_epoch_9.pth", map_location=device)
)

model.eval()