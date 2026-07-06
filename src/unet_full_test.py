import torch
import torch.nn as nn
# import torch.nn.functional as F

from src.model.unet import Block, SinusoidalPositionEmbeddings,UNet


# -----------------------------
# 1. Sinusoidal embedding test
# -----------------------------
def test_sinusoidal():
    # dim = 64 # v1
    dim = 128
    t = torch.arange(10).float()  # [T]

    emb_layer = SinusoidalPositionEmbeddings(dim)
    out = emb_layer(t)

    print("\n[Sinusoidal Test]")
    print("shape:", out.shape)

    assert out.shape == (10, dim)

    print("min:", out.min().item(), "max:", out.max().item())


# -----------------------------
# 2. Time MLP test
# -----------------------------
def test_time_mlp():
    # base = 64 # v1
    base = 96
    t = torch.arange(10).float()

    time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(base),
        nn.Linear(base, base),
        nn.SiLU(),
        nn.Linear(base, base),
    )

    out = time_mlp(t)

    print("\n[Time MLP Test]")
    print("shape:", out.shape)

    assert out.shape == (10, base)


# -----------------------------
# 3. Block test 
# -----------------------------
def test_block():
    B = 2
    H = W = 32

    in_ch = 128
    out_ch = 128
    time_dim = 64

    x = torch.randn(B, in_ch, H, W)

    # IMPORTANT: this is NOT raw timestep anymore
    t_emb = torch.randn(B, time_dim)

    block = Block(in_ch, out_ch, time_dim)

    out = block(x, t_emb)

    print("\n[Block Test]")
    print("input shape:", x.shape)
    print("output shape:", out.shape)

    assert out.shape == (B, out_ch, H, W)


# -----------------------------
# 4. U-Net test
# -----------------------------
def test_unet():
    B = 32
    x = torch.randn(B, 4, 64, 64)
    t = torch.randint(0, 1000, (B,))
    # print(f"t        " ,t)
    model = UNet()

    out = model(x, t)

    print("\n[UNet Test]")
    print("input:", x.shape)
    print("output:", out.shape)

    assert out.shape == x.shape


# -----------------------------
# RUN ALL TESTS
# -----------------------------
if __name__ == "__main__":
    test_sinusoidal()
    test_time_mlp()
    test_block()
    test_unet()

    print("\nALL TESTS PASSED ✔")