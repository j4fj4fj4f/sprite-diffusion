import torch
import torch.nn as nn
import torch.nn.functional as F


B = 2
H = W = 16
C = 10

x = torch.randn(B,H,W,C)
print(f"initial ",x.shape)
x = x.permute(0,3,1,2)
print(f"permuted ",x.shape) # -> B,C,H,W
proj = nn.Conv2d(10,10,1)
x = proj(x)
print("projected", x.shape)