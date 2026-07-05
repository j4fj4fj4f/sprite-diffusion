import torch
import numpy as np

dim = 12
t = torch.tensor(np.arange(5))

half_dim = dim // 2
embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
# print(f"embd1", embeddings, "shape", embeddings.shape)
print(f"1   ",embeddings.shape)
embeddings = torch.exp(torch.arange(half_dim,) * -embeddings)
# print(f"embd2", embeddings, "shape", embeddings.shape)
print(f"2   ",embeddings.shape)
embeddings = t[:, None] * embeddings[None, :]
# print(f"embd3", embeddings, "shape", embeddings.shape)
print(f"3   ",embeddings.shape)
embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
# print(f"embd4", embeddings, "shape", embeddings.shape)
print(f"4   ",embeddings.shape)
