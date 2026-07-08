import gc
import torch

# del x0, xt, noise, pred_noise, losss
gc.collect()
torch.cuda.empty_cache()