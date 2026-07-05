import copy
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def to(self, device):
        self.ema_model.to(device)
        return self