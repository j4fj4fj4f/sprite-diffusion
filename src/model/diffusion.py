import torch

import math

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1

    x = torch.linspace(0, timesteps, steps)

    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
    ) ** 2

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (
        alphas_cumprod[1:] / alphas_cumprod[:-1]
    )

    return torch.clamp(betas, 0.0001, 0.9999)

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # #cosine beta schedulee
        # self.betas = cosine_beta_schedule(timesteps).to(device)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion: x0 -> xt
        """

        if noise is None:
            noise = torch.randn_like(x0)

        # gather alpha_bar at timestep t
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise   #return xt
    
    @torch.no_grad()
    def p_sample(self, x, t, pred_noise):
        # ensure correct shape indexing
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)

        # correct per-sample noise handling
        noise = torch.randn_like(x)
        noise = torch.where((t > 0).view(-1, 1, 1, 1), noise, torch.zeros_like(noise))

        mean = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
        )

        return mean + torch.sqrt(beta) * noise  #return xt-1 from xt

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        device,
        steps=100,
        eta=0.0,
    ):
        """
        DDIM sampling.

        eta = 0.0 -> deterministic DDIM
        eta = 1.0 -> behaves similarly to DDPM
        """

        x = torch.randn(shape, device=device)

        # choose timesteps
        times = torch.linspace(
            self.timesteps - 1,
            0,
            steps,
            dtype=torch.long,
            device=device,
        )

        for i in range(len(times) - 1):

            t = times[i].repeat(shape[0])
            t_next = times[i + 1].repeat(shape[0])

            alpha_bar = self.alpha_bar[t].view(-1,1,1,1)
            alpha_bar_next = self.alpha_bar[t_next].view(-1,1,1,1)

            pred_noise = model(x, t)

            # predicted x0
            x0 = (
                x - torch.sqrt(1 - alpha_bar) * pred_noise
            ) / torch.sqrt(alpha_bar)

            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_next)
            )

            noise = torch.randn_like(x)

            direction = torch.sqrt(
                torch.clamp(1 - alpha_bar_next - sigma**2, min=0)
            ) * pred_noise

            x = (
                torch.sqrt(alpha_bar_next) * x0
                + direction
                + sigma * noise
            )

        return x
    

    # old sampling mismatched t and noise (but it worked)
    # @torch.no_grad()
    # def p_sample(self, x, t, pred_noise):
    #     alpha = self.alphas[t][:, None, None, None]
    #     alpha_bar = self.alpha_bar[t][:, None, None, None]
    #     beta = self.betas[t][:, None, None, None]

    #     noise = torch.randn_like(x) if (t[0] > 0) else 0

    #     mean = (1 / torch.sqrt(alpha)) * (
    #         x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
    #     )

    #     return mean + torch.sqrt(beta) * noise