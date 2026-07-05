import torch


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

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

        return mean + torch.sqrt(beta) * noise


    # old sampling mismatched t and noise
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