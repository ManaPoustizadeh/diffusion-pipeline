import torch


class DiffusionScheduler:
    def __init__(self, timesteps: int):
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0)
        alpha_t = self.alpha_cumprod[t]
        x_t = x_0 * torch.sqrt(alpha_t) + noise * torch.sqrt(1 - alpha_t)
        return x_t, noise

    def reverse_diffusion_step(self, x_t, t, pred_noise):
        alpha_t = self.alpha_cumprod[t]
        x = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) * pred_noise)
        if t > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(1 - self.alpha_cumprod[t - 1]) * noise
        return x
