"""DDPM noise prediction, cosine schedule, and ancestral reverse sampling."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class DDPMSchedule(nn.Module):
    def __init__(self, timesteps=200):
        super().__init__()
        if timesteps < 2:
            raise ValueError("At least two timesteps are required.")
        self.timesteps = timesteps
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
        cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-8, 0.999).float()
        alpha = 1 - beta
        abar = torch.cumprod(alpha, 0)
        previous = F.pad(abar[:-1], (1, 0), value=1.0)
        for name, value in {
            "betas": beta,
            "alpha_bar": abar,
            "posterior_variance": beta * (1 - previous) / (1 - abar),
            "mean_x0": beta * previous.sqrt() / (1 - abar),
            "mean_xt": (1 - previous) * alpha.sqrt() / (1 - abar),
        }.items():
            self.register_buffer(name, value)

    def add_noise(self, clean, t, noise):
        a = self.alpha_bar[t][:, None, None, None]
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def training_loss(self, model, clean):
        t = torch.randint(self.timesteps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        return F.mse_loss(model(self.add_noise(clean, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model, count, generator):
        device = self.betas.device
        image = torch.randn((count, 3, 32, 32), generator=generator, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((count,), step, device=device, dtype=torch.long)
            epsilon = model(image, t)
            a = self.alpha_bar[step]
            x0 = ((image - (1 - a).sqrt() * epsilon) / a.sqrt()).clamp(-1, 1)
            mean = self.mean_x0[step] * x0 + self.mean_xt[step] * image
            if step:
                noise = torch.randn(image.shape, generator=generator, device=device)
                image = mean + self.posterior_variance[step].sqrt() * noise
            else:
                image = mean
        return image.clamp(-1, 1)
