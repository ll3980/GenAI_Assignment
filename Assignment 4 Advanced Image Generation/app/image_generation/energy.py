"""Sampling changes pixels; the later training backward changes weights."""

from contextlib import contextmanager
import torch


@contextmanager
def frozen_parameters(model):
    flags = [p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)


def langevin_sample(
    model, initial, steps=60, step_size=10.0, noise_std=0.005, generator=None
):
    """Practical short-run x <- clamp(x-eta*grad(E)+sigma*z).

    Gradient clipping and the small noise setting make this approximate synthesis,
    not exact equilibrium sampling. Ideal ULA for exp(-E) uses sigma=sqrt(2*eta).
    """
    if steps < 1 or step_size <= 0 or noise_std < 0:
        raise ValueError("Invalid sampler settings.")
    image = initial.detach().clone()
    for _ in range(steps):
        with torch.enable_grad():
            image = image.detach().requires_grad_(True)
            (gradient,) = torch.autograd.grad(
                model(image).sum(), image, create_graph=False
            )
        with torch.no_grad():
            noise = torch.randn(image.shape, device=image.device, generator=generator)
            image = (
                image - step_size * gradient.clamp(-0.03, 0.03) + noise_std * noise
            ).clamp(-1, 1)
    return image.detach()


class ReplayBuffer:
    def __init__(self, size=1024):
        self.images = torch.empty(size, 3, 32, 32).uniform_(-1, 1)

    def sample(self, count, device):
        old = self.images[torch.randint(len(self.images), (count,))].to(device)
        fresh = torch.empty_like(old).uniform_(-1, 1)
        restart = torch.rand(count, 1, 1, 1, device=device) < 0.05
        return torch.where(restart, fresh, old)

    def update(self, images):
        indices = torch.randperm(len(self.images))[: len(images)]
        self.images[indices] = images[: len(indices)].detach().cpu()
