"""Alternating discriminator and generator optimization."""

import torch
from torch import nn

from app.gan.models import LATENT_DIM


def train_batch(real, generator, discriminator, optimizer_g, optimizer_d):
    criterion = nn.BCELoss()
    batch_size = real.shape[0]
    ones = torch.ones(batch_size, 1, device=real.device)
    zeros = torch.zeros(batch_size, 1, device=real.device)
    z = torch.randn(batch_size, LATENT_DIM, device=real.device)
    fake = generator(z)

    # D should classify real as 1 and generated images as 0.
    # detach() prevents this update from changing the generator.
    discriminator.requires_grad_(True)
    optimizer_d.zero_grad(set_to_none=True)
    loss_d = criterion(discriminator(real), ones)
    loss_d = loss_d + criterion(discriminator(fake.detach()), zeros)
    loss_d.backward()
    optimizer_d.step()

    # G should make D predict 1 on generated images (non-saturating loss).
    # D remains differentiable with respect to its input, but its parameters
    # are frozen during this update. BatchNorm stays in training mode.
    discriminator.requires_grad_(False)
    optimizer_g.zero_grad(set_to_none=True)
    loss_g = criterion(discriminator(fake), ones)
    loss_g.backward()
    optimizer_g.step()
    discriminator.requires_grad_(True)
    return float(loss_d.detach()), float(loss_g.detach())
