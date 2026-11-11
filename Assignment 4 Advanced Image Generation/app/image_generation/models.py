"""Time-conditioned U-Net and scalar energy CNN for native 32x32 CIFAR-10."""

import math
import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128, max_period=10000):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("Time dimension must be positive and even.")
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(max_period) * torch.arange(dim // 2) / (dim // 2)),
        )

    def forward(self, t):
        angles = t.float()[:, None] * self.frequencies[None]
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


class TimeResidualBlock(nn.Module):
    def __init__(self, cin, cout, time_dim):
        super().__init__()
        self.first = nn.Sequential(
            nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1)
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, cout))
        self.second = nn.Sequential(
            nn.GroupNorm(8, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1)
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x, time):
        return self.skip(x) + self.second(
            self.first(x) + self.time(time)[:, :, None, None]
        )


class CIFARUNet(nn.Module):
    def __init__(self, base_channels=32, time_dim=128):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of eight.")
        b = base_channels
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(3, b, 3, padding=1)
        self.enc0 = TimeResidualBlock(b, b, time_dim)
        self.down0 = nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.enc1 = TimeResidualBlock(2 * b, 2 * b, time_dim)
        self.down1 = nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1)
        self.enc2 = TimeResidualBlock(4 * b, 4 * b, time_dim)
        self.down2 = nn.Conv2d(4 * b, 4 * b, 4, stride=2, padding=1)
        self.mid1 = TimeResidualBlock(4 * b, 4 * b, time_dim)
        self.mid2 = TimeResidualBlock(4 * b, 4 * b, time_dim)
        self.up2 = nn.ConvTranspose2d(4 * b, 4 * b, 4, stride=2, padding=1)
        self.dec2 = TimeResidualBlock(8 * b, 4 * b, time_dim)
        self.up1 = nn.ConvTranspose2d(4 * b, 2 * b, 4, stride=2, padding=1)
        self.dec1 = TimeResidualBlock(4 * b, 2 * b, time_dim)
        self.up0 = nn.ConvTranspose2d(2 * b, b, 4, stride=2, padding=1)
        self.dec0 = TimeResidualBlock(2 * b, b, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, b), nn.SiLU(), nn.Conv2d(b, 3, 3, padding=1)
        )

    def forward(self, x, t):
        time = self.time(t)
        h0 = self.enc0(self.input(x), time)
        h1 = self.enc1(self.down0(h0), time)
        h2 = self.enc2(self.down1(h1), time)
        h = self.mid2(self.mid1(self.down2(h2), time), time)
        h = self.dec2(torch.cat((self.up2(h), h2), 1), time)
        h = self.dec1(torch.cat((self.up1(h), h1), 1), time)
        h = self.dec0(torch.cat((self.up0(h), h0), 1), time)
        return self.output(h)


class CIFAREnergyModel(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(b, 2 * b, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(2 * b, 4 * b, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(4 * b, 4 * b, 4, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(4 * b * 4 * 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
