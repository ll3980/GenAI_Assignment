"""Behavior checks for gradient boundaries and sampling, independent of image quality."""

import torch, pytest
from torch import nn
from app.image_generation.models import (
    CIFARUNet,
    SinusoidalTimeEmbedding,
    CIFAREnergyModel,
)
from app.image_generation.diffusion import DDPMSchedule
from app.image_generation.energy import frozen_parameters, langevin_sample

torch.set_num_threads(2)


def test_embedding():
    expected = torch.tensor(
        [
            0.84147098,
            0.54030231,
            0.09983342,
            0.99500417,
            0.00999983,
            0.99995000,
            0.00100000,
            0.99999950,
        ]
    )
    torch.testing.assert_close(
        SinusoidalTimeEmbedding(8)(torch.tensor([1]))[0], expected, atol=1e-7, rtol=1e-6
    )


def test_ddpm_weight_gradients():
    model = CIFARUNet(8)
    x = torch.randn(2, 3, 32, 32)
    loss = DDPMSchedule(20).training_loss(model, x)
    loss.backward()
    assert torch.isfinite(loss) and x.grad is None
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_forward_closed_form():
    schedule = DDPMSchedule(200)
    t = torch.tensor([0, 199])
    x = torch.ones(2, 3, 32, 32)
    noise = torch.full_like(x, 0.5)
    expected = schedule.alpha_bar[t].sqrt() + (1 - schedule.alpha_bar[t]).sqrt() * 0.5
    torch.testing.assert_close(schedule.add_noise(x, t, noise)[:, 0, 0, 0], expected)
    assert schedule.alpha_bar[-1] < 1e-6


def test_reproducible_reverse_sampling():
    model = CIFARUNet(8).eval().requires_grad_(False)
    schedule = DDPMSchedule(4)
    a = schedule.sample(model, 2, torch.Generator().manual_seed(12))
    b = schedule.sample(model, 2, torch.Generator().manual_seed(12))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert a.shape == (2, 3, 32, 32) and a.min() >= -1 and a.max() <= 1


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * x.square().flatten(1).sum(1)


def test_input_descent_keeps_weights_fixed():
    model = Quadratic()
    x = torch.full((2, 3, 32, 32), 0.5)
    with frozen_parameters(model), torch.no_grad():
        y = langevin_sample(model, x, 10, 0.1, 0.0)
    assert torch.all(model(y) < model(x))
    assert (
        model.scale.item() == 1
        and model.scale.grad is None
        and model.scale.requires_grad
    )
    assert not y.requires_grad


def test_freeze_restores_after_exception():
    model = Quadratic()
    with pytest.raises(RuntimeError):
        with frozen_parameters(model):
            raise RuntimeError("test")
    assert model.scale.requires_grad


def test_ebm_negative_chain_is_detached():
    model = CIFAREnergyModel(8)
    real = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    with frozen_parameters(model):
        fake = langevin_sample(model, torch.rand_like(real), 2)
    er, en = model(real), model(fake)
    loss = er.mean() - en.mean() + 0.1 * (er.square().mean() + en.square().mean())
    loss.backward()
    assert not fake.requires_grad and all(
        p.grad is not None for p in model.parameters()
    )


def test_missing_checkpoint(tmp_path):
    from app.image_generation.api import ImageGenerator

    with pytest.raises(FileNotFoundError):
        ImageGenerator("energy", tmp_path / "missing.pt")


def test_invalid_requests():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.image_generation.api import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    for body in [
        {"count": 0},
        {"count": 17},
        {"count": 1.2},
        {"seed": -1},
        {"prompt": "cat"},
    ]:
        assert client.post("/diffusion/generate", json=body).status_code == 422
    for body in [{"steps": 0}, {"steps": 501}, {"steps": True}]:
        assert client.post("/energy/generate", json=body).status_code == 422
