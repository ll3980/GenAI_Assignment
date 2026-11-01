"""Shape, gradient, and HTTP behavior checks without downloading MNIST."""

import base64
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.gan.api import get_service
from app.gan.models import Generator, Discriminator
from app.gan.training import train_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def test_shapes_and_training():
    generator, discriminator = Generator(), Discriminator()
    fake = generator(torch.randn(3, 100))
    assert fake.shape == (3, 1, 28, 28)
    assert fake.min() >= -1 and fake.max() <= 1
    predictions = discriminator(fake)
    assert predictions.shape == (3, 1)
    assert torch.all((predictions >= 0) & (predictions <= 1))
    before_g = generator.fc.weight.detach().clone()
    before_d = discriminator.features[0].weight.detach().clone()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    losses = train_batch(torch.rand(3, 1, 28, 28) * 2 - 1,
                         generator, discriminator, optimizer_g, optimizer_d)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses)
    assert not torch.equal(before_g, generator.fc.weight)
    assert not torch.equal(before_d, discriminator.features[0].weight)


def test_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("GAN_MODEL_PATH", str(tmp_path / "missing.pt"))
    get_service.cache_clear()
    try:
        with TestClient(app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/gan/health").status_code == 503
            assert client.post("/gan/generate", json={"count": 1}).status_code == 503
    finally:
        get_service.cache_clear()


def test_api_png_seed_and_validation(monkeypatch, tmp_path):
    # This temporary randomly initialized fixture only tests serving mechanics.
    # It is not the submitted trained generator or evidence of image quality.
    path = tmp_path / "fixture.pt"
    torch.save({"generator_state_dict": Generator().state_dict(),
                "latent_dim": 100, "epoch": 1, "steps": 1}, path)
    monkeypatch.setenv("GAN_MODEL_PATH", str(path))
    get_service.cache_clear()
    try:
        with TestClient(app) as client:
            assert client.get("/gan/health").status_code == 200
            first = client.post("/gan/generate", json={"count": 2, "seed": 42})
            second = client.post("/gan/generate", json={"count": 2, "seed": 42})
            assert first.status_code == 200
            assert first.json() == second.json()
            for encoded in first.json()["images"]:
                picture = Image.open(io.BytesIO(base64.b64decode(encoded)))
                assert picture.size == (28, 28) and picture.mode == "L"
            png = client.get("/gan/generate.png?count=16&seed=42")
            assert png.status_code == 200 and png.headers["content-type"] == "image/png"
            assert Image.open(io.BytesIO(png.content)).size == (224, 56)
            for invalid in [{"count": 0}, {"count": 65}, {"seed": -1}]:
                assert client.post("/gan/generate", json=invalid).status_code == 422
    finally:
        get_service.cache_clear()
