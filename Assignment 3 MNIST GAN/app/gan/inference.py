"""Load the trained generator and encode 28x28 grayscale PNGs."""

import io
import math
from pathlib import Path

import torch
from PIL import Image

from app.gan.models import Generator, LATENT_DIM


class GeneratorService:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("latent_dim") != LATENT_DIM or checkpoint.get("steps", 0) < 1:
            raise ValueError("Expected a trained Assignment 3 generator checkpoint")
        self.model = Generator()
        self.model.load_state_dict(checkpoint["generator_state_dict"])
        self.model.to(self.device).eval()
        self.epoch = int(checkpoint["epoch"])

    def generate(self, count: int, seed: int | None = None) -> list[Image.Image]:
        # A request-local CPU RNG also works when inference is on another device.
        # It avoids changing the process-wide seed in concurrent API requests.
        rng = torch.Generator()
        rng.seed() if seed is None else rng.manual_seed(seed)
        z = torch.randn(count, LATENT_DIM, generator=rng).to(self.device)
        with torch.inference_mode():
            pixels = ((self.model(z).cpu() + 1) / 2).clamp(0, 1)
        arrays = (pixels[:, 0] * 255).round().to(torch.uint8).numpy()
        return [Image.fromarray(array) for array in arrays]


def encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def image_grid(images: list[Image.Image]) -> Image.Image:
    columns = min(8, len(images))
    rows = math.ceil(len(images) / columns)
    grid = Image.new("L", (columns * 28, rows * 28), color=0)
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * 28, (index // columns) * 28))
    return grid
