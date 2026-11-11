"""JSON and PNG routes for trained, unconditional CIFAR-10 generators."""

import base64, io, math, os
from functools import lru_cache
from pathlib import Path
import torch
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image
from app.image_generation.models import CIFARUNet, CIFAREnergyModel
from app.image_generation.diffusion import DDPMSchedule
from app.image_generation.energy import langevin_sample

router = APIRouter(tags=["Assignment 4 image generation"])
ROOT = Path(__file__).resolve().parents[2]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(default=1, ge=1, le=16, strict=True)
    seed: int | None = Field(default=42, ge=0, le=4294967295, strict=True)


class EnergyRequest(GenerateRequest):
    steps: int = Field(default=60, ge=1, le=500, strict=True)


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def grid(images):
    columns = min(4, len(images))
    canvas = Image.new("RGB", (columns * 32, math.ceil(len(images) / columns) * 32))
    for i, image in enumerate(images):
        canvas.paste(image, ((i % columns) * 32, (i // columns) * 32))
    return canvas


class ImageGenerator:
    def __init__(self, kind, path):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Missing {Path(path).name}; train and include this checkpoint."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("model_type") != ("ddpm" if kind == "diffusion" else "ebm")
            or checkpoint.get("dataset") != "CIFAR10"
            or checkpoint.get("steps", 0) < 1
            or checkpoint.get("image_shape") != [3, 32, 32]
        ):
            raise ValueError(
                "Expected a trained CIFAR-10 checkpoint of the correct model type."
            )
        self.kind = kind
        self.config = checkpoint["config"]
        self.device = torch.device(os.getenv("IMAGE_DEVICE", "cpu"))
        self.metadata = {
            "model": checkpoint["model_type"],
            "dataset": "CIFAR-10",
            "width": 32,
            "height": 32,
            "channels": 3,
            "epoch": checkpoint["epoch"],
            "training_steps": checkpoint["steps"],
            "images_seen": checkpoint["images_seen"],
            "conditional": False,
        }
        if kind == "diffusion":
            self.model = CIFARUNet(
                self.config["base_channels"], self.config["time_dim"]
            )
            self.model.load_state_dict(
                checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
            )
            self.schedule = DDPMSchedule(self.config["timesteps"]).to(self.device)
        else:
            self.model = CIFAREnergyModel(self.config["base_channels"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval().requires_grad_(False)

    def generate(self, count, seed, steps=60):
        generator = torch.Generator(device=self.device)
        generator.seed() if seed is None else generator.manual_seed(seed)
        if self.kind == "diffusion":
            samples = self.schedule.sample(self.model, count, generator)
        else:
            initial = (
                torch.rand((count, 3, 32, 32), device=self.device, generator=generator)
                * 2
                - 1
            )
            # Pixel autograd stays enabled; no inference_mode around EBM sampling.
            samples = langevin_sample(
                self.model,
                initial,
                steps,
                self.config["step_size"],
                self.config["noise_std"],
                generator,
            )
        pixels = (
            ((samples.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        )
        return [Image.fromarray(array) for array in pixels.permute(0, 2, 3, 1).numpy()]


@lru_cache(maxsize=2)
def get_service(kind):
    path = Path(
        os.getenv(f"{kind.upper()}_MODEL_PATH", str(ROOT / f"models/cifar10_{kind}.pt"))
    )
    try:
        return ImageGenerator(kind, path)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def result(kind, body, steps=None):
    service = get_service(kind)
    images = service.generate(body.count, body.seed, steps or 60)
    return {
        **service.metadata,
        "count": body.count,
        "seed": body.seed,
        "format": "png",
        "encoding": "base64",
        "sampling_steps": service.config["timesteps"] if kind == "diffusion" else steps,
        "images": [
            base64.b64encode(png_bytes(image)).decode("ascii") for image in images
        ],
    }


@router.get("/diffusion/health")
def diffusion_health():
    return {"status": "ready", **get_service("diffusion").metadata}


@router.get("/energy/health")
def energy_health():
    return {"status": "ready", **get_service("energy").metadata}


@router.post("/diffusion/generate")
def diffusion_generate(body: GenerateRequest):
    return result("diffusion", body)


@router.post("/energy/generate")
def energy_generate(body: EnergyRequest):
    return result("energy", body, body.steps)


@router.get("/diffusion/generate.png", response_class=Response)
def diffusion_png(
    count: int = Query(4, ge=1, le=16), seed: int = Query(42, ge=0, le=4294967295)
):
    return Response(
        png_bytes(grid(get_service("diffusion").generate(count, seed))),
        media_type="image/png",
    )


@router.get("/energy/generate.png", response_class=Response)
def energy_png(
    count: int = Query(4, ge=1, le=16),
    seed: int = Query(42, ge=0, le=4294967295),
    steps: int = Query(60, ge=1, le=500),
):
    return Response(
        png_bytes(grid(get_service("energy").generate(count, seed, steps))),
        media_type="image/png",
    )
