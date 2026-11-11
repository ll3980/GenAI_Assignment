"""Reusable router: existing_app.include_router(router)."""

import base64
import os
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.gan.inference import GeneratorService, encode_png, image_grid

router = APIRouter(prefix="/gan", tags=["GAN"])
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models/mnist_generator.pt"


@lru_cache(maxsize=1)
def get_service() -> GeneratorService:
    path = Path(os.environ.get("GAN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Generator checkpoint missing. Run python -m training.train_gan first.")
    return GeneratorService(path, device=os.environ.get("GAN_DEVICE", "cpu"))


class GenerateRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=64)
    seed: int | None = Field(default=None, ge=0, le=4294967295)


@router.get("/health")
def gan_health():
    service = get_service()
    return {"status": "ready", "model": "unconditional MNIST GAN", "epoch": service.epoch}


@router.post("/generate")
def generate(body: GenerateRequest):
    images = get_service().generate(body.count, body.seed)
    return {
        "count": body.count, "seed": body.seed, "width": 28, "height": 28,
        "format": "png", "encoding": "base64",
        "images": [base64.b64encode(encode_png(image)).decode("ascii") for image in images],
    }


@router.get("/generate.png", response_class=Response)
def generate_png(count: int = Query(default=16, ge=1, le=64),
                 seed: int | None = Query(default=None, ge=0, le=4294967295)):
    images = get_service().generate(count, seed)
    return Response(content=encode_png(image_grid(images)), media_type="image/png")
