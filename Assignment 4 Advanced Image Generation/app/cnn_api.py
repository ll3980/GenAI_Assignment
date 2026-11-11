"""Attach this router to the existing FastAPI app using include_router()."""

import io
import warnings

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.cnn_model import ARCHITECTURE, IMAGE_SIZE
from app.cnn_preprocessing import CLASSES
from app.cnn_service import get_classifier


router = APIRouter(tags=["CNN image classifier"])
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


def ready_classifier():
    try:
        return get_classifier()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/cnn/health")
def cnn_health():
    classifier = ready_classifier()
    return {
        "status": "ready",
        "architecture": ARCHITECTURE,
        "image_size": IMAGE_SIZE,
        "checkpoint_epoch": classifier.epoch,
    }


@router.get("/cnn/classes")
def cnn_classes():
    return {"classes": list(CLASSES)}


@router.post("/predict", response_model=PredictionResponse)
def predict_image(file: UploadFile = File(...)):
    """Upload an image and return its most likely CIFAR-10 class."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image must be at most 10 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="The image has too many pixels.")
                source.load()
                image = source.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=413, detail="The image has too many pixels.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Upload a readable image such as PNG or JPEG.") from exc
    return ready_classifier().predict(image)
