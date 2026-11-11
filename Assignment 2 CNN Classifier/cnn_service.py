"""Load a trained checkpoint once and classify an RGB image on CPU."""

import os
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from app.cnn_model import ARCHITECTURE, IMAGE_SIZE, AssignmentCNN
from app.cnn_preprocessing import CLASSES, MEAN, STD, image_transform


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models/cifar10_cnn.pt"


class CNNClassifier:
    def __init__(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Trained checkpoint not found. Run train.py and include "
                "models/cifar10_cnn.pt in the deployment."
            )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True,
        )
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("Checkpoint architecture does not match this assignment.")
        if checkpoint.get("image_size") != IMAGE_SIZE:
            raise ValueError("Checkpoint image size must be 64.")
        if tuple(checkpoint.get("classes", [])) != CLASSES:
            raise ValueError("Checkpoint class order does not match CIFAR-10.")
        if checkpoint.get("normalization") != {"mean": list(MEAN), "std": list(STD)}:
            raise ValueError("Checkpoint normalization differs from inference preprocessing.")
        self.model = AssignmentCNN()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.transform = image_transform(training=False)
        self.classes = CLASSES
        self.checkpoint_path = checkpoint_path
        self.epoch = int(checkpoint["epoch"])

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(probabilities.argmax().item())
        return {
            "class_id": index,
            "class_name": self.classes[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                name: float(probabilities[i].item())
                for i, name in enumerate(self.classes)
            },
        }


@lru_cache(maxsize=1)
def get_classifier() -> CNNClassifier:
    return CNNClassifier(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
