"""Verify assignment dimensions, gradients, checkpoint and real API responses."""

import io
import os
import unittest
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.cnn_api import MAX_UPLOAD_BYTES, router
from app.cnn_model import AssignmentCNN
from app.cnn_preprocessing import CLASSES, image_transform
from app.cnn_service import get_classifier
from app.main import app


class AssignmentTests(unittest.TestCase):
    def test_dimensions_and_gradients(self):
        model = AssignmentCNN()
        images = torch.randn(2, 3, 64, 64)
        shapes = []
        value = images
        for layer in list(model.features) + list(model.classifier):
            value = layer(value)
            shapes.append(tuple(value.shape[1:]))
        self.assertEqual(shapes, [
            (16, 64, 64), (16, 64, 64), (16, 32, 32),
            (32, 32, 32), (32, 32, 32), (32, 16, 16),
            (8192,), (100,), (100,), (10,),
        ])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 825398)
        torch.nn.CrossEntropyLoss()(value, torch.tensor([0, 9])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 32, 32))

    def test_training_mode_theory(self):
        model = AssignmentCNN()
        self.assertTrue(model.training)
        images = torch.randn(1, 3, 64, 64)
        train_output = model(images)
        model.eval()
        self.assertFalse(model.training)
        eval_output = model(images)
        torch.testing.assert_close(train_output, eval_output)
        self.assertTrue(eval_output.requires_grad)

    def test_trained_checkpoint_and_uploaded_image(self):
        checkpoint = get_classifier()
        image = Image.open(Path(__file__).resolve().parents[1] / "examples/cat.png")
        expected = checkpoint.predict(image)
        content = io.BytesIO()
        image.save(content, format="PNG")
        with TestClient(app) as client:
            self.assertEqual(client.get("/cnn/health").status_code, 200)
            response = client.post("/predict", files={"file": ("cat.png", content.getvalue(), "image/png")})
            self.assertEqual(response.status_code, 200)
            actual = response.json()
            self.assertEqual(actual["class_name"], expected["class_name"])
            self.assertEqual(actual["class_name"], CLASSES[actual["class_id"]])
            self.assertEqual(actual["class_name"], max(actual["probabilities"], key=actual["probabilities"].get))
            self.assertAlmostEqual(sum(actual["probabilities"].values()), 1.0, places=5)
            self.assertGreaterEqual(actual["confidence"], 0.0)
            self.assertLessEqual(actual["confidence"], 1.0)

    def test_grayscale_and_rgba(self):
        with TestClient(app) as client:
            for mode in ("L", "RGBA"):
                image = Image.new(mode, (37, 51))
                self.assertEqual(tuple(image_transform()(image.convert("RGB")).shape), (3, 64, 64))
                content = io.BytesIO()
                image.save(content, format="PNG")
                response = client.post("/predict", files={"file": ("image.png", content.getvalue(), "image/png")})
                self.assertEqual(response.status_code, 200)

    def test_bad_uploads(self):
        with TestClient(app) as client:
            self.assertEqual(client.post("/predict").status_code, 422)
            for content in (b"", b"not an image"):
                response = client.post("/predict", files={"file": ("bad.png", content, "image/png")})
                self.assertEqual(response.status_code, 400)
            response = client.post("/predict", files={"file": ("large.png", b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")})
            self.assertEqual(response.status_code, 413)

    def test_existing_app_integration(self):
        existing = FastAPI()

        @existing.post("/generate")
        def original_endpoint():
            return {"generated_text": "existing endpoint still works"}

        existing.include_router(router)
        with TestClient(existing) as client:
            self.assertEqual(client.post("/generate").status_code, 200)
            self.assertEqual(client.get("/cnn/health").status_code, 200)

    def test_missing_checkpoint_is_unavailable(self):
        previous_path = os.environ.get("MODEL_PATH")
        try:
            os.environ["MODEL_PATH"] = "models/does_not_exist.pt"
            get_classifier.cache_clear()
            with TestClient(app) as client:
                self.assertEqual(client.get("/cnn/health").status_code, 503)
        finally:
            if previous_path is None:
                os.environ.pop("MODEL_PATH", None)
            else:
                os.environ["MODEL_PATH"] = previous_path
            get_classifier.cache_clear()


if __name__ == "__main__":
    unittest.main()
