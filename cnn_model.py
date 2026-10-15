"""The exact CNN architecture specified in Assignment2.pdf."""

import torch
from torch import nn


ARCHITECTURE = "assignment2_cnn_64_v1"
IMAGE_SIZE = 64


class AssignmentCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("Expected a tensor of shape (batch, 3, 64, 64).")
        # Return logits: CrossEntropyLoss already applies log-softmax internally.
        return self.classifier(self.features(images))
