"""Optional lossless CIFAR-10 mirror cache; normal training uses torchvision."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from app.cnn_preprocessing import CLASSES


class PreparedCIFAR10(Dataset):
    classes = list(CLASSES)

    def __init__(self, directory: Path, train: bool, transform):
        filename = "train.npz" if train else "test.npz"
        expected_count = 50_000 if train else 10_000
        with np.load(Path(directory) / filename, allow_pickle=False) as cached:
            self.data = cached["images"]
            labels = cached["labels"]
        if self.data.shape != (expected_count, 32, 32, 3) or self.data.dtype != np.uint8:
            raise ValueError("Invalid CIFAR-10 image cache.")
        if labels.shape != (expected_count,) or labels.dtype.kind not in "iu":
            raise ValueError("Invalid CIFAR-10 labels.")
        if not np.all((labels >= 0) & (labels <= 9)):
            raise ValueError("CIFAR-10 labels must be in [0, 9].")
        if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, expected_count // 10)):
            raise ValueError("Unexpected CIFAR-10 class counts.")
        self.targets = labels.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        return self.transform(image), self.targets[index]
