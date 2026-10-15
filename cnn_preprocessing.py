"""Shared resizing and normalization for training and deployed inference."""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.cnn_model import IMAGE_SIZE


CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def image_transform(training: bool = False) -> transforms.Compose:
    operations = []
    if training:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transforms.Compose(operations)
