"""Train the specified CNN on CIFAR-10 and save the best validation checkpoint.

The official 50,000-image training set is split into 45,000 training and 5,000
validation images. The 10,000-image test set is evaluated once, after model
selection. Run from the project root: python train.py --epochs 10 --seed 617
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from app.cifar_dataset import PreparedCIFAR10
from app.cnn_model import ARCHITECTURE, IMAGE_SIZE, AssignmentCNN
from app.cnn_preprocessing import CLASSES, MEAN, STD, image_transform


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prepared-data", type=Path, default=None,
                        help="Optional NPZ cache produced by scripts/prepare_cifar10_mirror.py")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("results"))
    parser.add_argument("--examples-dir", type=Path, default=Path("examples"))
    parser.add_argument("--workers", type=int, default=0,
                        help="0 is portable; increase for faster loading on a normal workstation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive.")
    if args.workers < 0 or args.threads < 1:
        parser.error("workers must be nonnegative; threads must be positive.")
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, collect_confusion=False):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        count += labels.size(0)
        correct += int((predictions == labels).sum().item())
        loss_sum += float(loss.item()) * labels.size(0)
        if collect_confusion:
            pairs = (labels * 10 + predictions).cpu()
            confusion += torch.bincount(pairs, minlength=100).reshape(10, 10)
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}, confusion


def main():
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(selected_device)
    for directory in [args.model_dir, args.report_dir, args.examples_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Two dataset objects share raw examples but use different transforms.
    if args.prepared_data is None:
        augmented_data = CIFAR10(args.data_dir, train=True, download=True,
                                 transform=image_transform(training=True))
        validation_data = CIFAR10(args.data_dir, train=True, download=False,
                                  transform=image_transform(training=False))
        test_data = CIFAR10(args.data_dir, train=False, download=True,
                            transform=image_transform(training=False))
        data_source = {"loader": "torchvision.datasets.CIFAR10", "source": "Official CIFAR-10 Python archive"}
    else:
        augmented_data = PreparedCIFAR10(args.prepared_data, train=True,
                                         transform=image_transform(training=True))
        validation_data = PreparedCIFAR10(args.prepared_data, train=True,
                                          transform=image_transform(training=False))
        test_data = PreparedCIFAR10(args.prepared_data, train=False,
                                    transform=image_transform(training=False))
        data_source = json.loads((args.prepared_data / "source.json").read_text(encoding="utf-8"))
    if tuple(test_data.classes) != CLASSES:
        raise ValueError("Unexpected CIFAR-10 class order.")
    indices = torch.randperm(50_000, generator=torch.Generator().manual_seed(args.seed))
    training_set = Subset(augmented_data, indices[:45_000].tolist())
    validation_set = Subset(validation_data, indices[45_000:].tolist())
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    training_loader = DataLoader(
        training_set, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), **loader_options,
    )
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs * 7 // 10), gamma=0.5,
    )
    checkpoint_path = args.model_dir / "cifar10_cnn.pt"
    history = []
    best_accuracy = -1.0
    start = time.perf_counter()
    print(f"Device={device}; train=45000, validation=5000, test=10000", flush=True)

    for epoch in range(1, args.epochs + 1):
        # Validation switches to eval mode: restore training mode EVERY epoch.
        model.train()
        epoch_start = time.perf_counter()
        loss_sum, correct, count = 0.0, 0, 0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for step, (images, labels) in enumerate(training_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            count += labels.size(0)
            loss_sum += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            if step % 100 == 0:
                print(f"  epoch {epoch}: {count}/45000 images; loss={loss_sum/count:.4f}", flush=True)
        validation, _ = evaluate(model, validation_loader, criterion, device)
        row = {
            "epoch": epoch, "learning_rate": learning_rate,
            "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        with (args.report_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            checkpoint = {
                "architecture": ARCHITECTURE, "image_size": IMAGE_SIZE,
                "classes": list(CLASSES),
                "normalization": {"mean": list(MEAN), "std": list(STD)},
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch, "seed": args.seed,
                "validation_accuracy": best_accuracy,
            }
            torch.save(checkpoint, checkpoint_path)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}: train_acc={row['train_accuracy']:.4f}, "
              f"val_acc={validation['accuracy']:.4f}, seconds={row['seconds']:.1f}", flush=True)

    # Model selection used validation only. Evaluate this selected model once.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test, confusion = evaluate(model, test_loader, criterion, device, collect_confusion=True)
    checkpoint["test_accuracy"] = test["accuracy"]
    # Keep the delivered artifact portable even if training used a GPU.
    checkpoint["state_dict"] = {key: value.cpu() for key, value in checkpoint["state_dict"].items()}
    torch.save(checkpoint, checkpoint_path)
    report = {
        "architecture": ARCHITECTURE, "image_size": IMAGE_SIZE,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "classes": list(CLASSES), "seed": args.seed,
        "train_count": 45_000, "validation_count": 5_000, "test_count": 10_000,
        "epochs": args.epochs, "selected_epoch": checkpoint["epoch"],
        "batch_size": args.batch_size, "initial_learning_rate": args.learning_rate,
        "optimizer": "Adam", "loss": "CrossEntropyLoss",
        "augmentation": "RandomHorizontalFlip(p=0.5) on training images only",
        "normalization": {"mean": list(MEAN), "std": list(STD)},
        "validation_accuracy": best_accuracy,
        "test_accuracy": test["accuracy"], "test_loss": test["loss"],
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "device": str(device), "threads": args.threads, "workers": args.workers,
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "elapsed_training_and_evaluation_seconds": time.perf_counter() - start,
        "selection_rule": "Highest validation accuracy; test evaluated only after selection",
        "data_source": data_source,
    }
    (args.report_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    exported = set()
    for index, label in enumerate(test_data.targets):
        if label not in exported:
            from PIL import Image
            Image.fromarray(test_data.data[index]).save(args.examples_dir / f"{CLASSES[label]}.png")
            exported.add(label)
        if len(exported) == 10:
            break
    print(f"Selected epoch={checkpoint['epoch']}; test_accuracy={test['accuracy']:.4f}", flush=True)
    print(f"Saved {checkpoint_path} and {args.report_dir / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
