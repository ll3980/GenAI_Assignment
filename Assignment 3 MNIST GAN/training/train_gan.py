"""Train on MNIST and save weights, fixed-noise samples, and loss logs."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan.models import Discriminator, Generator, LATENT_DIM, initialize_weights
from app.gan.training import train_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/assignment3"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("models/mnist_generator.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        parser.error("epochs, batch-size, lr, and threads must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device_used=str(device), torch_version=str(torch.__version__))
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Tanh outputs [-1, 1]; normalize real MNIST images to the same range.
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # A local RNG prevents preview sampling from changing training noise.
    fixed_z = torch.randn(64, LATENT_DIM, generator=torch.Generator().manual_seed(args.seed)).to(device)
    total_steps = 0
    print(f"Training on {len(dataset)} MNIST images with device={device}", flush=True)

    with (args.output_dir / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss_d", "loss_g", "seconds", "steps"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            generator.train()
            discriminator.train()
            sum_d = sum_g = 0.0
            seen = 0
            for real, _labels in loader:
                real = real.to(device)
                loss_d, loss_g = train_batch(real, generator, discriminator, optimizer_g, optimizer_d)
                sum_d += loss_d * real.shape[0]
                sum_g += loss_g * real.shape[0]
                seen += real.shape[0]
                total_steps += 1
            generator.eval()
            with torch.inference_mode():
                preview = generator(fixed_z).cpu()
            save_image((preview + 1) / 2, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
            # CPU tensors make the inference checkpoint portable.
            torch.save({
                "generator_state_dict": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
                "latent_dim": LATENT_DIM, "epoch": epoch, "steps": total_steps,
                "dataset": "MNIST train", "num_training_images": len(dataset),
            }, args.checkpoint_path)
            # Full checkpoint is also available for inspecting/reusing training state.
            torch.save({
                "epoch": epoch, "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
            }, args.output_dir / "training_checkpoint.pt")
            row = dict(epoch=epoch, loss_d=sum_d / seen, loss_g=sum_g / seen,
                       seconds=round(time.perf_counter() - start, 2), steps=total_steps)
            writer.writerow(row)
            stream.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
