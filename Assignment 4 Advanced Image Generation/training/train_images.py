"""CIFAR-10 training shared by the diffusion and energy command-line entry points."""

import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from app.image_generation.models import CIFARUNet, CIFAREnergyModel
from app.image_generation.diffusion import DDPMSchedule
from app.image_generation.energy import ReplayBuffer, frozen_parameters, langevin_sample

ROOT = Path(__file__).resolve().parents[1]


def main(kind):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50 if kind == "diffusion" else 30)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--langevin-steps", type=int, default=20)
    p.add_argument("--step-size", type=float, default=10.0)
    p.add_argument("--noise-std", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=2e-4 if kind == "diffusion" else 1e-4)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--prepared-data", default=None)
    p.add_argument("--output", default=str(ROOT / f"models/cifar10_{kind}.pt"))
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42 if kind == "diffusion" else 43)
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional total-step limit for smoke runs",
    )
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.threads) < 1:
        p.error("epochs, batch size and threads must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if a.device == "auto"
        else a.device
    )
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    if a.prepared_data:
        from app.cifar_dataset import PreparedCIFAR10

        data = PreparedCIFAR10(Path(a.prepared_data), train=True, transform=transform)
    else:
        data = datasets.CIFAR10(
            a.data_dir, train=True, download=True, transform=transform
        )
    loader = DataLoader(
        data,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(a.seed),
    )
    model = (
        CIFARUNet(a.base_channels)
        if kind == "diffusion"
        else CIFAREnergyModel(a.base_channels)
    ).to(device)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        if kind == "diffusion"
        else torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.0, 0.999))
    )
    ema = (
        copy.deepcopy(model).eval().requires_grad_(False)
        if kind == "diffusion"
        else None
    )
    schedule = DDPMSchedule(a.timesteps).to(device) if kind == "diffusion" else None
    buffer = ReplayBuffer()
    steps, seen, start_epoch, history = 0, 0, 1, []
    config = (
        {"base_channels": a.base_channels, "time_dim": 128, "timesteps": a.timesteps}
        if kind == "diffusion"
        else {
            "base_channels": a.base_channels,
            "langevin_steps": a.langevin_steps,
            "step_size": a.step_size,
            "noise_std": a.noise_std,
        }
    )
    if a.resume:
        ckpt = torch.load(a.resume, map_location=device, weights_only=True)
        if ckpt["config"] != config:
            raise ValueError("Resume architecture and sampler config must match.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ema is not None:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if "replay_buffer" in ckpt:
            buffer.images = ckpt["replay_buffer"].cpu()
        steps, seen, start_epoch = ckpt["steps"], ckpt["images_seen"], ckpt["epoch"] + 1
        history = ckpt.get("history", [])
    start = time.monotonic()
    for epoch in range(start_epoch, start_epoch + a.epochs):
        model.train()
        total, real_total, negative_total, count = 0.0, 0.0, 0.0, 0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            if kind == "diffusion":
                loss = schedule.training_loss(model, images)
            else:
                real = (images + 0.005 * torch.randn_like(images)).clamp(-1, 1)
                with frozen_parameters(model):
                    negative = langevin_sample(
                        model,
                        buffer.sample(len(real), device),
                        a.langevin_steps,
                        a.step_size,
                        a.noise_std,
                    )
                buffer.update(negative)
                er, en = model(real), model(negative.detach())
                loss = (
                    er.mean()
                    - en.mean()
                    + 0.1 * (er.square().mean() + en.square().mean())
                )
                real_total += er.mean().item() * len(images)
                negative_total += en.mean().item() * len(images)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0 if kind == "diffusion" else 10.0
            )
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for avg, param in zip(ema.parameters(), model.parameters()):
                        avg.lerp_(param, 1 - min(0.999, (1 + steps) / (10 + steps)))
            steps += 1
            seen += len(images)
            count += len(images)
            total += loss.item() * len(images)
            if steps % 50 == 0:
                print(
                    f"{kind} epoch={epoch} step={steps} loss={loss.item():.6f} seconds={time.monotonic()-start:.1f}",
                    flush=True,
                )
            if a.max_steps and steps >= a.max_steps:
                break
        record = {
            "epoch": epoch,
            "steps": steps,
            "images_seen": seen,
            "loss": total / count,
            "complete_epoch": count == len(data),
            "elapsed_seconds_this_run": time.monotonic() - start,
        }
        if kind == "diffusion":
            record["noise_mse"] = total / count
        else:
            record.update(
                real_energy=real_total / count, negative_energy=negative_total / count
            )
        history.append(record)
        payload = {
            "format_version": 1,
            "model_type": "ddpm" if kind == "diffusion" else "ebm",
            "dataset": "CIFAR10",
            "image_shape": [3, 32, 32],
            "epoch": epoch,
            "steps": steps,
            "images_seen": seen,
            "train_dataset_size": len(data),
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "seed": a.seed,
        }
        if ema is not None:
            payload["ema_state_dict"] = ema.state_dict()
        else:
            payload["replay_buffer"] = buffer.images
        output = Path(a.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        results = ROOT / "results/assignment4"
        results.mkdir(parents=True, exist_ok=True)
        (results / f"{kind}_training.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(record, flush=True)
        if a.max_steps and steps >= a.max_steps:
            break
    generator = torch.Generator(device=device).manual_seed(123)
    model.eval().requires_grad_(False)
    if kind == "diffusion":
        samples = schedule.sample(ema, 16, generator)
    else:
        initial = (
            torch.rand((16, 3, 32, 32), device=device, generator=generator) * 2 - 1
        )
        samples = langevin_sample(
            model, initial, 60, a.step_size, a.noise_std, generator
        )
    save_image(
        (samples.cpu() + 1) / 2,
        ROOT / f"results/assignment4/{kind}_samples.png",
        nrow=4,
    )
