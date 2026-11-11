# Assignment 4: Advanced Image Generation

Liuyang Li

This project extends the existing course FastAPI with two unconditional CIFAR-10 generators: a DDPM and an energy-based model. Run commands from this directory, which contains `app/`, `models/`, and `Dockerfile`.

## Docker deployment

```bash
docker compose up --build
```

Open http://localhost:8000/docs . The image bundles model weights and does not download CIFAR-10 for inference. The first build needs internet to install Python dependencies and the existing spaCy word-vector package. The root `app/main.py` now registers the earlier CNN/GAN routes as well as the new generators; `main.py` aliases the same app.

| Method | Endpoint | Input / result |
|---|---|---|
| GET | `/health` | Readiness and training metadata for both new models |
| GET | `/diffusion/health` | DDPM checkpoint metadata |
| POST | `/diffusion/generate` | `count`, `seed`; base64-encoded 32x32 RGB PNGs |
| GET | `/diffusion/generate.png` | A PNG grid with count/seed query parameters |
| GET | `/energy/health` | Energy-model checkpoint metadata |
| POST | `/energy/generate` | `count`, `seed`, `steps`; base64 PNGs |
| GET | `/energy/generate.png` | A PNG grid with count/seed/steps parameters |
| POST | `/generate` | Existing bigram text generator |
| GET or POST | `/embedding` | Existing word-vector endpoint |
| POST | `/predict` | Existing CIFAR-10 classifier |
| POST | `/gan/generate` | Existing MNIST GAN |

Example requests:

```bash
curl -X POST http://localhost:8000/diffusion/generate -H "Content-Type: application/json" -d '{"count":2,"seed":123}'
curl -X POST http://localhost:8000/energy/generate -H "Content-Type: application/json" -d '{"count":2,"seed":123,"steps":60}'
curl "http://localhost:8000/diffusion/generate.png?count=16&seed=123" -o diffusion.png
curl "http://localhost:8000/energy/generate.png?count=16&seed=123&steps=60" -o energy.png
```

Use `curl.exe` in Windows PowerShell, or enter the JSON through `/docs`. The models are unconditional and do not accept text prompts. `count` is 1-16; the optional seed is nonnegative; energy sampling allows 1-500 steps. A repeated seed is reproducible within the same runtime/device. Missing or incompatible checkpoints return HTTP 503 rather than random-weight output.

## Local development

```bash
uv sync --locked
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
uv run pytest -q
uv run python scripts/assignment4_theory.py
```

Python 3.11 or 3.12 is supported. The repository's older Assignment 3 folder is historical; the root entry point is the integrated application for this submission.

## Code organization

| Location | Responsibility |
|---|---|
| `app/image_generation/models.py` | Sinusoidal embedding, three-stage U-Net, scalar energy CNN |
| `app/image_generation/diffusion.py` | Forward noise, cosine schedule, MSE and reverse DDPM sampling |
| `app/image_generation/energy.py` | Input gradients, weight freezing, short-run Langevin sampling, replay buffer |
| `app/image_generation/api.py` | Checkpoint validation, JSON and PNG routes |
| `training/train_images.py` | Shared CIFAR-10 training loop and checkpointing |
| `training/train_diffusion.py` and `training/train_energy.py` | CLI entry points |
| `notebooks/Assignment4_Training.ipynb` | Self-contained Colab training notebook |
| `docs/assignment4/THEORY_ANSWERS.md` | All nine questions with derivations |
| `results/assignment4/` | Actual metrics, test results, API validation and generated samples |

## Training and limitations

Training uses the CIFAR-10 training split only, native 32x32 RGB images, and normalization to [-1,1]. Labels are ignored. The supplied checkpoints are small CPU demonstration models. Check their exact architecture, completed epochs, images seen and optimizer steps in the checkpoint metadata and `*_training.json`. Generated samples can remain noisy; no convergence, FID or Inception Score claim is made.

Use the included Colab notebook for a longer GPU run. It embeds its own model/training source and does not depend on a GitHub push. The normal training configuration uses a larger 32-channel model:

```bash
python -m training.train_diffusion --epochs 50 --base-channels 32 --batch-size 64
python -m training.train_energy --epochs 30 --base-channels 32 --batch-size 64 --langevin-steps 20
```

The default dataset loader is torchvision; it downloads CIFAR-10 on the first training run. `--prepared-data <directory>` optionally reads a lossless University of Toronto/Hugging Face mirror prepared with `scripts/prepare_cifar10_mirror.py` (requires pyarrow). Official training/test splits are preserved; data provenance is recorded when that path is used.

To continue a supplied checkpoint, match its configuration. The supplied small models use `base_channels=8`, diffusion `timesteps=200`, and energy `langevin_steps=10`:

```bash
python -m training.train_diffusion --resume models/cifar10_diffusion.pt --base-channels 8 --timesteps 200 --epochs 20
python -m training.train_energy --resume models/cifar10_energy.pt --base-channels 8 --langevin-steps 10 --epochs 20
```

`--epochs` means additional epochs when resuming. Optimizer state and the EBM replay buffer are restored; the original data/RNG stream is not replayed bit for bit. Checkpoints are saved atomically each epoch. Restart the API or rebuild Docker after replacing weights.

The DDPM predicts Gaussian noise, minimizing pixel-averaged MSE. The cosine schedule has 200 steps and a near-zero terminal signal; reverse sampling uses every step and adds no noise at the final step. EMA weights are used for inference. The embedding convention matches the theoretical answers.

During EBM sampling, weights are frozen while the image requires gradients. `autograd.grad` computes only image derivatives, and each update is detached before the next iteration. The subsequent contrastive training loss uses detached negatives to update weights: mean(E(real)) - mean(E(negative)) plus squared-energy regularization. A replay buffer uses 5% fresh-noise restarts. The sampler has finite steps, clipped gradients/pixels and small added noise; it is a synthesis approximation rather than exact equilibrium MCMC.

## Submission status

See `results/assignment4/VALIDATION.md` for checks actually completed. The preparation environment has no Docker runtime and no authenticated GitHub account. Docker execution and a remote push cannot be marked complete from local algorithm/API tests.

After copying the project files into your local clone of https://github.com/ll3980/GenAI_Assignment , inspect `git status`, then commit and push:

```bash
git add app models training scripts tests docs notebooks main.py Dockerfile compose.yaml pyproject.toml uv.lock requirements.txt requirements-dev.txt README.md .gitignore .dockerignore .gitattributes results/assignment4
git commit -m "Add CIFAR-10 diffusion and energy generators for Assignment 4"
git push
```

Do not commit the dataset or virtual environment. Confirm that the new code and two checkpoints appear on GitHub, then run the Docker deployment and upload the requested submission to CourseWorks.
