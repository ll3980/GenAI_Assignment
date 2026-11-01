# Generative AI Assignments

One FastAPI application serves the course's bigram text generator, pretrained
spaCy word embeddings, CIFAR-10 CNN classifier, and MNIST GAN generator.
The canonical entry point is `app.main:app`. Both image-model checkpoints are
included under `models/`; serving them does not require retraining.

## Start with Docker

With Docker running, execute from the repository root:

```bash
docker compose up --build -d --wait --wait-timeout 180
docker compose ps
```

Open http://localhost:8000/docs or view generated digits at
http://localhost:8000/gan/generate.png?count=16&seed=42.
The container listens on port 80, mapped to host port 8000. Compose selects
Linux/amd64 for the pinned CPU wheels. The first build downloads dependencies,
including the original pretrained spaCy model.

For startup, API verification, and saved evidence:

```powershell
# Windows PowerShell
.\scripts\start_docker.ps1
```

```bash
# macOS/Linux
bash scripts/start_docker.sh
```

See [DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md). Actual Docker image build and
container execution still need verification on a Docker-enabled machine.
Stop the service with `docker compose down`.

## Run locally

Use Python 3.12 and UV with the included lockfile:

```bash
uv sync --locked
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The original embedding endpoint uses `en_core_web_lg` 3.8.0 and returns its
complete 300-dimensional vector. The root `main.py` forwards to the same app
for compatibility with `uvicorn main:app`.

## Endpoints

| Assignment | Method | Path | Purpose |
| --- | --- | --- | --- |
| 1 | GET | `/` | Original status response |
| 1 | POST | `/generate` | Bigram text generation |
| 1 | GET/POST | `/embedding` | Full pretrained word vector |
| 2 | POST | `/predict` | CIFAR-10 image classification |
| 2 | GET | `/cnn/health` | Trained CNN readiness |
| 2 | GET | `/cnn/classes` | Classifier output classes |
| 3 | POST | `/gan/generate` | Generate 1-64 PNG images as base64 |
| 3 | GET | `/gan/generate.png` | Return a generated image grid |
| 3 | GET | `/gan/health` | Trained GAN readiness and epoch |

For GAN generation, send `{"count":4,"seed":42}` in `/docs`. This GAN takes
random noise rather than a digit class as input.

## Repository organization

| Directory | Contents |
| --- | --- |
| `app/` | API entry point, original models, CNN modules, and `gan/` package |
| `models/` | CNN and GAN inference checkpoints |
| `training/` | CNN and GAN training entry points |
| `scripts/` | API checks, Docker readiness/startup, optional dataset preparation |
| `tests/` | API, CNN, GAN, and shared-entry-point regression tests |
| `examples/` | Classifier images and earlier API examples |
| `docs/assignment2/` | CNN architecture, integration notes, and theory answers |
| `docs/assignment3/` | GAN architecture notes and theory answers |
| `results/assignment1/` | Historical Assignment 1 validation |
| `results/assignment2/` | CNN training and evaluation records |
| `results/assignment3/` | GAN logs, fixed-noise samples, and standalone checks |
| `results/integration/` | Current combined-service verification |

[REPOSITORY_FIX.md](docs/REPOSITORY_FIX.md) explains the corrected upload paths.
The incorrectly named download files were restored as `.dockerignore`,
`.gitignore`, and `.python-version`. Canonical implementations live under
`app/`; the root `main.py` and `train.py` are compatibility entry points.

## Verify

```bash
uv run pytest tests -q
```

With the server running, query all assignments over HTTP:

```bash
uv run python scripts/check_api.py --base-url http://127.0.0.1:8000 --output results/integration/live_api_check.json
```

The client verifies CNN image upload, the original text/embedding endpoints,
and GAN PNG dimensions and fixed-seed reproducibility. Successful checks save
JSON evidence and generated PNGs. See
[the verification record](results/integration/VERIFICATION.md) for actual
results and deployment limitations.

## Train

From the repository root:

```bash
uv run python -m training.train_cnn --epochs 10 --seed 617
uv run python -m training.train_gan --epochs 20 --seed 42 --device cpu
```

CNN training saves `models/cifar10_cnn.pt` and `results/assignment2/`.
GAN training saves `models/mnist_generator.pt` and `results/assignment3/`.
Use the output-path options to preserve previous runs. Restart the API after
changing checkpoints. The root `train.py` remains a CNN training entry point.

The included CNN is the original selected epoch 8 model. The GAN is the
original final epoch 10 model trained on all 60,000 MNIST training images.
Their records are preserved; integration does not claim new training or
new accuracy measurements.

## GitHub submission

Continue using `ll3980/GenAI_Assignment`. Review the organized changes and
commit/push the resulting branch. Git preserves directory structure, including
hidden configuration files. Confirm that Docker runs on the target machine
before submitting its deployment evidence.

For the downloadable update patch and Git remote troubleshooting, follow
[UPDATE_GITHUB.md](docs/UPDATE_GITHUB.md).
