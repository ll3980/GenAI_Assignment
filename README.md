# Assignment 2: Convolutional Neural Networks

This project extends the previous Assignment 1 course API, implements the exact CNN architecture in the assignment, trains it on CIFAR-10, and exposes image classification through FastAPI. The package includes a trained checkpoint, the training script, actual evaluation results, Docker configuration, example images, and worked answers to all six theory questions.

The original `GET /`, `POST /generate`, and both `GET /embedding` and `POST /embedding` routes are retained. `app/main.py` already attaches the new CNN router. Use this merged project in your existing GitHub repository; **INTEGRATION.md** also explains how to add only the CNN module if needed.

The included CNN was trained for **10 epochs**. Validation selected **epoch 8** (69.48% validation accuracy); the selected model achieved **68.50% test accuracy**, or **6,850 correct predictions out of 10,000**. The actual history, confusion matrix, and API check results are in `results/`. All **26 tests** and the live Uvicorn HTTP checks passed in the locked dependency environment.

## Run with Docker

From this project directory, with Docker running:

```bash
docker build -t sps-cnn-assignment2 .
docker run --rm -p 8000:80 sps-cnn-assignment2
```

Open http://127.0.0.1:8000/docs. Expand `POST /predict`, click **Try it out**, choose an image from `examples/`, and click **Execute**. The response includes `class_name`, `class_id`, `confidence`, and the probability of each of the ten classes. No CIFAR-10 dataset download or CNN training is needed to serve the included checkpoint. The first build installs the same pretrained spaCy `en_core_web_lg` model used by the previous embedding endpoint.

Readiness: http://127.0.0.1:8000/cnn/health.

To query all three model endpoints over HTTP, run in a second terminal:

```bash
python scripts/check_api.py --base-url http://127.0.0.1:8000 --output results/docker_api_check.json
```

This check uses only Python's standard library and writes evidence after all checks pass. `START_HERE_CN.md` gives the short Chinese run/submission steps.

The container listens on port **80**; `-p 8000:80` maps it to port **8000** on your computer. On an ARM computer that cannot find the pinned CPU wheels, use `docker build --platform linux/amd64 -t sps-cnn-assignment2 .` and add `--platform linux/amd64` to `docker run`.

## Run locally

Use **Python 3.11 or 3.12** for the pinned PyTorch version. In Windows PowerShell, from this folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On Linux/macOS, create an environment using `python3.12 -m venv .venv`. Use `.venv/bin/python` in place of `.venv\Scripts\python.exe`. On macOS, install the PyTorch pair without `--index-url`.

## Query the API

Using Windows PowerShell in a second terminal:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@examples/cat.png"
```

On Linux/macOS, use `curl` instead of `curl.exe`. The API accepts multipart field **file**. It converts readable images to RGB and applies the same deterministic resize and normalization used for validation and testing. The classifier's output space is the ten CIFAR-10 classes. `confidence` is the largest softmax probability.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/predict` | Upload an image and return its class |
| GET | `/cnn/health` | Confirm that the trained checkpoint can load |
| GET | `/cnn/classes` | List class names in model output order |
| GET | `/docs` | Interactive API documentation |
| POST | `/generate` | Existing bigram text generation |
| GET/POST | `/embedding` | Existing full 300-dimensional spaCy word vector |

## Train again

```bash
python train.py --epochs 10 --seed 617
```

The default loader downloads the official CIFAR-10 archive through torchvision. The original images are **32 x 32**, resized to **64 x 64** because the assignment explicitly specifies that model input. Training uses a random horizontal flip; validation, test, and inference use deterministic preprocessing. RGB values are converted to tensors and normalized with mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`.

The official training set is split reproducibly into **45,000 training** and **5,000 validation** images. The checkpoint is selected by the highest validation accuracy. The separate **10,000-image test set** is evaluated once after selection. The loss is CrossEntropyLoss, receiving raw logits; softmax is applied only when producing API probabilities. Each training epoch restores `model.train()` after validation has called `model.eval()`.

If the original download is slow, an optional script prepares the University of Toronto's lossless CIFAR-10 mirror:

```bash
python -m pip install pyarrow==19.0.1
python scripts/prepare_cifar10_mirror.py --output-dir data/mirror
python train.py --prepared-data data/mirror --epochs 10 --seed 617
```

This optional preparation preserves image pixels, labels, and the official training/test partitions. It records the repository revision and source-file SHA-256 digests. The training report states which source was used. PyArrow is required only for preparing this optional cache, not for inference or ordinary torchvision training. Dataset files are excluded from Git and Docker.

Outputs: `models/cifar10_cnn.pt`, `results/training_history.csv`, `results/training_report.json`, and one example test image per class in `examples/`. The seed helps reproduce data splits and sampling; exact floating-point results can differ across devices or software versions.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Checks cover every required layer's output shape, parameter count, gradient propagation, train/eval behavior, trained-checkpoint loading, actual image-upload responses, RGB conversion, invalid uploads, and the original text generation and embedding endpoints. See `results/VERIFICATION.md` for the checks actually performed on this delivery.

## Files

| File | Role |
| --- | --- |
| `app/cnn_model.py` | Specified CNN architecture |
| `app/cnn_preprocessing.py` | Shared resize, tensor conversion, normalization, class order |
| `app/cnn_service.py` | Load trained weights and run inference |
| `app/cnn_api.py` | Router to add to the existing API |
| `app/main.py` | Merged FastAPI entry point, retaining the earlier routes |
| `app/bigram_model.py` and `app/embedding_model.py` | Original Assignment 1 implementations |
| `train.py` | CIFAR-10 training, validation selection, final test evaluation |
| `models/cifar10_cnn.pt` | Trained weights and preprocessing metadata |
| `Dockerfile` | CPU inference deployment |
| `pyproject.toml` and `uv.lock` | Updated UV dependency specification and lockfile |
| `ARCHITECTURE.md` | Layer dimensions and parameter calculations |
| `INTEGRATION.md` | Existing-API and GitHub instructions |
| `Assignment2_Theory_LiuyangLi.pdf` | English worked answers to Questions 1-6 |
| `THEORY_ANSWERS.md` | Editable source of the worked answers |

## Submission

The assignment rubric also requires a GitHub commit and a Docker deployment that works on the instructor's computer. Merge the CNN router into the existing course project, commit the code **and the trained checkpoint**, and include the repository link with your submission. Upload the theory PDF, or this project package if that is the required submission format. Code files alone do not establish that a GitHub commit or Docker run has happened.

References: the supplied Assignment2.pdf; https://www.cs.toronto.edu/~kriz/cifar.html; https://docs.pytorch.org/docs/stable/notes/modules.html; https://docs.pytorch.org/docs/stable/notes/autograd.html; https://pytorch.org/get-started/previous-versions/.
