# Verification performed for this delivery

## Model and data

The specified CNN was trained for 10 epochs on a seeded 45,000/5,000 training/validation split of the official CIFAR-10 training partition. The University of Toronto's lossless CIFAR-10 mirror was used because the original archive server was slow. Repository revision, source-file hashes, and preparation details are recorded in `training_report.json`.

Model selection used validation accuracy only. The selected checkpoint is from epoch 8 (validation accuracy **69.48%**). Its final test accuracy is **68.50%**, or **6,850 / 10,000**. The recorded confusion matrix contains exactly 10,000 observations and its diagonal sums to 6,850. The model has exactly **825,398** trainable parameters.

The checkpoint is a real trained state dictionary with architecture, input size, normalization, label-order and selection metadata. Its SHA-256 is:

```text
8105714e9968d2fbfce461ebbff60ea9dafcf00a3a840cdfc8fceb5ed6a94a19
```

Training used Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, NumPy 1.26.4, Pillow 11.2.1, seed 617, batch size 128, four CPU threads, and zero DataLoader workers. The merged API environment uses NumPy 2.2.6 to support the retained spaCy dependency. The trained weights were verified in that updated, locked environment. Repeated training can vary across environments; the delivered checkpoint and its measured results are preserved.

## Checks

| Check | Result | Evidence |
| --- | --- | --- |
| Required layer outputs, parameter count and finite gradients | Passed | `tests/test_cnn.py` |
| Train/eval behavior described in Question 6 | Passed | Fresh model defaults to training; this architecture's outputs agree in train/eval, and eval does not disable gradients |
| Checkpoint loading and image-upload responses | Passed | Real checkpoint, RGB/grayscale/RGBA handling, argmax/class ID/probability consistency |
| Invalid inputs and missing weights | Passed | 400 for empty/invalid images, 413 for oversized uploads, 422 for missing uploads, 503 for missing checkpoint |
| Previous text generation and embedding behavior | Passed | Original tests retained; complete pretrained vector agrees with spaCy's classroom operation |
| Full test suite in the frozen UV environment | **26 passed** | `python -m pytest tests -q` |
| Actual Uvicorn HTTP requests | Passed | `live_api_check.json`, including CNN, Swagger, bigram generation and both embedding methods |
| Updated lockfile and dependency installation | Passed | `uv lock` and `uv sync --locked` completed with Python 3.12.14 |
| Theory PDF | Passed | Six answers extracted and checked; both rendered pages visually inspected |
| Docker build/container start | **Not run in this environment** | No Docker engine/CLI available; run the included Docker/Compose commands on the target machine |
| Remote GitHub commit/push | **Not performed** | A target repository URL was not supplied; the code and instructions are ready for committing |

Pytest produced one Starlette TestClient deprecation notice and no failures. The live HTTP checks used Uvicorn with the same locked libraries, rather than TestClient. The temporary validation server was stopped after the checks.

The Dockerfile installs frozen dependencies, includes trained CNN weights and example images, checks both pretrained models during the build, serves FastAPI on port 80, and checks CNN readiness. Actual container execution should be confirmed using:

```bash
docker compose up --build
python scripts/check_api.py --base-url http://127.0.0.1:8000 --output results/docker_api_check.json
```

Only a successful execution of that command creates Docker API evidence. The included `live_api_check.json` records the local Python server checks, not a Docker run.

## Previous project

The earlier `Assignment1_API_Code_Final.zip` supplied the existing API. `app/bigram_model.py` and `app/embedding_model.py` were copied byte-for-byte. The existing endpoint definitions and request/response formats were retained in `app/main.py`, with the CNN router added and the API title/version updated. The earlier assignment package itself was left unchanged.
