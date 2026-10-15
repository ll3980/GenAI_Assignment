# Add the CNN to your existing course API

The assignment asks you to add classification to the API from the previous modules. This package has already been merged with the earlier `Assignment1_API_Code_Final.zip`: the bigram generator, embedding implementation, and original endpoint request/response formats are retained. The new CNN router is attached in `app/main.py`; dependencies and deployment files are updated. Put this merged code into your existing GitHub project, then commit and push it.

The instructions below describe the small additions if you prefer to copy only the CNN module into another checkout of the course project.

## 1. Copy the new files

Copy these files into the existing project, retaining their relative paths:

```text
app/cnn_model.py
app/cnn_preprocessing.py
app/cnn_service.py
app/cnn_api.py
models/cifar10_cnn.pt
```

Also copy `train.py`, `app/cifar_dataset.py`, `scripts/prepare_cifar10_mirror.py`, the theory PDF, `ARCHITECTURE.md`, and the result files if you want all assignment source and evidence in that repository. Inference needs only the first five files and their dependencies.

## 2. Attach the router

In your existing `app/main.py`, add this import:

```python
from app.cnn_api import router as cnn_router
```

After the existing `app = FastAPI(...)` line, add:

```python
app.include_router(cnn_router)
```

Keep the existing endpoint definitions. The new routes are `POST /predict`, `GET /cnn/health`, and `GET /cnn/classes`. If your existing app already uses `/predict`, choose another route name in `cnn_api.py` and update the README/query examples accordingly.

The classifier uses a cached service, so the existing app's startup/lifespan function can stay as it is. Missing or incompatible checkpoints produce a clear HTTP 503 response instead of returning an untrained model's prediction.

## 3. Update dependencies

PyTorch 2.5.1/torchvision 0.20.1 support the Python versions used by the standalone project: 3.11 and 3.12. If your `pyproject.toml` currently requires Python 3.13 or newer, change its `requires-python` to `">=3.11,<3.13"` before the following commands. If your course project uses UV, select Python 3.12 and add the new dependencies to the existing project rather than replacing its dependency file:

```bash
uv python pin 3.12
uv add --python 3.12 "torch==2.5.1" "torchvision==0.20.1" "python-multipart==0.0.20" Pillow
uv sync --python 3.12
```

Retain existing dependencies, including spaCy if used by your embedding endpoint. Commit the updated `pyproject.toml` and `uv.lock`. Your existing project's resolver should decide a compatible NumPy version; the standalone `requirements.txt` pins the environment used for this package.

For a Linux CPU-only UV deployment, a named explicit CPU index avoids unnecessarily downloading CUDA packages. Add this configuration before `uv add` if using that deployment, or refresh the lockfile with `uv lock` after adding it:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }
torchvision = { index = "pytorch-cpu", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }
```

If `[tool.uv.sources]` already exists, add these two entries to that table. For a pip-based project, merge the new libraries into the existing requirements. The merged requirements, `pyproject.toml`, lockfile and Dockerfile can be used directly when testing this package on its own.

## 4. Include weights in Docker

The original class example copied `app/` into `/code/app`. Add this line to the **existing** Dockerfile so the model can be loaded inside the container:

```dockerfile
COPY ./models /code/models
```

The old `uv sync --frozen` step will require the updated `pyproject.toml` and `uv.lock`. Keep the app listening on internal port 80, as in the class setup.

```bash
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```

Then open `/docs`, verify the old endpoints are still present, and upload an image to `/predict`. Confirm `/cnn/health` reports `ready`.

## 5. Commit to GitHub

From the existing repository, review the changes and stage the relevant files:

```bash
git status
git add app/cnn_model.py app/cnn_preprocessing.py app/cnn_service.py app/cnn_api.py app/main.py models/cifar10_cnn.pt Dockerfile
```

Stage whichever dependency files your project uses (`pyproject.toml` and `uv.lock`, or `requirements.txt`) and any training/theory/result files copied above. Then:

```bash
git commit -m "Add Assignment 2 CIFAR-10 CNN classifier and image API"
git push
```

Ensure `models/cifar10_cnn.pt` is tracked. The included checkpoint is only a few megabytes, so it can be committed normally. Include the actual GitHub repository URL with the submission; no repository URL is assumed in this package.

Packaging alone does not complete this step: these GitHub commands and the original project's Docker check need to be run in your own repository.

## 中文操作提示

先用本包运行 `/docs`，确认图片分类正常。再将 CNN 的四个模块和 `models/cifar10_cnn.pt` 复制到原项目，在原 `app/main.py` 增加上面的 import 和 `include_router` 两行。更新依赖，并在原 Dockerfile 增加复制 `models` 的一行。最后在原项目运行 Docker、确认旧接口和新接口都可用，然后提交并推送 GitHub。

这个包已经基于之前的 Assignment 1 API 完成合并。原 `/generate` 和 `/embedding` 接口保留，同时加入 `/predict`。将本包代码放入原 GitHub 项目后，运行 Docker、确认接口，再提交并推送。GitHub 仓库地址尚未提供，因此这里没有修改远程仓库。
