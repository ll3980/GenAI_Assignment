# Repository organization and integration fixes

The inspected `main` revision was `e81219b`. Several files were present but
stored at paths that disagreed with imports, tests, and Docker COPY commands.
This layout suggests that folder structure was lost during upload.

| Previous location | Canonical location |
| --- | --- |
| Root `cnn_*.py` and `cifar_dataset.py` | `app/` |
| Root `cifar10_cnn.pt` | `models/cifar10_cnn.pt` |
| Root example PNGs | `examples/` |
| Root `test_cnn.py` | `tests/test_cnn.py` |
| Root CNN training implementation | `training/train_cnn.py` |
| Root CIFAR-10 preparation script | `scripts/prepare_cifar10_mirror.py` |
| Root CNN history and report | `results/assignment2/` |
| Root theory and architecture documents | `docs/assignment2/` |
| `download` | `.dockerignore` |
| `download (2)` | `.gitignore` |
| `download (3)` | `.python-version` |

The two `main.py` files exposed different APIs. Docker launched the older
`app/main.py`, while the CNN router was registered only in the root copy.
The combined app now lives in `app/main.py` and registers CNN and GAN routers
alongside the original text and embedding routes. The root `main.py` forwards
to the same app.

Identical root copies of the bigram model, embedding model, and original API
tests were removed. Their canonical implementations remain under `app/` and
`tests/`. The empty duplicate initializer was removed. The original CNN
checkpoint and training measurements were preserved.

Assignment 3 adds `app/gan/`, `models/mnist_generator.pt`, a training entry
point, tests, English theory answers, and original training evidence. GAN
weights use `GAN_MODEL_PATH`; the CNN retains its existing `MODEL_PATH`
behavior so the two loaders can use different checkpoints.

All project instructions and filenames are in English. Historical validation
records are labeled; current integration results are in `results/integration/`.

If an actual Git error mentions `origin`, inspect remotes with `git remote -v`.
The repository URL is `https://github.com/ll3980/GenAI_Assignment.git`.
The exact error is needed to distinguish a missing remote from authentication
or URL problems.
