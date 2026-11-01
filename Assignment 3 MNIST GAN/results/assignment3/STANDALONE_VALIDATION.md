> Historical validation record from the original assignment package.
> See `results/integration/VERIFICATION.md` for the current merged API checks.

# Validation record

## Completed training

- Data: all 60,000 MNIST training images; digit labels unused.
- Run: 10 complete epochs, 469 minibatches per epoch, 4,690 alternating
  D/G training batches (4,690 updates of each network).
- Seed: 42. Batch size: 128. CPU threads: 4.
- Optimizer: Adam, learning rate 0.0002, betas (0.5, 0.999).
- Runtime: PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, Python 3.12.
- Sum of recorded epoch durations: 586.78 seconds.
- Final mean discriminator loss: 0.478690.
- Final mean generator loss: 2.314699.
- The included inference checkpoint is the final epoch, not a selected earlier
  checkpoint. GAN losses alone are not a quality metric.
- Generator parameters: 765,761. Discriminator parameters: 138,817.
- `artifacts/environment.txt` records installed dependency versions.

Command actually executed:

```bash
python train.py --epochs 10 --batch-size 128 --device cpu --threads 4 --seed 42
```

The portable checkpoint is `artifacts/generator.pt`. Its SHA-256 is:

```text
6478bc779bb526d94397a4344a9dc9226da55a164819f0a271149f04ad16ecd7
```

`artifacts/training_checkpoint.pt` contains the final G/D and optimizer states.
The training data are not bundled; train.py downloads MNIST when needed.
Fixed-noise samples for every epoch are in `artifacts/samples/`. Visual review
of the final samples checks that generated outputs resemble varied digits;
some samples remain ambiguous. No FID, classifier accuracy, or test-set
generalization result is claimed.

## Functional checks

`python -m pytest -q`: **3 passed**. The dependency emitted one deprecation
warning about an AnyIO alias used by Starlette's test client; it did not fail
the checks. The tests cover:

- G output shape/range and D probability shape/range.
- Finite losses and actual parameter updates for both networks.
- Missing-checkpoint HTTP 503 behavior.
- Valid 28x28 grayscale PNGs and a 224x56 grid of 16 images.
- Identical responses for repeated fixed-seed requests.
- HTTP 422 for out-of-range counts and negative seeds.

`python -m pip check`: **No broken requirements found**.

The theory calculations were also checked using actual PyTorch layers:
Q1 gives 17x17; Q2 gives 24x24; Q4 gives 32x32; Q5 with affine=False
and eps=0 gives [-0.9045341, 0.3015113, 1.5075567, -0.9045341].
Both pages of the theory PDF were rendered and visually inspected.

## Actual HTTP inference check with final weights

A Uvicorn process was started locally and queried over HTTP, with the final
epoch-10 checkpoint. The checks succeeded:

| Request | Result |
| --- | --- |
| GET /health | HTTP 200 |
| GET /gan/health | HTTP 200; ready; epoch 10 |
| POST /gan/generate, count=4, seed=42 | HTTP 200; four decoded 28x28 grayscale PNGs |
| Same POST repeated | Identical JSON/image outputs |
| GET /gan/generate.png?count=16&seed=42 | HTTP 200; PNG grid, 224x56 |

The saved JSON responses and decoded PNGs are in `artifacts/api_evidence/`.
The reusable client is `verify_api.py`.

## Work still required for the course submission

- **Docker build/run has not been verified**: Docker was unavailable in this
  environment. The Dockerfile, Compose configuration, health check, startup/verification scripts,
  and x86-64 build/run commands are included.
- **Module 6 integration has not been performed**: the original class API
  source was not supplied. `gan.api.router` is ready to include in that app;
  see the README integration instructions.
- **No GitHub commit/push was made**: no target class repository was supplied.
  Merge into the actual class repository, commit, push, and record its real
  URL and commit hash before submitting.

## Docker additions (2026-10-30)

Added `compose.yaml`, `start_docker.ps1`, `start_docker.sh`, and
`DOCKER_GUIDE.md`. The Dockerfile now copies the inference verification client
and checks model readiness through `/gan/health`. The startup scripts wait for
readiness, query the actual API inside the container, and copy evidence to
`docker_evidence/` on the user's machine when run successfully.

Checks performed for these additions:

- Bash startup script: `bash -n` passed.
- Compose YAML parsed successfully; the service settings, platform, and build
  paths were checked. This is not an executed `docker compose config` check.
- All Dockerfile COPY sources exist in the project.
- The exact Python command embedded in the Docker HEALTHCHECK was executed
  against a local test HTTP server: ready/HTTP 200 passed, HTTP 503 failed,
  and an HTTP 200 response with a non-ready status failed as expected.
- The trained generator SHA-256 still matches the checkpoint recorded above.

The PowerShell startup script has not been executed in PowerShell here.
Actual Docker image build and container execution remain unverified: this
preparation environment has no Docker engine, and its permissions do not allow
creating a nested user namespace for a container runtime. The earlier training
and direct FastAPI checks above remain the prior run's record. No model training
or theoretical answers were changed in this update.
