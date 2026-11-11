# Assignment 4 validation

Date: November 11, 2026

## Completed

- `python -m pytest -q`: 28 passed. The suite includes the new gradient/sampling checks and existing API tests. A non-failing Starlette/httpx deprecation warning was emitted.
- `python scripts/assignment4_theory.py`: executed the embedding calculation and all autograd examples; assertions verified gradients 7, [4,2], 6, and accumulation 2 -> 5 or 3 after reset. Actual results are in theory_results.json.
- `uv lock --check --offline`: passed. This verifies manifest/lock consistency, not a Docker image build.
- A real Uvicorn server was started on localhost. `scripts/check_assignment4_api.py` sent live HTTP requests, then the server was shut down. See live_api_validation.json.
- Both new models returned HTTP 200 for health, JSON generation and PNG generation. Individual images decoded as 32x32 RGB; 16-image grids decoded as 128x128 PNGs. Repeated seeds produced identical bytes within this CPU runtime.
- Earlier text generation, 300-dimensional word embeddings, CIFAR-10 CNN prediction, MNIST GAN generation and their health routes also passed live HTTP checks.
- Both new models completed three full passes over the 50,000-image CIFAR-10 training split, for 150,000 training-image exposures each. Batch size was 128; each model completed 1,173 optimizer steps. No test images were used for fitting.
- The source was the lossless uoft-cs/cifar10 mirror. Its revision and parquet hashes are in dataset_source.json. PreparedCIFAR10 verified image sizes, row counts, label ranges and class counts.
- The supplied CPU models use eight base channels. DDPM: 200 cosine-schedule timesteps and EMA inference. EBM: 10 sampling steps during training, replay buffer with 5% restarts, and 60 sampling steps for the default API output.

## Not completed here

- Docker is unavailable in this environment. The Dockerfile is included and its app/checkpoint paths are consistent with the tested project, but `docker compose up --build` must be run on a machine with Docker.
- No authenticated GitHub account is available. A local implementation commit is recorded with the package; a remote push is still required.
- Sample quality remains preliminary. No convergence, FID, Inception Score or image-quality threshold is claimed. The notebook provides longer training with a larger model.

## Reproduce live API checks

Start the application with `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`, then in another terminal run `uv run python scripts/check_assignment4_api.py --url http://localhost:8000`.
