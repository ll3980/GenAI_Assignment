# Combined API verification

The repository layout was repaired and Assignment 3 was integrated with the
existing Assignment 1 and Assignment 2 APIs on 2026-10-30. The original
application revision was `e81219b`.

## Checks performed

| Check | Result |
| --- | --- |
| Dependency installation from the original frozen lockfile | Passed with Python 3.12.14 |
| Updated project metadata and lockfile | `uv lock`, `uv sync --locked --offline`, and `uv lock --check --offline` passed |
| Full test suite | **30 passed**, one upstream Starlette TestClient deprecation warning |
| Original text and embedding endpoints | Passed, including exact full pretrained-vector comparison |
| CNN architecture and actual checkpoint inference | Passed; original epoch 8 checkpoint retained |
| GAN training update, image responses, validation, and missing-weight behavior | Passed; original epoch 10 checkpoint retained |
| Shared application entry point | Root `main.py` and `app/main.py` expose the same FastAPI object and all assignment routes |
| Live Uvicorn HTTP queries | All checks in `live_api_check.json` passed |
| Docker readiness command | Executed against the live server; both trained models reported ready |
| Bash startup script | `bash -n scripts/start_docker.sh` passed |
| Python source and Docker COPY paths | Parsed/checked successfully; all required sources are present |
| CNN/GAN training CLI help and root training compatibility entry point | Passed without downloading datasets or retraining |
| English filenames and text | Checked successfully |
| Actual Docker image build and container execution | **Not run**; Docker engine unavailable in this environment |
| PowerShell startup script execution | **Not run**; PowerShell unavailable in this environment |

The live HTTP checks exercise the trained CNN, invalid image rejection,
bigram generation, both embedding methods, GAN generation, valid PNG sizes,
and repeated fixed-seed requests. The Docker readiness command was tested
against that directly launched Uvicorn service, rather than inside a Docker
container. Its base URL was overridden only for this local check.

Evidence files:

- `live_api_check.json`: successful HTTP checks and response metadata.
- `server.log`: actual Uvicorn request log.
- `gan_generated_grid.png`: 16 generated images.
- `gan_digit_0.png` through `gan_digit_3.png`: individual decoded images.

## Preserved weights and measurements

CNN checkpoint SHA-256:

```text
8105714e9968d2fbfce461ebbff60ea9dafcf00a3a840cdfc8fceb5ed6a94a19
```

GAN checkpoint SHA-256:

```text
6478bc779bb526d94397a4344a9dc9226da55a164819f0a271149f04ad16ecd7
```

Both match their original delivered checkpoints. CNN evaluation measurements
and GAN training logs were preserved and labeled as historical records.
This integration did not retrain either model or create new quality metrics.

Run the included Docker deployment on a Docker-enabled target machine and
save its actual results to `docker_evidence/` before claiming Docker success.
