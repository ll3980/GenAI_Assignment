# Docker deployment for the combined course API

## Start and query

With Docker running, open a terminal in the repository root:

```bash
docker version
docker compose version
docker compose up --build -d --wait --wait-timeout 180
docker compose ps
```

The build installs frozen UV dependencies and the original spaCy model,
then copies `app/`, `models/`, `scripts/`, and `examples/`. Build checks load
spaCy, the CNN, and the GAN. Serving inference does not require downloading
the image datasets or training either model.

The service is named `api`. It listens on internal port 80, mapped to host
port 8000. The health check queries both `/cnn/health` and `/gan/health`.
Open:

- `http://localhost:8000/docs`
- `http://localhost:8000/cnn/health`
- `http://localhost:8000/gan/health`
- `http://localhost:8000/gan/generate.png?count=16&seed=42`

Upload `examples/cat.png` to POST `/predict` for classification.
Send `{"count":4,"seed":42}` to POST `/gan/generate` for generation.
The original `/generate` and GET/POST `/embedding` routes are retained.

## Automatic verification

Windows PowerShell:

```powershell
.\scripts\start_docker.ps1
```

macOS/Linux:

```bash
bash scripts/start_docker.sh
```

These scripts start the service, wait for readiness, run the combined API
client inside the container, and copy results to `docker_evidence/` on the
host. Evidence includes `verification.json`, generated PNGs, container status,
and logs. Successful execution is required before describing them as actual
Docker results.

Manual verification inside the running container:

```bash
docker compose exec -T api python scripts/check_api.py --base-url http://127.0.0.1:80 --output /tmp/course-api-evidence/verification.json
```

Create a host directory named `docker_evidence`, then copy the output:

```bash
docker compose cp api:/tmp/course-api-evidence/. docker_evidence/
```

If PowerShell blocks `.ps1` execution, use the manual commands above.

## Logs, ports, and stopping

```bash
docker compose logs --tail 100 api
docker compose down
```

If host port 8000 is occupied, set `GAN_PORT=8001` before starting Compose.
For example, in PowerShell:

```powershell
$env:GAN_PORT = "8001"
docker compose up --build -d --wait --wait-timeout 180
```

Then use `http://localhost:8001/docs`. The internal port remains 80.
Compose selects Linux/amd64 for the pinned CPU wheels; ARM hosts may use
emulation. Start the Docker engine if `docker version` cannot show a Server
section. Update Compose if `--wait` is unsupported, or inspect readiness
manually with `docker compose ps`. Inspect build output or container logs
for specific startup errors.

Actual Docker build/run has not been verified in this preparation environment
because no Docker engine is available. Current direct-server evidence is
recorded separately under `results/integration/`.
