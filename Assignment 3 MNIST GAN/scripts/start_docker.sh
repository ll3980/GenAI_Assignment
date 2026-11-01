#!/usr/bin/env bash
# Build, start, wait for the trained model, and save actual Docker API evidence.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed. Install/start Docker and run this script again." >&2
    exit 1
fi
if [[ ! -f models/mnist_generator.pt ]]; then
    echo "models/mnist_generator.pt is missing. Extract the full archive first." >&2
    exit 1
fi

docker compose version
docker info --format '{{.ServerVersion}}'
docker compose up --build -d --wait --wait-timeout 180
docker compose exec -T api python scripts/check_api.py --base-url http://127.0.0.1:80 --output /tmp/course-api-evidence/verification.json

mkdir -p docker_evidence
docker compose cp 'api:/tmp/course-api-evidence/.' docker_evidence/
docker compose ps > docker_evidence/compose_ps.txt
docker compose logs --no-color api > docker_evidence/container.log

printf 'Docker API verification passed. Evidence: docker_evidence/\n'
printf 'API docs: http://localhost:%s/docs\n' "${GAN_PORT:-8000}"
printf 'Generated digits: http://localhost:%s/gan/generate.png?count=16&seed=42\n' "${GAN_PORT:-8000}"
printf 'Stop the service when finished: docker compose down\n'
