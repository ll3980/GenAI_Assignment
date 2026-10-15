"""Query the deployed CNN and the earlier endpoints using only Python's stdlib.

python scripts/check_api.py --base-url http://127.0.0.1:8000 --output results/docker_api_check.json
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def request(base_url, path, payload=None, timeout=120, image_bytes=None):
    headers = {}
    data = None
    if image_bytes is not None:
        boundary = "cnn-assignment2-upload-boundary"
        data = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\n'
            'Content-Type: image/png\r\n\r\n'
        ).encode() + image_bytes + f'\r\n--{boundary}--\r\n'.encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(base_url + path, data=data, headers=headers), timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def check_api(base_url, image_path, wait_seconds=180):
    base_url = base_url.rstrip("/")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status, body = request(base_url, "/cnn/health", timeout=3)
            if status == 200 and json.loads(body).get("status") == "ready":
                break
        except (OSError, URLError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("The CNN did not become ready at the specified base URL.")
        time.sleep(1)
    health = json.loads(body)
    require(health.get("image_size") == 64, "CNN input size is not 64.")
    checks = ["Trained CNN is ready"]

    status, body = request(base_url, "/docs")
    require(status == 200 and b"swagger" in body.lower(), "Swagger UI is unavailable.")
    status, body = request(base_url, "/openapi.json")
    require(status == 200, "OpenAPI schema is unavailable.")
    paths = json.loads(body)["paths"]
    require("post" in paths.get("/predict", {}), "POST /predict is missing.")
    require("post" in paths.get("/generate", {}), "POST /generate is missing.")
    require({"get", "post"} <= set(paths.get("/embedding", {})), "Embedding endpoints are missing.")
    checks.append("Swagger schema includes the CNN and original endpoints")

    status, body = request(base_url, "/predict", image_bytes=Path(image_path).read_bytes())
    require(status == 200, f"Image inference failed with HTTP {status}: {body.decode()}")
    prediction = json.loads(body)
    probabilities = prediction.get("probabilities", {})
    require(set(probabilities) == set(CLASSES), "Expected ten CIFAR-10 probabilities.")
    require(all(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1
                for value in probabilities.values()), "Invalid prediction probabilities.")
    require(abs(sum(probabilities.values()) - 1) < 1e-5, "Probabilities do not sum to one.")
    index = prediction.get("class_id")
    require(type(index) is int and 0 <= index < 10, "Invalid class ID.")
    require(prediction.get("class_name") == CLASSES[index], "Class name and ID disagree.")
    require(prediction["class_name"] == max(probabilities, key=probabilities.get), "Class is not the argmax.")
    require(abs(prediction["confidence"] - probabilities[CLASSES[index]]) < 1e-7, "Confidence disagrees.")
    checks.append("Uploaded image returns a class and a valid ten-class probability distribution")
    status, _ = request(base_url, "/predict", image_bytes=b"not an image")
    require(status == 400, "Invalid images should return HTTP 400.")
    checks.append("Invalid images are rejected")

    status, body = request(base_url, "/generate", {"start_word": "the", "length": 20})
    require(status == 200, "Original text generation endpoint failed.")
    generation = json.loads(body)
    words = generation.get("generated_text", "").split()
    require(1 <= len(words) <= 20 and words[0] == "the", "Invalid generated text.")
    checks.append("Original text generation endpoint remains usable")

    status, body = request(base_url, "/embedding", {"word": "apple"})
    require(status == 200, f"Original embedding endpoint returned HTTP {status}.")
    embedding = json.loads(body)
    vector = embedding.get("embedding", [])
    require(embedding.get("model") == "en_core_web_lg" and embedding.get("model_version") == "3.8.0",
            "Incorrect pretrained embedding model.")
    require(embedding.get("dimensions") == len(vector) == 300, "Expected all 300 embedding dimensions.")
    require(embedding.get("has_vector") is True and all(math.isfinite(value) for value in vector)
            and any(value != 0 for value in vector), "Invalid pretrained vector.")
    status, body = request(base_url, "/embedding?word=apple")
    require(status == 200 and json.loads(body) == embedding, "GET and POST embeddings disagree.")
    checks.append("Original GET and POST embeddings return identical complete pretrained vectors")

    return {
        "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url, "checks": checks, "cnn_health": health,
        "prediction_response": prediction, "generation_response": generation,
        "embedding_model": embedding["model"], "embedding_dimensions": embedding["dimensions"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait-seconds", type=float, default=180)
    parser.add_argument("--image", type=Path, default=Path(__file__).resolve().parents[1] / "examples/cat.png")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        evidence = check_api(args.base_url, args.image, args.wait_seconds)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    for check in evidence["checks"]:
        print(f"PASS: {check}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
