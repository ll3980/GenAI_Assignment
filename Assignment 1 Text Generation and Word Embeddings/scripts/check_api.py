"""Check the real HTTP API using only the Python standard library.

Run against the published port on the host, or from inside the container.
Exit with a nonzero status if an endpoint or vector check fails.
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def request(base_url: str, path: str, payload=None, timeout: float = 120):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if payload is None else {"Content-Type": "application/json"}
    http_request = Request(base_url + path, data=data, headers=headers)
    try:
        with urlopen(http_request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def check_api(base_url: str, wait_seconds: float = 180) -> dict:
    """Return evidence only after all live checks pass."""
    base_url = base_url.rstrip("/")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status, body = request(base_url, "/", timeout=3)
            if status == 200 and json.loads(body) == {"Hello": "World"}:
                break
        except (OSError, URLError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"The API did not become ready at {base_url}.")
        time.sleep(1)

    checks = ["HTTP server responds"]

    status, body = request(base_url, "/docs")
    require(status == 200 and b"swagger" in body.lower(), "Swagger UI is unavailable.")
    status, body = request(base_url, "/openapi.json")
    require(status == 200, "OpenAPI schema is unavailable.")
    paths = json.loads(body)["paths"]
    require("post" in paths.get("/generate", {}), "Missing POST /generate.")
    require({"get", "post"} <= set(paths.get("/embedding", {})), "Missing embedding endpoints.")
    checks.append("Swagger UI and API schema are available")

    status, body = request(base_url, "/embedding", {"word": "apple"})
    require(status == 200, f"POST /embedding returned HTTP {status}: {body.decode()}")
    embedding = json.loads(body)
    vector = embedding.get("embedding", [])
    require(embedding.get("word") == "apple", "Incorrect query word.")
    require(embedding.get("model") == "en_core_web_lg", "Incorrect vector model.")
    require(embedding.get("model_version") == "3.8.0", "Incorrect model version.")
    require(embedding.get("has_vector") is True, "No pretrained vector was returned.")
    require(embedding.get("dimensions") == len(vector) == 300, "Expected all 300 vector values.")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in vector), "The vector contains invalid values.")
    require(any(x != 0 for x in vector), "The vector is entirely zero.")
    checks.append("POST /embedding returns a complete, finite, nonzero 300-dimensional vector")

    status, body = request(base_url, "/embedding?word=apple")
    require(status == 200 and json.loads(body) == embedding, "GET and POST embeddings differ.")
    checks.append("GET and POST return the same embedding")

    status, _ = request(base_url, "/embedding", {"word": "apple banana"})
    require(status == 422, "Invalid input should return HTTP 422.")
    status, _ = request(base_url, "/embedding", {"word": "zzzzzzqqqqxxxyyyzzzzzz"})
    require(status == 404, "An unknown word should return HTTP 404.")
    checks.append("Invalid and unknown words produce the documented HTTP errors")

    status, body = request(base_url, "/generate", {"start_word": "the", "length": 20})
    require(status == 200, f"POST /generate returned HTTP {status}.")
    generation = json.loads(body)
    words = generation.get("generated_text", "").split()
    require(1 <= len(words) <= 20 and words[0] == "the", "Invalid generated text.")
    checks.append("The original text generation endpoint remains queryable")

    return {
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "checks": checks,
        "embedding_response": embedding,
        "generation_response": generation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait-seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, help="Save the actual checked responses as JSON.")
    args = parser.parse_args()
    try:
        evidence = check_api(args.base_url, args.wait_seconds)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    for check in evidence["checks"]:
        print(f"PASS: {check}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print("All API checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
