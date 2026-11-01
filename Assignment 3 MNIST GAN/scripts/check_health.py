"""Docker readiness check for both trained image models."""

import json
import os
from urllib.request import urlopen


def main():
    base_url = os.environ.get("API_HEALTH_BASE_URL", "http://127.0.0.1:80").rstrip("/")
    for path in ("/cnn/health", "/gan/health"):
        with urlopen(base_url + path, timeout=3) as response:
            if response.status != 200 or json.load(response).get("status") != "ready":
                raise RuntimeError(f"Model is not ready: {path}")


if __name__ == "__main__":
    main()
