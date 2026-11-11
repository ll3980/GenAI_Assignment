"""Verify live HTTP inference: python scripts/check_assignment4_api.py --url http://localhost:8000."""

import argparse, base64, io, json, urllib.request
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "results/assignment4"
    out.mkdir(parents=True, exist_ok=True)
    result = {"transport": "live HTTP"}

    def request(path, body=None, content_type="application/json"):
        data = (
            body
            if isinstance(body, bytes)
            else (None if body is None else json.dumps(body).encode())
        )
        req = urllib.request.Request(
            args.url + path, data=data, headers={"Content-Type": content_type}
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            assert response.status == 200
            return response.read(), response.headers.get_content_type()

    for kind in ["diffusion", "energy"]:
        raw, _ = request(f"/{kind}/health")
        health = json.loads(raw)
        assert health["training_steps"] > 0
        body = {"count": 2, "seed": 123}
        if kind == "energy":
            body["steps"] = 60
        raw, _ = request(f"/{kind}/generate", body)
        payload = json.loads(raw)
        assert len(payload["images"]) == 2
        for encoded in payload["images"]:
            im = Image.open(io.BytesIO(base64.b64decode(encoded)))
            assert im.size == (32, 32) and im.mode == "RGB"
        again, _ = request(f"/{kind}/generate", body)
        assert json.loads(again)["images"] == payload["images"]
        raw, mime = request(f"/{kind}/generate.png?count=16&seed=123")
        assert mime == "image/png" and Image.open(io.BytesIO(raw)).size == (128, 128)
        (out / f"{kind}_api_grid.png").write_bytes(raw)
        result[kind] = {
            "status": "passed",
            "health": health,
            "seed_reproducible": True,
            "individual_image_size": [32, 32],
            "grid_size": [128, 128],
        }
    for path in ["/", "/health", "/cnn/health", "/gan/health", "/embedding?word=apple"]:
        raw, _ = request(path)
        payload = json.loads(raw)
        if "embedding" in path:
            assert payload["dimensions"] == 300 and len(payload["embedding"]) == 300
        result[path] = {"status": "passed"}
    raw, _ = request("/generate", {"start_word": "the", "length": 5})
    assert json.loads(raw)["generated_text"]
    result["/generate"] = {"status": "passed"}
    raw, _ = request("/gan/generate", {"count": 2, "seed": 123})
    gan = json.loads(raw)
    assert len(gan["images"]) == 2
    for encoded in gan["images"]:
        im = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert im.size == (28, 28) and im.mode == "L"
    result["/gan/generate"] = {"status": "passed"}
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 150, 180)).save(stream, format="PNG")
    boundary = "assignment4boundary"
    data = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
        + stream.getvalue()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    raw, _ = request("/predict", data, f"multipart/form-data; boundary={boundary}")
    prediction = json.loads(raw)
    assert 0 <= prediction["class_id"] < 10 and len(prediction["probabilities"]) == 10
    assert abs(sum(prediction["probabilities"].values()) - 1) < 1e-5
    result["/predict"] = {"status": "passed", "input": "synthetic RGB image"}
    (out / "live_api_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
