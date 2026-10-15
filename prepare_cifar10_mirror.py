"""Optional faster data preparation from University of Toronto's CIFAR-10 mirror.

pip install pyarrow==19.0.1
python scripts/prepare_cifar10_mirror.py --output-dir data/mirror
python train.py --prepared-data data/mirror --epochs 10 --seed 617

The images are lossless PNG encodings of CIFAR-10, decoded to their original
32 x 32 RGB arrays. No relabeling, recompression or split mixing is performed.
"""

import argparse
import hashlib
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


REPOSITORY = "uoft-cs/cifar10"
FILES = {
    "train": "plain_text/train-00000-of-00001.parquet",
    "test": "plain_text/test-00000-of-00001.parquet",
}


def prepare_split(split, relative_file, revision, output_dir):
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{relative_file}"
    # A cache from an older revision must not be attributed to a newer one.
    parquet_path = output_dir / f"{revision[:12]}-{Path(relative_file).name}"
    temporary_path = parquet_path.with_suffix(".part")
    start = time.perf_counter()
    if not parquet_path.exists():
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary_path.open("wb") as handle:
                    count = 0
                    next_report = 20 * 1024 * 1024
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        count += len(chunk)
                        if count >= next_report:
                            print(f"{split}: downloaded {count / 1024**2:.0f} MiB", flush=True)
                            next_report += 20 * 1024 * 1024
                temporary_path.replace(parquet_path)
                break
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                print(f"{split}: retrying download ({attempt}/3)", flush=True)
    table = pq.read_table(parquet_path, columns=["img", "label"])
    expected_count = 50_000 if split == "train" else 10_000
    if len(table) != expected_count:
        raise ValueError(f"Unexpected {split} row count: {len(table)}")
    images = np.empty((expected_count, 32, 32, 3), dtype=np.uint8)
    labels = np.asarray(table["label"].to_numpy(), dtype=np.int64)
    for index, record in enumerate(table["img"].to_pylist()):
        with Image.open(io.BytesIO(record["bytes"])) as image:
            if image.size != (32, 32):
                raise ValueError("The mirror contains a non-CIFAR-10 image size.")
            images[index] = np.asarray(image.convert("RGB"))
    np.savez(output_dir / f"{split}.npz", images=images, labels=labels)
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    print(f"{split}: prepared {expected_count} images in {time.perf_counter()-start:.1f}s", flush=True)
    return {"file": relative_file, "sha256": digest, "rows": expected_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mirror"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPOSITORY}", timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            split: executor.submit(prepare_split, split, path, revision, args.output_dir)
            for split, path in FILES.items()
        }
        splits = {split: future.result() for split, future in futures.items()}
    source = {
        "loader": "PreparedCIFAR10", "repository": REPOSITORY,
        "revision": revision, "splits": splits,
        "dataset_card": "https://huggingface.co/datasets/uoft-cs/cifar10",
        "original_homepage": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "preparation": "Lossless PNG decoded to original 32x32 RGB arrays; official train/test splits preserved",
    }
    (args.output_dir / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
