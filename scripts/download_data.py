"""Download the four raw PCOS datasets into data/raw/ and verify their SHA-256 hashes.

The datasets belong to their original publishers and are not redistributed in this
repository. Each one is pinned to a specific upstream commit, so the downloaded files
are byte-identical to the ones the published model was trained on.

Usage: python scripts/download_data.py [--raw-dir data/raw] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# key: (destination relative to raw dir, pinned URL, sha256)
DATASETS = {
    "kerala": (
        "kerala/PCOS_data_without_infertility.xlsx",
        "https://raw.githubusercontent.com/aryashah2k/Prediction-Of-PCOS-Using-Optimized-ML-Classifiers/"
        "66b48ae2834620e46319f3b333b9cafa9d85e5c9/Dataset/PCOS_data_without_infertility.xlsx",
        "b663ec9f491be419718e0935eb1c5b6c4923b527282230a73cfc9aca8533f742",
    ),
    "survey_in_2018": (
        "survey_in_2018/dataset1.csv",
        "https://raw.githubusercontent.com/manika1430/PCOS-detection-using-classifiation-techniques/"
        "9a673149f00538d7caff36d00c61f42cfd73903f/dataset1.csv",
        "4a6bf6c59a288c9a0fb2b80102607f5ca4c97ea8171608da34d8a0c6b3b8e9de",
    ),
    "survey_mumbai": (
        "survey_mumbai/allData.csv",
        "https://raw.githubusercontent.com/Sathwikapaturi/PCOD-Prediction/"
        "0cb2e21ac82afe18ae661978e8b82abff4d44c98/allData.csv",
        "1908ee3cd325fd7598709ed57996992ffa06fd674a6d3691fcc0bad7f1f94ed3",
    ),
    "survey_lifestyle": (
        "survey_lifestyle/pcos_raw.csv",
        "https://raw.githubusercontent.com/Caspal26/pcos-risk-prediction/"
        "a5bc0114730c474da8ba60aed771f3bdf74a58f1/data/raw/pcos_raw.csv",
        "dfda7b68734d25738555cd3425de2412113fea2736a24166b7ba64c5824f63f9",
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pcos-early-check-downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download and verify the raw PCOS datasets")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"), help="Destination folder (default: data/raw)")
    ap.add_argument("--force", action="store_true", help="Download again even if a verified copy exists")
    args = ap.parse_args(argv)
    raw_dir = Path(args.raw_dir)
    failures = 0
    for key, (rel, url, expected) in DATASETS.items():
        dest = raw_dir / rel
        if dest.is_file() and not args.force and sha256(dest.read_bytes()) == expected:
            print(f"[ok]   {key:17} already present and verified")
            continue
        try:
            data = fetch(url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[fail] {key:17} download failed: {exc}")
            failures += 1
            continue
        if sha256(data) != expected:
            print(f"[fail] {key:17} hash mismatch; the upstream file changed, so it was not saved")
            failures += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"[new]  {key:17} {len(data):>8,} bytes -> {dest}")
    if failures:
        print(f"\n{failures} dataset(s) could not be fetched. See data/raw/README.md for the original sources.")
        return 1
    print("\nAll datasets present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
