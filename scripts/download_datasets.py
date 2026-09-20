"""
Download and prepare external privacy policy datasets for System 1 training.

Supported datasets:
  1. APP-350       — 350 Android app privacy policies annotated with OPP-115 scheme
  2. PolicyIE      — Information extraction from privacy policies (IE annotations)
  3. OPP-115       — (already present) 115 website privacy policies
  4. PrivacyQA     — Question-answer pairs grounded in privacy policies
  5. PolicyQA      — Paragraph-level QA on privacy policies

Usage:
    python scripts/download_datasets.py [--dataset all|app350|policyie|privacyqa|policyqa]
    python scripts/download_datasets.py --list
"""

import argparse
import os
import sys
import json
import zipfile
import tarfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    "app350": {
        "name": "APP-350",
        "description": "350 Android app privacy policies with OPP-115-style annotations",
        "url": "https://github.com/citp/privacy-policy-historical/archive/refs/heads/master.zip",
        "alt_instructions": (
            "APP-350 may need manual download.\n"
            "1. Visit https://github.com/citp/privacy-policy-historical\n"
            "2. Download the repository as ZIP\n"
            "3. Extract annotation CSVs into data/raw/APP-350/annotations/\n"
            "Alternative: Search for 'APP-350 Android privacy policy dataset' on Google Scholar."
        ),
        "output_dir": "APP-350",
    },
    "policyie": {
        "name": "PolicyIE",
        "description": "Information extraction annotations from privacy policies (data types, actions, purposes)",
        "url": "https://github.com/wasiahmad/PolicyIE/archive/refs/heads/main.zip",
        "output_dir": "PolicyIE",
    },
    "privacyqa": {
        "name": "PrivacyQA",
        "description": "Question-answer pairs about privacy policies (useful for RAG training)",
        "url": "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP/archive/refs/heads/master.zip",
        "output_dir": "PrivacyQA",
    },
    "policyqa": {
        "name": "PolicyQA",
        "description": "Paragraph-level QA dataset on privacy policies",
        "url": "https://github.com/wasiahmad/PolicyQA/archive/refs/heads/main.zip",
        "output_dir": "PolicyQA",
    },
}


def download_file(url, dest_path):
    """Download a file with progress indication."""
    print(f"  Downloading from {url}")
    try:
        req = Request(url, headers={"User-Agent": "GuardianAgent/1.0"})
        with urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r  Progress: {pct}% ({downloaded}/{total} bytes)", end="", flush=True)
            print()
        return True
    except (URLError, OSError) as e:
        print(f"  Download failed: {e}")
        return False


def extract_archive(archive_path, output_dir):
    """Extract zip or tar archive."""
    if str(archive_path).endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif str(archive_path).endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(output_dir)
    else:
        print(f"  Unknown archive format: {archive_path}")
        return False
    return True


def flatten_single_subfolder(output_dir):
    """If extraction created a single subfolder, move its contents up."""
    items = list(output_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        subfolder = items[0]
        for child in subfolder.iterdir():
            shutil.move(str(child), str(output_dir / child.name))
        subfolder.rmdir()


def download_dataset(key):
    """Download and extract a single dataset."""
    info = DATASETS[key]
    output_dir = DATA_DIR / info["output_dir"]

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[{info['name']}] Already exists at {output_dir}, skipping.")
        return True

    print(f"\n[{info['name']}] {info['description']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / f"{key}_download.zip"

    success = download_file(info["url"], archive_path)
    if not success:
        if "alt_instructions" in info:
            print(f"\n  Manual download instructions:\n{info['alt_instructions']}")
        return False

    print(f"  Extracting to {output_dir}...")
    extract_archive(archive_path, output_dir)
    flatten_single_subfolder(output_dir)

    # Clean up archive
    archive_path.unlink(missing_ok=True)

    print(f"  Done. Contents: {list(output_dir.iterdir())[:5]}...")
    return True


def create_dataset_manifest(datasets_downloaded):
    """Write a manifest file listing available datasets."""
    manifest = {
        "datasets": {},
        "note": "Auto-generated by download_datasets.py. Do not edit manually.",
    }
    for key in datasets_downloaded:
        info = DATASETS[key]
        output_dir = DATA_DIR / info["output_dir"]
        manifest["datasets"][key] = {
            "name": info["name"],
            "path": str(output_dir.relative_to(BASE_DIR)),
            "exists": output_dir.exists(),
        }

    manifest_path = DATA_DIR / "datasets_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download privacy policy datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available datasets and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for key, info in DATASETS.items():
            output_dir = DATA_DIR / info["output_dir"]
            status = "present" if output_dir.exists() else "not downloaded"
            print(f"  {key:12s} — {info['name']:12s} [{status}]")
            print(f"               {info['description']}")
        return

    if args.dataset == "all":
        keys = list(DATASETS.keys())
    else:
        keys = [args.dataset]

    print(f"Data directory: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for key in keys:
        if download_dataset(key):
            downloaded.append(key)

    # Always include opp115 in manifest
    create_dataset_manifest(downloaded)

    print(f"\nDownloaded {len(downloaded)}/{len(keys)} datasets successfully.")
    if len(downloaded) < len(keys):
        print("Some datasets failed. Check output above for manual download instructions.")


if __name__ == "__main__":
    main()
