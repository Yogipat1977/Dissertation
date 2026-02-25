#!/usr/bin/env python3
"""
download_data.py — Download BraTS2023-GLI training data from Synapse.
=====================================================================
Downloads only the training data folder (with ground-truth labels) from
the BraTS2023-GLI challenge on Synapse.

Usage:
    # Using .env file:
    python scripts/download_data.py

    # Using explicit token:
    python scripts/download_data.py --token "your-token"

    # Custom output directory:
    python scripts/download_data.py --output /path/to/data

    # Preview only:
    python scripts/download_data.py --dry-run
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# syn51514132 = BraTS2023-GLI Training Data (with ground-truth labels)
SYNAPSE_TRAINING_ID = "syn51514132"

# Files that signal this is the project root
PROJECT_MARKERS = ["GEMINI.md", "requirements.txt", ".gitignore"]
EXPECTED_MODALITIES = {"t1n", "t1c", "t2w", "t2f", "seg"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    """Walk upward from `start` looking for a project marker file."""
    current = start.resolve()
    for _ in range(10):
        if any((current / m).exists() for m in PROJECT_MARKERS):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return start.resolve()


def get_synapse_token(args, project_dir: Path) -> str:
    """Resolve the Synapse auth token from CLI > env var > .env file."""
    # 1. Explicit CLI flag
    if args.token:
        return args.token

    # 2. Environment variable
    env_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if env_token:
        return env_token

    # 3. .env file in project root
    env_file = project_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip() == "SYNAPSE_AUTH_TOKEN":
                val = val.strip().strip("'\"")
                if val:
                    return val

    print("[ERROR] No Synapse auth token found. Provide one via:\n"
          "  --token <TOKEN>, or\n"
          "  SYNAPSE_AUTH_TOKEN env var, or\n"
          "  .env file in the project root.", file=sys.stderr)
    sys.exit(1)


def validate_patient_dir(patient_path: Path) -> list[str]:
    """Check a patient directory has the expected 5 NIfTI files."""
    problems = []
    nii_files = list(patient_path.glob("*.nii.gz"))
    if len(nii_files) != 5:
        problems.append(f"Expected 5 .nii.gz files, found {len(nii_files)}")
        return problems

    name = patient_path.name
    for mod in EXPECTED_MODALITIES:
        expected = patient_path / f"{name}-{mod}.nii.gz"
        if not expected.exists():
            problems.append(f"Missing modality: {mod}")
    return problems


def extract_zips(download_path: Path, keep_zip: bool = False):
    """Find and extract any .zip files in the download directory.

    The BraTS zip contains a wrapper directory (e.g.
    ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/) with patient
    folders inside. After extraction we flatten so patients sit
    directly in download_path.
    """
    zip_files = list(download_path.rglob("*.zip"))
    if not zip_files:
        print("  No zip files found — data may already be extracted.")
        return

    for zf in zip_files:
        print(f"  Extracting: {zf.name} ...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(download_path)

        if not keep_zip:
            zf.unlink()
            print(f"  Removed zip: {zf.name}")

    # Flatten: if extraction created a single wrapper directory,
    # move its contents up one level
    children = [d for d in download_path.iterdir() if d.is_dir()]
    if len(children) == 1 and not children[0].name.startswith("BraTS-GLI-"):
        wrapper = children[0]
        print(f"  Flattening wrapper directory: {wrapper.name}/")
        import shutil
        for item in wrapper.iterdir():
            dest = download_path / item.name
            shutil.move(str(item), str(dest))
        wrapper.rmdir()

    print(f"  Extracted to: {download_path}")


def verify_download(data_path: Path):
    """Verify downloaded data integrity."""
    if not data_path.exists():
        print("  [SKIP] Directory does not exist.")
        return

    patients = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"\n  Verifying {len(patients)} patient directories...")

    errors = 0
    for p in patients:
        problems = validate_patient_dir(p)
        if problems:
            errors += 1
            print(f"  [FAIL] {p.name}: {'; '.join(problems)}")

    if errors == 0:
        print(f"  ✅ All {len(patients)} patients passed integrity checks.")
    else:
        print(f"\n  ⚠️  {errors} patient(s) have issues.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download BraTS2023-GLI training data from Synapse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py
  python scripts/download_data.py --token "your-token"
  python scripts/download_data.py --output /mnt/storage/brats
  python scripts/download_data.py --keep-zip
  python scripts/download_data.py --dry-run
        """,
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="Synapse personal access token. "
             "Alternatively set SYNAPSE_AUTH_TOKEN env var or use .env file.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Download directory (default: data/BraTS2023-Training/ in project root).",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the zip file after extraction (default: remove it).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview actions without downloading.",
    )
    args = parser.parse_args()

    # Resolve paths
    project_dir = find_project_root(Path.cwd())
    if args.output:
        download_path = Path(args.output).expanduser().resolve()
    else:
        download_path = project_dir / "data" / "BraTS2023-Training"

    print(f"Project root   : {project_dir}")
    print(f"Download path  : {download_path}")
    print(f"Synapse folder : {SYNAPSE_TRAINING_ID}")

    if args.dry_run:
        print("\n[DRY RUN] Would download and extract training data to the path above.")
        print("Exiting without changes.")
        return

    # Check if data already exists
    existing_patients = list(download_path.glob("BraTS-GLI-*")) if download_path.exists() else []
    existing_zips = list(download_path.rglob("*.zip")) if download_path.exists() else []

    if existing_patients:
        print(f"\n[SKIP] Found {len(existing_patients)} patient directories already in:")
        print(f"  {download_path}")
        print("  Data appears to be downloaded and extracted. Nothing to do.")
        print("  (Delete the directory and re-run to force a fresh download.)\n")
        return

    if existing_zips:
        print(f"\n[SKIP] Found zip file, skipping download. Extracting...")
        extract_zips(download_path, keep_zip=args.keep_zip)
        verify_download(download_path)
        print(f"\n{'='*60}")
        print(f"Training data is ready at: {download_path}")
        print(f"{'='*60}\n")
        return

    # Get token and download
    token = get_synapse_token(args, project_dir)

    try:
        import synapseclient
        import synapseutils
    except ImportError:
        print("[ERROR] synapseclient is not installed. Run:\n"
              "  pip install synapseclient", file=sys.stderr)
        sys.exit(1)

    download_path.mkdir(parents=True, exist_ok=True)

    print("\nAuthenticating with Synapse...")
    syn = synapseclient.login(authToken=token)
    print("Authenticated. Starting download...")
    print("(This will take a while for ~1,250 patients.)\n")

    synapseutils.syncFromSynapse(syn, SYNAPSE_TRAINING_ID, path=str(download_path))
    print("\nDownload complete.")

    # Extract zip files
    print("\nExtracting downloaded archives...")
    extract_zips(download_path, keep_zip=args.keep_zip)

    # Verify
    verify_download(download_path)

    print(f"\n{'='*60}")
    print(f"Training data is ready at: {download_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

