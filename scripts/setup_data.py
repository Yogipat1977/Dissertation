#!/usr/bin/env python3
"""
setup_data.py — Automated BraTS2023 Data Acquisition Pipeline
=============================================================
Downloads the BraTS2023-GLI dataset from Synapse and creates a reproducible
patient subset for prototyping. Designed to run on any server with a single
command.

Usage:
    # Full pipeline (download + subset):
    export SYNAPSE_AUTH_TOKEN="your-token"
    python scripts/setup_data.py --project-dir /path/to/Dissertation

    # Subset only (data already downloaded):
    python scripts/setup_data.py --skip-download --project-dir .

    # Preview without changing anything:
    python scripts/setup_data.py --dry-run --skip-download --project-dir .
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYNAPSE_FOLDER_ID = "syn51514105"
FULL_DATA_SUBDIR = "BraTS2023_GLI"
TRAINING_SUBDIR = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
PROTOTYPE_SUBDIR = "Prototype_Data"
EXPECTED_MODALITIES = {"t1n", "t1c", "t2w", "t2f", "seg"}

# Files that signal this is the project root
PROJECT_MARKERS = ["GEMINI.md", "requirements.txt", ".gitignore"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    """Walk upward from `start` looking for a project marker file."""
    current = start.resolve()
    for _ in range(10):  # safety limit
        if any((current / m).exists() for m in PROJECT_MARKERS):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return start.resolve()


def resolve_project_dir(arg: str | None) -> Path:
    """Return the project root from an explicit path or auto-detection."""
    if arg:
        candidate = Path(arg).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"--project-dir path does not exist: {candidate}")
    return find_project_root(Path.cwd())


def get_synapse_token(args) -> str:
    """Resolve the Synapse auth token from CLI > env var > file, in order."""
    # 1. Explicit CLI flag
    if args.token:
        return args.token

    # 2. Environment variable
    env_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if env_token:
        return env_token

    # 3. .env file in project root
    env_file = args._project_dir / ".env"
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
          "  data/api.txt in the project directory.", file=sys.stderr)
    sys.exit(1)


def validate_patient_dir(patient_path: Path) -> list[str]:
    """Check a patient directory has the expected 5 NIfTI files. Returns
    a list of problems (empty = valid)."""
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


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def step_download(token: str, download_path: Path, dry_run: bool):
    """Download the full BraTS2023-GLI dataset from Synapse."""
    print(f"\n{'='*60}")
    print("STEP 1: Download BraTS2023-GLI from Synapse")
    print(f"{'='*60}")
    print(f"  Synapse folder : {SYNAPSE_FOLDER_ID}")
    print(f"  Download path  : {download_path}")

    if dry_run:
        print("  [DRY RUN] Skipping actual download.")
        return

    try:
        import synapseclient
        import synapseutils
    except ImportError:
        print("[ERROR] synapseclient is not installed. Run:\n"
              "  pip install synapseclient", file=sys.stderr)
        sys.exit(1)

    download_path.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.login(authToken=token)
    print("  Authenticated successfully. Starting download...")
    print("  (This may take a long time for the full dataset.)\n")

    synapseutils.syncFromSynapse(syn, SYNAPSE_FOLDER_ID, path=str(download_path))
    print("\n  Download complete.")


def step_subset(
    training_path: Path,
    prototype_path: Path,
    num_patients: int,
    seed: int,
    dry_run: bool,
):
    """Create a reproducible subset of patients for prototyping."""
    print(f"\n{'='*60}")
    print("STEP 2: Create Prototype Subset")
    print(f"{'='*60}")
    print(f"  Source    : {training_path}")
    print(f"  Dest      : {prototype_path}")
    print(f"  Patients  : {num_patients}")
    print(f"  Seed      : {seed}")

    if not training_path.exists():
        print(f"\n[ERROR] Training data directory not found:\n"
              f"  {training_path}\n"
              f"  Run without --skip-download first, or check the path.",
              file=sys.stderr)
        sys.exit(1)

    all_patients = sorted([d for d in training_path.iterdir() if d.is_dir()])
    print(f"  Found {len(all_patients)} patient directories in source.")

    if len(all_patients) < num_patients:
        print(f"\n[ERROR] Need at least {num_patients} patients, "
              f"but only found {len(all_patients)}.", file=sys.stderr)
        sys.exit(1)

    # Reproducible selection
    rng = random.Random(seed)
    selected = rng.sample(all_patients, num_patients)
    selected.sort(key=lambda p: p.name)  # sort for readability

    print(f"\n  Selected {num_patients} patients (seed={seed}):")
    for i, p in enumerate(selected, 1):
        print(f"    {i:3d}. {p.name}")

    if dry_run:
        print("\n  [DRY RUN] Skipping file copy.")
        return selected

    # Clear and recreate prototype directory
    if prototype_path.exists():
        print(f"\n  Clearing existing prototype directory...")
        shutil.rmtree(prototype_path)

    prototype_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  Copying {num_patients} patients...")
    for i, patient_folder in enumerate(selected, 1):
        dest = prototype_path / patient_folder.name
        shutil.copytree(patient_folder, dest)
        print(f"    [{i:3d}/{num_patients}] {patient_folder.name}")

    print("  Subset creation complete.")
    return selected


def step_verify(prototype_path: Path, expected_count: int):
    """Verify the integrity of the prototype dataset."""
    print(f"\n{'='*60}")
    print("STEP 3: Verification")
    print(f"{'='*60}")

    if not prototype_path.exists():
        print("  [SKIP] Prototype directory does not exist (dry run?).")
        return

    patients = sorted([d for d in prototype_path.iterdir() if d.is_dir()])
    print(f"  Patient directories found: {len(patients)}")

    if len(patients) != expected_count:
        print(f"  [WARN] Expected {expected_count}, found {len(patients)}.")

    errors = 0
    for p in patients:
        problems = validate_patient_dir(p)
        if problems:
            errors += 1
            print(f"  [FAIL] {p.name}: {'; '.join(problems)}")

    if errors == 0:
        print(f"  ✅ All {len(patients)} patients passed integrity checks.")
        print(f"     Each has 5 modalities: {', '.join(sorted(EXPECTED_MODALITIES))}")
    else:
        print(f"\n  ⚠️  {errors} patient(s) have issues.")


def save_manifest(prototype_path: Path, selected: list[Path], seed: int):
    """Save a JSON manifest of the selected patients for reproducibility."""
    manifest_path = prototype_path / "manifest.json"
    manifest = {
        "seed": seed,
        "num_patients": len(selected),
        "patients": [p.name for p in selected],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  📄 Manifest saved to: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare BraTS2023 data for prototyping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline:
  export SYNAPSE_AUTH_TOKEN="your-token"
  python scripts/setup_data.py --project-dir /path/to/Dissertation

  # Subset only (data already downloaded):
  python scripts/setup_data.py --skip-download --project-dir .

  # Dry run (preview only):
  python scripts/setup_data.py --dry-run --skip-download --project-dir .
        """,
    )

    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Path to the Dissertation project root. "
             "Auto-detected if not provided.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Synapse personal access token. "
             "Alternatively set SYNAPSE_AUTH_TOKEN env var.",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=45,
        help="Number of patients to include in the prototype subset (default: 45).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible patient selection (default: 42).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Synapse download (use if data is already present).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without downloading or copying files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    project_dir = resolve_project_dir(args.project_dir)
    args._project_dir = project_dir

    data_dir = project_dir / "data"
    download_path = data_dir / FULL_DATA_SUBDIR
    training_path = download_path / TRAINING_SUBDIR
    prototype_path = data_dir / PROTOTYPE_SUBDIR

    print(f"Project root : {project_dir}")
    print(f"Data dir     : {data_dir}")

    # Step 1: Download
    if not args.skip_download:
        token = get_synapse_token(args)
        step_download(token, download_path, args.dry_run)
    else:
        print("\n  [SKIP] Download step skipped (--skip-download).")

    # Step 2: Subset
    selected = step_subset(
        training_path, prototype_path, args.num_patients, args.seed, args.dry_run
    )

    # Step 3: Verify
    if not args.dry_run and selected:
        step_verify(prototype_path, args.num_patients)
        save_manifest(prototype_path, selected, args.seed)

    print(f"\n{'='*60}")
    print("Done! Your prototype data is ready at:")
    print(f"  {prototype_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
