#!/usr/bin/env python3
"""
extract_prototype.py — Create a reproducible prototype subset.
==============================================================
Extracts a small subset of patients from the full training dataset for
local prototyping and testing. Uses a fixed random seed for reproducibility.

Usage:
    # Default: 45 patients, seed=42
    python scripts/extract_prototype.py

    # Custom count and seed:
    python scripts/extract_prototype.py --num-patients 20 --seed 123

    # Custom source/destination:
    python scripts/extract_prototype.py --source /path/to/training --dest /path/to/prototype

    # Preview selection without copying:
    python scripts/extract_prototype.py --dry-run
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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


def validate_patient_dir(patient_path: Path) -> list[str]:
    """Check a patient directory has the expected 5 NIfTI files."""
    problems = []
    nii_files = list(patient_path.glob("*.nii.gz"))
    if len(nii_files) != 5:
        problems.append(f"Expected 5 .nii.gz files, found {len(nii_files)}")
        return problems

    name = patient_path.name
    for mod in EXPECTED_MODALITIES:
        if not (patient_path / f"{name}-{mod}.nii.gz").exists():
            problems.append(f"Missing modality: {mod}")
    return problems


def find_training_dir(data_dir: Path) -> Path:
    """Auto-detect the training data directory within data/."""
    # Try the new default location first
    candidates = [
        data_dir / "BraTS2023-Training",
        data_dir / "BraTS2023_GLI" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # return first as default (will fail with clear error)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract a prototype subset from BraTS2023 training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/extract_prototype.py
  python scripts/extract_prototype.py --num-patients 20 --seed 123
  python scripts/extract_prototype.py --dry-run
        """,
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to full training data directory. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--dest", type=str, default=None,
        help="Path to prototype output directory (default: data/Prototype_Data/).",
    )
    parser.add_argument(
        "--num-patients", type=int, default=45,
        help="Number of patients to include (default: 45).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible selection (default: 42).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview selection without copying files.",
    )
    args = parser.parse_args()

    # Resolve paths
    project_dir = find_project_root(Path.cwd())
    data_dir = project_dir / "data"

    source_path = Path(args.source).resolve() if args.source else find_training_dir(data_dir)
    dest_path = Path(args.dest).resolve() if args.dest else data_dir / "Prototype_Data"

    print(f"Source      : {source_path}")
    print(f"Destination : {dest_path}")
    print(f"Patients    : {args.num_patients}")
    print(f"Seed        : {args.seed}")

    # Validate source
    if not source_path.exists():
        print(f"\n[ERROR] Source directory not found:\n"
              f"  {source_path}\n"
              f"  Download the data first with: python scripts/download_data.py",
              file=sys.stderr)
        sys.exit(1)

    all_patients = sorted([d for d in source_path.iterdir() if d.is_dir()])
    print(f"\nFound {len(all_patients)} patient directories in source.")

    if len(all_patients) < args.num_patients:
        print(f"\n[ERROR] Need at least {args.num_patients} patients, "
              f"but only found {len(all_patients)}.", file=sys.stderr)
        sys.exit(1)

    # Reproducible selection
    rng = random.Random(args.seed)
    selected = rng.sample(all_patients, args.num_patients)
    selected.sort(key=lambda p: p.name)

    print(f"\nSelected {args.num_patients} patients (seed={args.seed}):")
    for i, p in enumerate(selected, 1):
        print(f"  {i:3d}. {p.name}")

    if args.dry_run:
        print("\n[DRY RUN] No files copied.")
        return

    # Clear and recreate destination
    if dest_path.exists():
        print(f"\nClearing existing prototype directory...")
        shutil.rmtree(dest_path)

    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {args.num_patients} patients...")
    for i, patient_folder in enumerate(selected, 1):
        dest = dest_path / patient_folder.name
        shutil.copytree(patient_folder, dest)
        print(f"  [{i:3d}/{args.num_patients}] {patient_folder.name}")

    # Verify
    print(f"\nVerifying...")
    errors = 0
    for p in sorted(d for d in dest_path.iterdir() if d.is_dir()):
        problems = validate_patient_dir(p)
        if problems:
            errors += 1
            print(f"  [FAIL] {p.name}: {'; '.join(problems)}")

    if errors == 0:
        print(f"✅ All {args.num_patients} patients passed integrity checks.")
    else:
        print(f"⚠️  {errors} patient(s) have issues.")

    # Save manifest
    manifest = {
        "seed": args.seed,
        "num_patients": args.num_patients,
        "source": str(source_path),
        "patients": [p.name for p in selected],
    }
    manifest_path = dest_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"📄 Manifest saved to: {manifest_path}")

    print(f"\n{'='*60}")
    print(f"Prototype data ready at: {dest_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
