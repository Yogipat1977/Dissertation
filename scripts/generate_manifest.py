#!/usr/bin/env python3
"""
generate_manifest.py — Save the train/val/test patient split to manifest.json.

Reproduces the exact split used during training (same seed + sorting logic)
and saves patient IDs to a JSON file for reproducibility and documentation.

Usage:
    python scripts/generate_manifest.py --config configs/full_training_segresnet.yaml
    python scripts/generate_manifest.py --config configs/full_training_segresnet.yaml --output models/<run>/manifest.json
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test split manifest.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--output", type=str, default=None, help="Output path for manifest.json (default: results/manifest.json)")
    args = parser.parse_args()

    # Load config (reuse project logic)
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve data dir relative to project root
    project_root = Path(args.config).resolve().parent.parent
    data_dir = project_root / cfg["data"]["data_dir"]

    # Reproduce the exact split
    patient_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    seed = cfg["project"].get("seed", 42)
    random.seed(seed)
    random.shuffle(patient_dirs)

    train_split = cfg["data"]["train_split"]
    val_split = cfg["data"]["val_split"]

    train = patient_dirs[:train_split]
    val = patient_dirs[train_split : train_split + val_split]
    test = patient_dirs[train_split + val_split:]

    manifest = {
        "seed": seed,
        "total_patients": len(patient_dirs),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "train": train,
        "val": val,
        "test": test,
    }

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = project_root / "results" / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Train : {len(train)} patients")
    print(f"Val   : {len(val)} patients")
    print(f"Test  : {len(test)} patients")
    print(f"\nManifest saved to: {out_path}")


if __name__ == "__main__":
    main()
