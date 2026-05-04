#!/usr/bin/env python3
"""
merge_xai_csvs.py — Merge existing (backed-up) XAI CSVs with new results.

After running XAI scripts with --skip to avoid re-testing existing patients,
this script merges the old results with the new ones into a single CSV per
method, deduplicating by (Patient, Region).

Usage:
    python scripts/merge_xai_csvs.py

    This expects:
      - Backed-up CSVs in results/CSVs/backup_existing/
      - New CSVs in results/CSVs/
    
    Merged output overwrites results/CSVs/ with the combined data.
"""

import os
import csv
from pathlib import Path
import numpy as np

# Define which CSVs to merge and their key columns for deduplication
MERGE_TARGETS = [
    "xai_gradcam_metrics.csv",
    "xai_gbp_metrics.csv",
    "xai_guided_gradcam_metrics.csv",
    "xai_lrp_metrics.csv",
    "xai_occlusion_metrics.csv",
    "xai_mc_dropout_metrics.csv",
]

# Deduplication key columns
DEDUP_KEYS = ("Patient", "Region")


def merge_csv(old_path: Path, new_path: Path, out_path: Path):
    """Merge two CSVs, deduplicating by (Patient, Region). New data wins on conflict."""
    
    rows = {}  # keyed by (Patient, Region) tuple
    fieldnames = None
    
    # Read old (backed-up) CSV first
    if old_path.exists():
        with open(old_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                key = tuple(row.get(k, "") for k in DEDUP_KEYS)
                rows[key] = row
    
    # Read new CSV — overwrites duplicates
    if new_path.exists():
        with open(new_path, "r") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                key = tuple(row.get(k, "") for k in DEDUP_KEYS)
                rows[key] = row
    
    if not rows or fieldnames is None:
        print(f"  ⚠ No data found for {out_path.name}")
        return 0
    
    # Sort by Patient, then Region for consistent output
    sorted_rows = sorted(rows.values(), key=lambda r: (r.get("Patient", ""), r.get("Region", "")))
    
    # Write merged output
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)
    
    # Count unique patients
    unique_patients = set(r.get("Patient", "") for r in sorted_rows)
    return len(unique_patients)


def print_summary(csv_path: Path, method_name: str):
    """Print aggregate stats for a merged CSV."""
    
    if not csv_path.exists():
        return
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return
    
    # Find the Weighted Dice column (different CSVs use different names)
    wd_col = None
    for col in ["Weighted_Dice", "Weighted_Dice_LRP"]:
        if col in rows[0]:
            wd_col = col
            break
    
    if wd_col is None:
        return
    
    print(f"\n  {method_name}:")
    
    # Group by Region
    regions = sorted(set(r.get("Region", "") for r in rows))
    for region in regions:
        region_rows = [r for r in rows if r.get("Region") == region and r.get(wd_col, "N/A") != "N/A"]
        if not region_rows:
            continue
        
        values = [float(r[wd_col]) for r in region_rows]
        n = len(values)
        mean = np.mean(values)
        std = np.std(values)
        print(f"    {region:20s}  n={n:3d}  Weighted Dice = {mean:.4f} ± {std:.4f}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results" / "CSVs"
    backup_dir = results_dir / "backup_existing"
    
    if not backup_dir.exists():
        print(f"Error: Backup directory not found: {backup_dir}")
        print("Run this first: mkdir -p results/CSVs/backup_existing && cp results/CSVs/xai_*.csv results/CSVs/backup_existing/")
        return
    
    print("=" * 65)
    print("  Merging XAI CSVs (backup_existing + new results)")
    print("=" * 65)
    
    for csv_name in MERGE_TARGETS:
        old_path = backup_dir / csv_name
        new_path = results_dir / csv_name
        out_path = results_dir / csv_name  # overwrite with merged
        
        n_patients = merge_csv(old_path, new_path, out_path)
        status = f"✓ {n_patients} patients" if n_patients > 0 else "⚠ No data"
        print(f"\n  {csv_name}: {status}")
    
    # Print summary statistics
    print(f"\n{'=' * 65}")
    print("  AGGREGATE SUMMARY (Weighted Dice)")
    print(f"{'=' * 65}")
    
    method_names = {
        "xai_gradcam_metrics.csv": "Grad-CAM",
        "xai_gbp_metrics.csv": "GBP",
        "xai_guided_gradcam_metrics.csv": "Guided Grad-CAM",
        "xai_lrp_metrics.csv": "Input × Gradient (LRP Proxy)",
        "xai_occlusion_metrics.csv": "Occlusion Sensitivity",
        "xai_mc_dropout_metrics.csv": "MC Dropout",
    }
    
    for csv_name in MERGE_TARGETS:
        csv_path = results_dir / csv_name
        print_summary(csv_path, method_names.get(csv_name, csv_name))
    
    print(f"\n{'=' * 65}")
    print("  Done! Merged CSVs saved to results/CSVs/")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
