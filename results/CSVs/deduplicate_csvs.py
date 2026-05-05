"""Detect and remove duplicate Patient+Region rows in all XAI CSVs.
Keeps the FIRST occurrence (the newer run data that appears at top of file)
and removes later duplicates (the old appended data at bottom)."""
import csv
import os

CSV_DIR = '/home/yogipatel/Desktop/Dissertation/results/CSVs'

files = [
    'xai_gbp_metrics.csv',
    'xai_gradcam_metrics.csv',
    'xai_guided_gradcam_metrics.csv',
    'xai_lrp_metrics.csv',
    'xai_occlusion_metrics.csv',
    'xai_mc_dropout_metrics.csv',
]

print("=" * 70)
print("DUPLICATE DETECTION REPORT")
print("=" * 70)

for fname in files:
    path = os.path.join(CSV_DIR, fname)
    if not os.path.exists(path):
        print(f"\n{fname}: FILE NOT FOUND")
        continue
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # Detect duplicates by Patient+Region
    seen = {}
    duplicates = []
    unique_rows = []
    
    for i, row in enumerate(rows):
        key = (row['Patient'], row['Region'])
        if key in seen:
            duplicates.append((i+2, key))  # +2 for 1-indexed + header
        else:
            seen[key] = i
            unique_rows.append(row)
    
    patients = set(r['Patient'] for r in unique_rows)
    
    print(f"\n{fname}:")
    print(f"  Total rows: {len(rows)}")
    print(f"  Unique Patient+Region: {len(unique_rows)}")
    print(f"  Duplicates found: {len(duplicates)}")
    print(f"  Unique patients: {len(patients)}")
    
    if duplicates:
        dup_patients = set(d[1][0] for d in duplicates)
        print(f"  Duplicate patients: {', '.join(sorted(dup_patients))}")
        
        # Write deduplicated file
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unique_rows)
        print(f"  -> FIXED: Wrote {len(unique_rows)} unique rows back to file")
    else:
        print(f"  -> OK: No duplicates")

print("\n" + "=" * 70)
print("DONE — All CSVs deduplicated")
print("=" * 70)
