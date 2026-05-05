"""Compute MC Dropout calibration correlations and write results to file."""
import csv
import statistics

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# Load MC Dropout metrics
mc_rows = load_csv('xai_mc_dropout_metrics.csv')
# Load per-patient segmentation metrics
seg_rows = load_csv('per_patents_SegResNet_metrics.csv')

# Normalize region names in seg_rows: "Enhancing" -> "Enhancing Tumor"
for r in seg_rows:
    if r['Region'] == 'Enhancing':
        r['Region'] = 'Enhancing Tumor'

output_lines = []

for region in ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']:
    # Get MC patients for this region
    mc_data = {}
    for r in mc_rows:
        if r['Region'] == region and r['UAR'] != 'N/A':
            mc_data[r['Patient']] = float(r['UAR'])
    
    # Get Seg dice for same patients
    seg_data = {}
    for r in seg_rows:
        if r['Region'] == region and r['Patient'] in mc_data:
            dice_val = r.get('Dice', '')
            if dice_val and dice_val != 'N/A':
                seg_data[r['Patient']] = float(dice_val)
    
    # Find common patients
    common = sorted(set(mc_data.keys()) & set(seg_data.keys()))
    
    if len(common) < 3:
        output_lines.append(f"{region}: Only {len(common)} common patients, skipping correlation")
        continue
    
    uars = [mc_data[p] for p in common]
    errors = [1.0 - seg_data[p] for p in common]
    
    # Pearson correlation manually
    n = len(common)
    mean_u = sum(uars) / n
    mean_e = sum(errors) / n
    
    cov = sum((u - mean_u) * (e - mean_e) for u, e in zip(uars, errors)) / n
    std_u = (sum((u - mean_u)**2 for u in uars) / n) ** 0.5
    std_e = (sum((e - mean_e)**2 for e in errors) / n) ** 0.5
    
    if std_u > 0 and std_e > 0:
        r_val = cov / (std_u * std_e)
    else:
        r_val = 0.0
    
    output_lines.append(f"{region:16s}: n={n:2d}, Pearson r(UAR, 1-Dice) = {r_val:.4f}")

# Write output
with open('calibration_results.txt', 'w') as f:
    f.write("=== MC Dropout Calibration: UAR vs Segmentation Error ===\n")
    for line in output_lines:
        f.write(line + "\n")
    f.write("\nDone.\n")

print("Results written to calibration_results.txt")
