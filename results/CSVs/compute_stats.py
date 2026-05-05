import csv
import statistics

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def stats(vals):
    vals = [float(v) for v in vals if v not in ('N/A', '')]
    if not vals: return 0, 0, 0
    return len(vals), statistics.mean(vals), statistics.pstdev(vals) if len(vals)>1 else 0

# Count patients
for name, path in [
    ('GBP', 'xai_gbp_metrics.csv'),
    ('Grad-CAM', 'xai_gradcam_metrics.csv'),
    ('Guided GC', 'xai_guided_gradcam_metrics.csv'),
    ('IxG', 'xai_lrp_metrics.csv'),
    ('Occlusion', 'xai_occlusion_metrics.csv'),
    ('MC Dropout', 'xai_mc_dropout_metrics.csv'),
]:
    rows = load_csv(path)
    patients = set(r['Patient'] for r in rows)
    print(f'{name}: {len(patients)} patients')

print()

# Compute mean Weighted Dice per method per region
for name, path in [
    ('Grad-CAM', 'xai_gradcam_metrics.csv'),
    ('GBP', 'xai_gbp_metrics.csv'),
    ('Guided GC', 'xai_guided_gradcam_metrics.csv'),
    ('IxG', 'xai_lrp_metrics.csv'),
    ('Occlusion', 'xai_occlusion_metrics.csv'),
]:
    rows = load_csv(path)
    for region in ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']:
        r_abbr = region.replace('Whole Tumor','wt').replace('Tumor Core','tc').replace('Enhancing Tumor','et')
        wdice = [r['Weighted_Dice'] for r in rows if r.get('Region','') in (region, r_abbr) and r.get('Weighted_Dice','N/A') != 'N/A']
        n, mean, std = stats(wdice)
        print(f'{name:12s} | {region:16s} | n={n:2d} | W.Dice = {mean:.4f} +/- {std:.4f}')
    
    # Pointing Game
    for region in ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']:
        r_abbr = region.replace('Whole Tumor','wt').replace('Tumor Core','tc').replace('Enhancing Tumor','et')
        pg = [r['Pointing_Game'] for r in rows if r.get('Region','') in (region, r_abbr) and r.get('Pointing_Game','N/A') != 'N/A']
        n, mean, std = stats(pg)
        print(f'{name:12s} | {region:16s} | n={n:2d} | PG = {mean*100:.1f}%')
    print()

# MC Dropout stats
print('=== MC Dropout Aggregate ===')
rows = load_csv('xai_mc_dropout_metrics.csv')
for region in ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']:
    uar = [r['UAR'] for r in rows if r['Region']==region and r['UAR']!='N/A']
    br = [r['Boundary_Uncertainty_Ratio'] for r in rows if r['Region']==region and r['Boundary_Uncertainty_Ratio']!='N/A']
    corr = [r['Saliency_Unc_Correlation'] for r in rows if r['Region']==region and r['Saliency_Unc_Correlation']!='N/A']
    n1,m1,s1 = stats(uar)
    n2,m2,s2 = stats(br)
    n3,m3,s3 = stats(corr)
    print(f'{region:16s} | UAR={m1:.3f}+/-{s1:.3f} | BR={m2:.3f}+/-{s2:.3f} | Corr={m3:.4f}+/-{s3:.4f} (n={n1})')
