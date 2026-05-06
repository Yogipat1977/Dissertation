import pandas as pd
from pathlib import Path

csv_up = Path('results/CSVs/xai_gradcam_metrics.csv')
csv_nat = Path('results/CSVs/xai_gradcam_coarse_bottleneck_metrics.csv')

def get_stats(path):
    df = pd.read_csv(path)
    # Standardize region names
    region_map = {
        "wt": "Whole Tumor", "tc": "Tumor Core", "et": "Enhancing Tumor",
        "Whole Tumor": "Whole Tumor", "Tumor Core": "Tumor Core", "Enhancing Tumor": "Enhancing Tumor"
    }
    df["Region"] = df["Region"].map(region_map)
    # Convert Pointing_Game to numeric, N/A becomes NaN
    df["Pointing_Game"] = pd.to_numeric(df["Pointing_Game"], errors="coerce")
    metrics = ["Pointing_Game", "Saliency_Coverage", "Saliency_IoU", "Weighted_Dice"]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    
    stats = df.groupby("Region")[metrics].agg(['mean', 'std']).round(4)
    return stats

print("--- Upsampled Stats ---")
print(get_stats(csv_up))
print("\n--- Native Bottleneck Stats ---")
print(get_stats(csv_nat))
