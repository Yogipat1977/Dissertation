import pandas as pd
from pathlib import Path

csv_dir = Path('results/CSVs')
methods = {
    "Grad-CAM": "xai_gradcam_metrics.csv",
    "GBP": "xai_gbp_metrics.csv",
    "Guided Grad-CAM": "xai_guided_gradcam_metrics.csv",
    "IxG (LRP Proxy)": "xai_lrp_metrics.csv",
    "Occlusion": "xai_occlusion_metrics.csv"
}

def get_row(method, filename):
    df = pd.read_csv(csv_dir / filename)
    region_map = {
        "wt": "Whole Tumor", "tc": "Tumor Core", "et": "Enhancing Tumor",
        "Whole Tumor": "Whole Tumor", "Tumor Core": "Tumor Core", "Enhancing Tumor": "Enhancing Tumor"
    }
    df["Region"] = df["Region"].map(region_map)
    df["Weighted_Dice"] = pd.to_numeric(df["Weighted_Dice"], errors="coerce")
    stats = df.groupby("Region")["Weighted_Dice"].agg(['mean', 'std'])
    
    # Return in order WT, TC, ET
    wt = stats.loc["Whole Tumor"]
    tc = stats.loc["Tumor Core"]
    et = stats.loc["Enhancing Tumor"]
    return f"[{method}], [{wt['mean']:.3f} ± {wt['std']:.3f}], [{tc['mean']:.3f} ± {tc['std']:.3f}], [{et['mean']:.3f} ± {et['std']:.3f}],"

for method, filename in methods.items():
    print(get_row(method, filename))
