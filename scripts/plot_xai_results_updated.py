#!/usr/bin/env python3
"""
plot_xai_results_updated.py — Updated version with improved aesthetics and transparency.
Features: Jittered data points, premium color theme, and clear annotations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
    }
)

CSV_DIR = Path("/home/yogipatel/Desktop/Dissertation/results/CSVs")
FIG_DIR = Path("/home/yogipatel/Desktop/Dissertation/Final report/Figures/results_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGION_ORDER = ["Whole Tumor", "Tumor Core", "Enhancing Tumor"]

# Premium Palette (Deep Navy and Muted Teal)
BAR_COLORS = ["#1D3557", "#457B9D"]

def plot_bottleneck_resolution_enhanced():
    """Generates enhanced bottleneck comparison plots with individual data points for transparency."""
    try:
        df_up = pd.read_csv(CSV_DIR / "xai_gradcam_metrics.csv")
        df_nat = pd.read_csv(CSV_DIR / "xai_gradcam_coarse_bottleneck_metrics.csv")
    except FileNotFoundError as e:
        print(f"⚠️ Could not load data: {e}")
        return

    # Standardize region names
    region_map = {
        "wt": "Whole Tumor", "tc": "Tumor Core", "et": "Enhancing Tumor",
        "Whole Tumor": "Whole Tumor", "Tumor Core": "Tumor Core", "Enhancing Tumor": "Enhancing Tumor"
    }
    df_up["Region"] = df_up["Region"].map(region_map)
    df_nat["Region"] = df_nat["Region"].map(region_map)

    df_up["Resolution"] = "Upsampled (160³)"
    df_nat["Resolution"] = "Native (~20³)"

    df_combined = pd.concat([df_up, df_nat], ignore_index=True)
    
    # Ensure numeric
    metrics = ["Weighted_Dice", "Saliency_IoU"]
    for m in metrics:
        df_combined[m] = pd.to_numeric(df_combined[m], errors="coerce")
    
    df_combined = df_combined.dropna(subset=["Weighted_Dice", "Region", "Resolution"])

    for metric_name in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # 1. Bar Plot with Error Bars (Means + Std)
        sns.barplot(
            data=df_combined,
            x="Region",
            y=metric_name,
            hue="Resolution",
            palette=BAR_COLORS,
            order=REGION_ORDER,
            hue_order=["Upsampled (160³)", "Native (~20³)"],
            capsize=.05,
            errorbar="sd",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
            ax=ax
        )
        
        # 2. Transparency: Overlay jittered points (Stripplot)
        sns.stripplot(
            data=df_combined,
            x="Region",
            y=metric_name,
            hue="Resolution",
            palette=BAR_COLORS, # Match bar colors for points
            order=REGION_ORDER,
            hue_order=["Upsampled (160³)", "Native (~20³)"],
            dodge=True,
            alpha=0.3,
            size=4,
            jitter=0.2,
            ax=ax,
            legend=False
        )
        
        # Customizing Axes
        ylabel = metric_name.replace('_', ' ')
        ax.set_ylabel(ylabel, fontsize=13, fontweight="semibold")
        ax.set_xlabel("Tumour Sub-Region", fontsize=13, fontweight="semibold")
        
        # Annotate sample size
        n_up = df_up["Patient"].nunique()
        n_nat = df_nat["Patient"].nunique()
        title_metric = "Weighted Dice" if metric_name == "Weighted_Dice" else "Saliency IoU"
        ax.set_title(
            f"Grad-CAM Resolution Fidelity: {title_metric}\n"
            f"Comparing Upsampled (N={n_up}) vs. Native Bottleneck (N={n_nat})",
            fontsize=14, fontweight="bold", pad=20
        )

        # Better Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[0:2], labels[0:2],
            title="Evaluation Resolution",
            frameon=True, shadow=True, loc="upper right"
        )

        plt.tight_layout()
        out_name = f"xai_bottleneck_{metric_name.lower()}.svg"
        plt.savefig(FIG_DIR / out_name, format="svg", bbox_inches="tight")
        print(f" ✅ Saved updated plot: {out_name}")
        plt.close()

if __name__ == "__main__":
    print("\n🎨 Rebuilding XAI Bottleneck Comparison with Improved Aesthetics...\n")
    plot_bottleneck_resolution_enhanced()
    print(f"\n✨ Generation complete. Files saved to {FIG_DIR}")
