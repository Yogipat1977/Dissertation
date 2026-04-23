#!/usr/bin/env python3
"""
plot_xai_results_svg.py — Generates Graph 1 & Graph 6 as SVGs.
Fully customized with reduced thickness, custom palettes, and error-adjusted labels.
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
    }
)

# Update these paths to match your local machine
CSV_DIR = Path("/home/yogipatel/Desktop/Dissertation/results/CSVs")
FIG_DIR = Path("/home/yogipatel/Desktop/Dissertation/Final report/Figures/results_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGION_ORDER = ["Whole Tumor", "Tumor Core", "Enhancing Tumor"]

def plot_bottleneck_resolution_svg():
    """Generates Graph 1 (Dice and IoU) with Reduced Bar Thickness & Custom Labels."""
    try:
        df_up = pd.read_csv(CSV_DIR / "xai_gradcam_metrics.csv")
        df_nat = pd.read_csv(CSV_DIR / "xai_gradcam_coarse_bottleneck_metrics.csv")
    except FileNotFoundError as e:
        print(f"⚠️ Could not load Graph 1 data: {e}")
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

    metrics = ["Weighted_Dice", "Saliency_IoU"]
    for m in metrics:
        df_combined[m] = pd.to_numeric(df_combined[m], errors="coerce")

    bar_colors = ["#2b2d42", "#8d99ae"]
    resolutions = ["Upsampled (160³)", "Native (~20³)"]

    for metric_name in metrics:
        df_plot = df_combined.dropna(subset=[metric_name, 'Region', 'Resolution'])
        grouped = df_plot.groupby(['Region', 'Resolution'])[metric_name].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Keep groups close together
        group_spacing = 0.6
        x = np.arange(len(REGION_ORDER)) * group_spacing
        
        # --- REDUCED BAR THICKNESS FOR GRAPH 1 ---
        bar_width = 0.15 
        max_y_value = 0

        for i, res in enumerate(resolutions):
            res_data = grouped[grouped['Resolution'] == res].set_index('Region')
            
            means = [res_data.loc[r, 'mean'] if r in res_data.index else 0 for r in REGION_ORDER]
            stds = [res_data.loc[r, 'std'] if r in res_data.index else 0 for r in REGION_ORDER]
            
            offset = (i - 0.05) * bar_width
            pos = x + offset
            
            bars = ax.bar(
                pos,
                means,
                bar_width,
                label=res,
                color=bar_colors[i],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
                yerr=stds,
                capsize=4,
                error_kw={"linewidth": 1.2, "alpha": 0.8, "elinewidth": 1.2}
            )
            
            # Draw Labels above error bars
            for bar, mean_val, std_val in zip(bars, means, stds):
                if mean_val > 0.001:
                    std_safe = std_val if not pd.isna(std_val) else 0
                    y_pos = mean_val + std_safe + (0.015 if metric_name == "Weighted_Dice" else 0.005)
                    
                    if y_pos > max_y_value:
                        max_y_value = y_pos
                    
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        f"{mean_val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="medium"
                    )

        ax.set_xlabel("Tumor Sub-Region", fontsize=12, fontweight="medium")
        ax.set_ylabel(f"{metric_name.replace('_', ' ')} Score", fontsize=12, fontweight="medium")
        ax.set_title(
            f"Bottleneck Resolution Analysis: Grad-CAM {metric_name.replace('_', ' ')}\nUpsampled (160³) vs. Native Bottleneck (~20³)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(REGION_ORDER, fontsize=11)
        ax.set_ylim(0, max_y_value * 1.15)
        
        ax.legend(
            title="Evaluation Resolution",
            frameon=True, fancybox=True, shadow=True, fontsize=10, loc='upper right'
        )

        plt.tight_layout()
        outpath = FIG_DIR / f"graph1_bottleneck_{metric_name.lower()}.svg"
        plt.savefig(outpath, format="svg", bbox_inches="tight")
        plt.close()
        print(f" ✅ Saved: {outpath.name}")


METHOD_COLORS = {
    "Grad-CAM": "#000100",
    "GBP": "#a1a6b4",
    "Guided Grad-CAM": "#94c5cc",
    "LRP": "#b4d2e7",
    "Occlusion": "#f8f8f8",
}

def plot_regional_vulnerability_svg():
    """Generates Graph 6 Custom Palette, Thin Bars, and Error-Adjusted Labels."""
    sources = {
        "Grad-CAM": ("xai_gradcam_metrics.csv", "Region"),
        "GBP": ("xai_gbp_metrics.csv", "Region"),
        "Guided Grad-CAM": ("xai_guided_gradcam_metrics.csv", "Region"),
        "LRP": ("xai_lrp_metrics.csv", "Region"),
        "Occlusion": ("xai_occlusion_metrics.csv", "Region"),
    }

    rows = []
    for method, (csv_name, region_col) in sources.items():
        try:
            df = pd.read_csv(CSV_DIR / csv_name)
        except FileNotFoundError:
            print(f" ⚠️ Missing file: {csv_name}")
            continue
            
        df["Weighted_Dice"] = pd.to_numeric(df["Weighted_Dice"], errors="coerce")
        df = df.dropna(subset=["Weighted_Dice"])

        region_map = {
            "Whole Tumor": "Whole Tumor", "wt": "Whole Tumor",
            "Tumor Core": "Tumor Core", "tc": "Tumor Core",
            "Enhancing Tumor": "Enhancing Tumor", "et": "Enhancing Tumor",
        }
        df["Region_Norm"] = df[region_col].map(region_map)
        df = df.dropna(subset=["Region_Norm"])
        df = df[df["Weighted_Dice"] > 0.001]

        for region in REGION_ORDER:
            region_data = df[df["Region_Norm"] == region]["Weighted_Dice"]
            if len(region_data) > 0:
                rows.append({
                    "Method": method,
                    "Region": region,
                    "Mean_WDice": region_data.mean(),
                    "Std_WDice": region_data.std(),
                    "N": len(region_data),
                })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty: return

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(METHOD_COLORS.keys())
    n_methods = len(methods)
    n_regions = len(REGION_ORDER)

    group_width = 0.6  
    bar_width = group_width / n_methods
    x = np.arange(n_regions)
    
    max_y_value = 0

    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["Method"] == method].set_index("Region")
        vals = [method_data.loc[r, "Mean_WDice"] if r in method_data.index else 0 for r in REGION_ORDER]
        stds = [method_data.loc[r, "Std_WDice"] if r in method_data.index else 0 for r in REGION_ORDER]

        offset = (i - n_methods / 2 + 0.5) * bar_width
        
        bars = ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9,
            yerr=stds,
            capsize=3,
            error_kw={"linewidth": 1.0, "alpha": 0.8},
        )

        # --- ADD LABELS ABOVE ERROR BARS FOR GRAPH 6 ---
        for bar, val, std in zip(bars, vals, stds):
            if val > 0.01:
                std_safe = std if not pd.isna(std) else 0
                y_pos = val + std_safe + 0.015
                
                if y_pos > max_y_value:
                    max_y_value = y_pos
                
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="medium"
                )

    ax.set_xlabel("")
    ax.set_ylabel("Mean Weighted Dice Score", fontsize=12, fontweight="medium")
    ax.set_title(
        "Regional Vulnerability Analysis: XAI Spatial Alignment by Tumor Subregion",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, fontsize=11)
    
    # Dynamic Axis scaling so labels don't get cut off
    ax.set_ylim(0, max_y_value * 1.15)
    
    ax.legend(
        title="XAI Method",
        frameon=True, fancybox=True, shadow=True, fontsize=9,
        title_fontsize=10, ncol=3, loc="upper right",
    )

    plt.tight_layout()
    outpath = FIG_DIR / "graph6_regional_vulnerability.svg"
    plt.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close()
    print(f" ✅ Saved: {outpath.name}")


if __name__ == "__main__":
    print("\n📊 Generating SVG Results Figures...\n")

    print(" [1/2] Bottleneck Resolution Comparison (Separated, Thin, Labelled)...")
    plot_bottleneck_resolution_svg()

    print(" [2/2] Regional Vulnerability Analysis (Labelled Custom Palette)...")
    plot_regional_vulnerability_svg()

    print(f"\n✅ All configured SVG figures saved to: {FIG_DIR}/")