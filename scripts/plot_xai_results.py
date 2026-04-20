#!/usr/bin/env python3
"""
plot_xai_results.py — Generate publication-quality XAI figures for the Results chapter.

Produces three figures for Pillar 2:
1. Bottleneck Resolution Problem (Paired Bar Chart)
2. Regional Vulnerability Analysis (Grouped Bar Chart)
3. Coverage vs. Precision Trade-off (Scatter Plot)

Usage:
python scripts/plot_xai_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CSV_DIR = Path("/home/yogipatel/Desktop/Dissertation/results/CSVs")
FIG_DIR = Path(
    "/home/yogipatel/Desktop/Dissertation/Final report/Figures/results_figures"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "Grad-CAM": "#1a237e",
    "GBP": "#5c6bc0",
    "Guided Grad-CAM": "#ffb300",
    "LRP": "#e53935",
    "Occlusion": "#00695c",
}

REGION_ORDER = ["Whole Tumor", "Tumor Core", "Enhancing Tumor"]
REGION_SHORT = {
    "Whole Tumor": "WT",
    "Tumor Core": "TC",
    "Enhancing Tumor": "ET",
    "wt": "WT",
    "tc": "TC",
    "et": "ET",
}


def plot_bottleneck_resolution():
    """Paired bar chart comparing Grad-CAM Weighted Dice at upsampled (160³) vs native (~20³) resolution."""
    df_up = pd.read_csv(CSV_DIR / "xai_gradcam_metrics.csv")
    df_nat = pd.read_csv(CSV_DIR / "xai_gradcam_coarse_bottleneck_metrics.csv")

    df_up["PatientID"] = df_up["Patient"].str.extract(r"(\d{5})")
    df_nat["PatientID"] = df_nat["Patient"].str.extract(r"(\d{5})")

    df_up["RegionShort"] = df_up["Region"].map(REGION_SHORT).fillna(df_up["Region"])
    df_nat["RegionShort"] = df_nat["Region"].map(REGION_SHORT).fillna(df_nat["Region"])

    df_up["Weighted_Dice"] = pd.to_numeric(df_up["Weighted_Dice"], errors="coerce")
    df_nat["Weighted_Dice"] = pd.to_numeric(df_nat["Weighted_Dice"], errors="coerce")

    df_up = df_up.dropna(subset=["Weighted_Dice"])
    df_nat = df_nat.dropna(subset=["Weighted_Dice"])
    df_up = df_up[df_up["Weighted_Dice"] > 0.01]
    df_nat = df_nat[df_nat["Weighted_Dice"] > 0.01]

    merged = pd.merge(
        df_up[["PatientID", "Region", "RegionShort", "Weighted_Dice"]],
        df_nat[["PatientID", "Region", "Weighted_Dice"]],
        on=["PatientID", "Region"],
        suffixes=("_Upsampled", "_Native"),
        how="inner",
    )

    merged_long = merged.melt(
        id_vars=["PatientID", "Region", "RegionShort"],
        value_vars=["Weighted_Dice_Upsampled", "Weighted_Dice_Native"],
        var_name="Resolution",
        value_name="Weighted_Dice",
    )
    merged_long["Resolution"] = merged_long["Resolution"].map(
        {
            "Weighted_Dice_Upsampled": "Upsampled (160³)",
            "Weighted_Dice_Native": "Native (~20³)",
        }
    )

    merged_long["Label"] = merged_long["PatientID"] + "\n" + merged_long["RegionShort"]

    region_rank = {"Whole Tumor": 0, "Tumor Core": 1, "Enhancing Tumor": 2}
    merged_long["sort_key"] = (
        merged_long["Region"].map(region_rank).astype(str) + merged_long["PatientID"]
    )
    merged_long = merged_long.sort_values("sort_key")

    fig, ax = plt.subplots(figsize=(14, 6))

    bar_colors = ["#1a237e", "#5c6bc0"]

    labels = merged_long["Label"].unique()
    x = np.arange(len(labels))
    width = 0.35

    for i, (res, color) in enumerate(
        zip(["Upsampled (160³)", "Native (~20³)"], bar_colors)
    ):
        vals = (
            merged_long[merged_long["Resolution"] == res]
            .set_index("Label")
            .reindex(labels)["Weighted_Dice"]
            .values
        )
        bars = ax.bar(
            x + i * width - width / 2,
            vals,
            width,
            label=res,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="medium",
                )

    ax.set_xlabel("")
    ax.set_ylabel("Weighted Dice Score", fontsize=12, fontweight="medium")
    ax.set_title(
        "Bottleneck Resolution Analysis: Grad-CAM Weighted Dice\nUpsampled (160³) vs. Native Bottleneck (~20³)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, ha="center")
    ax.set_ylim(0, max(merged_long["Weighted_Dice"].max() * 1.2, 0.5))
    ax.legend(
        title="Evaluation Resolution",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    regions_in_data = merged_long.drop_duplicates("Label")[
        ["Label", "Region"]
    ].set_index("Label")
    prev_region = None
    for i, label in enumerate(labels):
        curr_region = regions_in_data.loc[label, "Region"]
        if prev_region is not None and curr_region != prev_region:
            ax.axvline(
                x=i - 0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8
            )
        prev_region = curr_region

    plt.tight_layout()
    outpath = FIG_DIR / "xai_bottleneck_resolution_comparison.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" ✅ Saved: {outpath.name}")


def plot_regional_vulnerability():
    """Grouped bar chart: Mean Weighted Dice per region for each XAI method."""
    sources = {
        "Grad-CAM": ("xai_gradcam_metrics.csv", "Region"),
        "GBP": ("xai_gbp_metrics.csv", "Region"),
        "Guided Grad-CAM": ("xai_guided_gradcam_metrics.csv", "Region"),
        "LRP": ("xai_lrp_metrics.csv", "Region"),
        "Occlusion": ("xai_occlusion_metrics.csv", "Region"),
    }

    rows = []
    for method, (csv_name, region_col) in sources.items():
        df = pd.read_csv(CSV_DIR / csv_name)
        df["Weighted_Dice"] = pd.to_numeric(df["Weighted_Dice"], errors="coerce")
        df = df.dropna(subset=["Weighted_Dice"])

        region_map = {
            "Whole Tumor": "Whole Tumor",
            "wt": "Whole Tumor",
            "Tumor Core": "Tumor Core",
            "tc": "Tumor Core",
            "Enhancing Tumor": "Enhancing Tumor",
            "et": "Enhancing Tumor",
        }
        df["Region_Norm"] = df[region_col].map(region_map)
        df = df.dropna(subset=["Region_Norm"])

        df = df[df["Weighted_Dice"] > 0.001]

        for region in REGION_ORDER:
            region_data = df[df["Region_Norm"] == region]["Weighted_Dice"]
            if len(region_data) > 0:
                rows.append(
                    {
                        "Method": method,
                        "Region": region,
                        "Mean_WDice": region_data.mean(),
                        "Std_WDice": region_data.std(),
                        "N": len(region_data),
                    }
                )

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(METHOD_COLORS.keys())
    n_methods = len(methods)
    n_regions = len(REGION_ORDER)

    group_width = 0.75
    bar_width = group_width / n_methods
    x = np.arange(n_regions)

    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["Method"] == method].set_index("Region")
        vals = [
            method_data.loc[r, "Mean_WDice"] if r in method_data.index else 0
            for r in REGION_ORDER
        ]
        stds = [
            method_data.loc[r, "Std_WDice"] if r in method_data.index else 0
            for r in REGION_ORDER
        ]

        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            yerr=stds,
            capsize=3,
            error_kw={"linewidth": 0.8, "alpha": 0.6},
        )

        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="medium",
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
    ax.set_ylim(0, min(plot_df["Mean_WDice"].max() * 1.5, 0.65))
    ax.legend(
        title="XAI Method",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
        title_fontsize=10,
        ncol=3,
        loc="upper right",
    )

    plt.tight_layout()
    outpath = FIG_DIR / "xai_regional_vulnerability.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" ✅ Saved: {outpath.name}")


def plot_coverage_vs_precision():
    """
    Clean publication-quality scatter plot for coverage vs precision.
    """
    sources = {
        "Grad-CAM": ("xai_gradcam_metrics.csv", "Region"),
        "GBP": ("xai_gbp_metrics.csv", "Region"),
        "Guided Grad-CAM": ("xai_guided_gradcam_metrics.csv", "Region"),
        "LRP": ("xai_lrp_metrics.csv", "Region"),
        "Occlusion": ("xai_occlusion_metrics.csv", "Region"),
    }

    scatter_df = pd.DataFrame()
    for method, (csv_name, region_col) in sources.items():
        df = pd.read_csv(CSV_DIR / csv_name)
        df["Weighted_Dice"] = pd.to_numeric(df["Weighted_Dice"], errors="coerce")
        df["Saliency_Coverage"] = pd.to_numeric(
            df["Saliency_Coverage"], errors="coerce"
        )
        df = df.dropna(subset=["Weighted_Dice", "Saliency_Coverage"])
        df = df[(df["Weighted_Dice"] > 0.001) | (df["Saliency_Coverage"] > 0.001)]
        df["Method"] = method
        scatter_df = pd.concat([scatter_df, df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(True, linestyle="-", alpha=0.15, color="gray", zorder=0)
    ax.set_axisbelow(True)

    for method in METHOD_COLORS:
        mdata = scatter_df[scatter_df["Method"] == method]
        if len(mdata) == 0:
            continue

        cov = mdata["Saliency_Coverage"].values
        dice = mdata["Weighted_Dice"].values

        ax.scatter(
            cov,
            dice,
            c=METHOD_COLORS[method],
            s=55,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=method,
        )

        cx, cy = cov.mean(), dice.mean()
        ax.plot(
            cx,
            cy,
            marker="D",
            markersize=8,
            c=METHOD_COLORS[method],
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=5,
        )

    ax.set_xlabel("Saliency Coverage", fontsize=12, fontweight="medium")
    ax.set_ylabel("Weighted Dice Score", fontsize=12, fontweight="medium")
    ax.set_title(
        "Coverage vs. Precision Trade-off Across XAI Methods",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, max(scatter_df["Weighted_Dice"].max() * 1.12, 0.45))

    leg = ax.legend(
        title="XAI Method",
        frameon=True,
        fancybox=False,
        edgecolor="gray",
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderpad=0.6,
        labelspacing=0.5,
    )

    plt.tight_layout()
    outpath = FIG_DIR / "xai_coverage_vs_precision.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f" ✅ Saved: {outpath.name}")


if __name__ == "__main__":
    print("\n📊 Generating XAI Results Figures...\n")

    print(" [1/3] Bottleneck Resolution Comparison...")
    plot_bottleneck_resolution()

    print(" [2/3] Regional Vulnerability Analysis...")
    plot_regional_vulnerability()

    print(" [3/3] Coverage vs. Precision Trade-off...")
    plot_coverage_vs_precision()

    print(f"\n✅ All figures saved to: {FIG_DIR}/")
