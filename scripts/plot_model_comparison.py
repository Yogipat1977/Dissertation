import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration & Aesthetics ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({'font.family': 'serif'})

# Approximated custom color palette from uploaded image
# Order: Dark Blue, Light Blue, Yellow, Red, Dark Green
CUSTOM_PALETTE = ['#1a237e', '#5c6bc0', '#ffb300', '#e53935', '#00695c']
sns.set_palette(CUSTOM_PALETTE)

CSV_DIR = Path("/home/yogipatel/Desktop/Dissertation/results/CSVs")
FIG_DIR = Path("/home/yogipatel/Desktop/Dissertation/results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Loading ---
models = {
    "Baseline version 0.1": "prototype_matrics_version-1_full.csv",
    "Baseline version 0.2": "prototype_matrics_version-2_full.csv",
    "Baseline version 0.3": "SegResNet_f32_250_patients_full.csv",
    "Final version 1.0": "SegResNet_f32_Full_Run.csv"
}

dataframes = []
for model_name, csv_file in models.items():
    file_path = CSV_DIR / csv_file
    if file_path.exists():
        df = pd.read_csv(file_path)
        if 'Metric' not in df.columns:
            df['Metric'] = 'dice' 
        df['Model'] = model_name
        dataframes.append(df)
    else:
        print(f"Warning: Missing file {file_path}")

full_df = pd.concat(dataframes, ignore_index=True)

test_df = full_df[['Metric', 'Region', 'Test', 'Model']].copy()
test_df.rename(columns={'Test': 'Score'}, inplace=True)
test_df['Metric'] = test_df['Metric'].str.upper()

model_order = ["Baseline version 0.1", "Baseline version 0.2", "Baseline version 0.3", "Final version 1.0"]

# --- Pre-processing for single-axis plotting ---
# To plot DICE (0-1) and HD95 (millimetres, often 10-40) on the same Y-axis,
# we multiply 0-1 metrics by 100 to convert them to percentages.
percentage_metrics = ['DICE', 'IOU', 'SENSITIVITY', 'SPECIFICITY'] 
test_df.loc[test_df['Metric'].isin(percentage_metrics), 'Score'] *= 100

# --- Plot 1: Separate Bar Charts for Version 0.1 and 1.0 ---
models_to_plot = ["Baseline version 0.1", "Final version 1.0"]

for model_name in models_to_plot:
    model_df = test_df[test_df['Model'] == model_name]
    
    # We only plot if the dataframe is not empty (just in case the CSV is missing)
    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=model_df,
            x="Region",
            y="Score",
            hue="Metric",
            order=["Whole Tumor", "Tumor Core", "Enhancing"],
            palette="crest", # Restored the 'crest' palette from the reference image
            edgecolor="black",
            alpha=0.9
        )
        
        plt.title(f"Performance Metrics by Region - {model_name}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Anatomical Region", fontsize=12)
        plt.ylabel("Score (% or Distance in mm)", fontsize=12)
        
        # Move legend outside the plot so it doesn't overlap bars
        plt.legend(title="Metrics", bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
            
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").replace(".", "")
        plt.savefig(FIG_DIR / f"bar_metrics_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

# --- Plot 2: Separated Heatmaps (To resolve scale discrepancy) ---
# Generating separate heatmaps prevents HD95 from squashing DICE/IOU colors.
metrics = test_df['Metric'].unique()

for metric in metrics:
    plt.figure(figsize=(8, 6))
    
    metric_df = test_df[test_df['Metric'] == metric]
    pivot_df = metric_df.pivot(index='Region', columns='Model', values='Score')
    pivot_df = pivot_df[[m for m in model_order if m in pivot_df.columns]]
    
    # Use inverted colormap for HD95 (lower is better, darker = better)
    cmap = "YlGnBu_r" if metric == "HD95" else "YlGnBu"
    
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        linewidths=1,
        linecolor="lightgray"
    )
    
    plt.title(f"{metric} Heatmap by Region and Model", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"heatmap_{metric.lower()}.png", dpi=300)
    plt.close()

print("Execution complete. Visualizations rendered to target directory.")