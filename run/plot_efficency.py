import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np

# Boolean filters - set these to False to exclude these model types
INCLUDE_BASE_MODELS = True  # Set to False to exclude base models
INCLUDE_DEEPSEEK_CODER = True  # Set to False to exclude DeepSeek Coder models

file_paths = [
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetReevaluation_2025-03-17_23-25-29/driver_stats_two_shot_removed.tsv",
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetFixTwoShot_2025-03-21_18-30-20/driver_stats.tsv",
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetFixTwoShot_2025-03-21_18-13-34/driver_stats.tsv",
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetFixTwoShot_2025-03-23_17-41-09/driver_stats.tsv",
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetReRunErrors_2025-03-25_09-44-37/driver_stats.tsv",
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetSizeSweep_2025-03-26_03-15-15/driver_stats.tsv",
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetSizeSweep_2025-03-26_16-43-02/driver_stats.tsv", 
    "/data1/shypula/prog_diversity/all_experiments/NewDatasetSizeSweep_2025-03-26_03-15-15/driver_stats.tsv",
]

dfs = []
for file_path in file_paths:
    df = pd.read_csv(file_path, sep="\t")
    dfs.append(df)

# Merge all dataframes
df = pd.concat(dfs)

# Extract model size from model name
df["model_size"] = df["model"].str.extract(r"(\d+\.?\d*[Bb])", expand=False)
df["model_size"] = df["model_size"].str.replace("B", "").str.replace("b", "").astype(float)

# Filter rows
df = df[df["temperature"] == 1.0]
df = df[df["top_p"] == 1.0]
df = df[df["all_coh_semantic_prop_of_all"] != "ERROR"]
df = df.drop_duplicates(subset=["model"], keep="first")
df["all_coh_semantic_prop_of_all"] = df["all_coh_semantic_prop_of_all"].astype(float)
df[["model", "model_size", "all_coh_semantic_prop_of_all"]].to_csv("model_efficiency.tsv", sep="\t", index=False)

# Define model alignment map
model_alignment_map = {
    # Llama 3.1 70B models
    "meta-llama/Llama-3.1-70B-Instruct": "dpo",
    "allenai/Llama-3.1-Tulu-3-70B": "rl",
    "allenai/Llama-3.1-Tulu-3-70B-DPO": "dpo",
    "allenai/Llama-3.1-Tulu-3-70B-SFT": "sft",
    "meta-llama/Llama-3.1-70B": "base",
    
    # Llama 3.1 8B models
    "meta-llama/Llama-3.1-8B-Instruct": "dpo",
    "allenai/Llama-3.1-Tulu-3.1-8B": "rl",
    "allenai/Llama-3.1-Tulu-3-8B": "rl",
    "allenai/Llama-3.1-Tulu-3-8B-DPO": "dpo",
    "allenai/Llama-3.1-Tulu-3-8B-SFT": "sft",
    "meta-llama/Llama-3.1-8B": "base",
    
    # Llama 2 70B models
    "meta-llama/Llama-2-70b-chat-hf": "rl",
    "allenai/tulu-2-70b": "rl",
    "allenai/tulu-2-dpo-70b": "dpo",
    "meta-llama/Llama-2-70b-hf": "base",
    
    # Llama 2 7B models
    "meta-llama/Llama-2-7b-chat-hf": "rl",
    "allenai/tulu-2-7b": "rl",
    "allenai/tulu-2-dpo-7b": "dpo",
    "meta-llama/Llama-2-7b-hf": "base",
    
    # DeepSeek Coder models
    "deepseek-ai/deepseek-coder-33b-instruct": "rl",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "rl",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": "rl",
    "deepseek-ai/deepseek-coder-1.3b-instruct": "rl",
    
    # DeepSeek R1 Distill models
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "reasoning",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "reasoning",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "reasoning",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "reasoning",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "reasoning",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "reasoning",
    
    # Qwen models
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "rl",
    "Qwen/Qwen2.5-0.5B-Instruct": "rl",
    "Qwen/Qwen2.5-1.5B-Instruct": "rl",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "rl",
    "Qwen/Qwen2.5-3B-Instruct": "rl"
}

# Create model family column
df["model_family"] = "Other"
df.loc[df["model"].str.contains("Llama-3.1"), "model_family"] = "Llama-3.1"
df.loc[df["model"].str.contains("Llama-2"), "model_family"] = "Llama-2"
df.loc[df["model"].str.contains("tulu-2"), "model_family"] = "Tulu-2"
df.loc[df["model"].str.contains("Tulu-3"), "model_family"] = "Tulu-3"
df.loc[df["model"].str.contains("Qwen2.5-Coder"), "model_family"] = "Qwen2.5-Coder"
df.loc[df["model"].str.contains("Qwen2.5") & ~df["model"].str.contains("Coder"), "model_family"] = "Qwen2.5"
df.loc[df["model"].str.contains("deepseek-coder"), "model_family"] = "DeepSeek-Coder"
df.loc[df["model"].str.contains("DeepSeek-R1"), "model_family"] = "DeepSeek-R1"

# Create clean model names
def create_clean_name(row):
    family = row["model_family"]
    size = f"{row['model_size']:.1f}B".replace(".0B", "B")
    alignment = model_alignment_map.get(row["model"], "unknown")
    
    if alignment == "base":
        align_suffix = " (Base)"
    elif alignment == "sft":
        align_suffix = " (SFT)"
    elif alignment == "dpo":
        align_suffix = " (DPO)"
    elif alignment == "rl":
        align_suffix = " (RL)"
    elif alignment == "reasoning":
        align_suffix = " (Reasoning)"
    else:
        align_suffix = f" ({alignment})"
    
    return f"{family} {size}{align_suffix}"

df["clean_name"] = df.apply(create_clean_name, axis=1)
df["alignment"] = df["model"].map(model_alignment_map)
# Fix the problematic line
df["alignment"] = df["alignment"].fillna("unknown")

# Apply filters
if not INCLUDE_BASE_MODELS:
    df = df[df["alignment"] != "base"]

if not INCLUDE_DEEPSEEK_CODER:
    df = df[~df["model_family"].str.contains("DeepSeek-Coder")]

# Define marker styles based on model family
family_marker_map = {
    "Llama-3.1": "o",     # circle
    "Llama-2": "s",       # square
    "Tulu-2": "^",        # triangle up
    "Tulu-3": "v",        # triangle down
    "Qwen2.5": "D",       # diamond
    "Qwen2.5-Coder": "p", # pentagon
    "DeepSeek-R1": "*",   # star
    "DeepSeek-Coder": "X", # x filled
    "Other": "+"          # plus
}

# Define colors based on alignment method
alignment_color_map = {
    "base": "#1f77b4",    # blue
    "sft": "#2ca02c",     # green
    "dpo": "#d62728",     # red
    "rl": "#9467bd",      # purple
    "reasoning": "#ff7f0e", # orange
    "unknown": "#7f7f7f"  # gray
}

# Calculate efficiency metric
df["efficiency"] = df["all_coh_semantic_prop_of_all"] / df["model_size"]

# Create the plot
plt.figure(figsize=(12, 8))

# Create a new dictionary to track which alignment types have been added to the legend
legend_entries = {}

# Plot each point individually to ensure proper scatter
for idx, row in df.iterrows():
    alignment = row["alignment"]
    family = row["model_family"]
    
    # Check if this alignment type is already in the legend
    if alignment not in legend_entries:
        label = alignment.upper()
        legend_entries[alignment] = True
    else:
        label = None  # No label for subsequent points of same alignment
    
    # Plot the point
    plt.scatter(
        row["model_size"], 
        row["efficiency"],
        marker=family_marker_map.get(family, "o"),
        color=alignment_color_map.get(alignment, "gray"),
        s=150,  # Larger point size
        alpha=0.9,
        label=label
    )

# Manual staggering for annotations
size_counts = df.groupby("model_size").size()
size_index = {size: 0 for size in df["model_size"].unique()}

# Add annotations with controlled staggering
for idx, row in df.iterrows():
    current_idx = size_index[row["model_size"]]
    size_index[row["model_size"]] += 1
    
    # Base offset depends on model size
    offset_x = row["model_size"] * 0.1
    
    # Compute vertical offset with more spacing for models with same size
    total_in_group = size_counts[row["model_size"]]
    
    if total_in_group > 1:
        # Apply more staggering for crowded areas
        vertical_offset = 0.2 * current_idx  # Staggering factor
        new_y = row["efficiency"] * (1 + vertical_offset)
    else:
        new_y = row["efficiency"] * 1.15  # Just a small offset for isolated points
    
    # Create the annotation
    label_text = row["clean_name"].split()[0] + " " + row["clean_name"].split()[1]
    plt.annotate(
        label_text,
        (row["model_size"], row["efficiency"]),
        xytext=(row["model_size"] + offset_x, new_y),
        fontsize=9,
        fontweight='bold',
        alpha=0.9,
        arrowprops=dict(arrowstyle='-', lw=0.5, alpha=0.6)
    )

# Add axis labels
plt.xlabel("Model Size (Billions of Parameters)", fontsize=14, weight='bold')
plt.ylabel("Coherence Semantic / Model Size", fontsize=14, weight='bold')

# EXPLICITLY set both axes to log scale
plt.xscale("log")
plt.yscale("log")

# Set custom x-ticks for model sizes
sizes = sorted(df["model_size"].unique())
plt.xticks(sizes, [f"{s:.1f}B".replace(".0B", "B") for s in sizes], fontsize=12)
plt.yticks(fontsize=12)

# Add gridlines
plt.grid(True, alpha=0.3, linestyle='--')

# Add title
title = "Model Performance Efficiency by Size and Alignment Method"
if not INCLUDE_BASE_MODELS:
    title += " (Base Models Excluded)"
if not INCLUDE_DEEPSEEK_CODER:
    title += " (DeepSeek Coder Excluded)"
plt.title(title, fontsize=16, weight='bold')

# Only use alignment in the legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc='upper right', title="Alignment Method", 
          fontsize=12, title_fontsize=14)

# Add box around the plot
plt.box(True)

# Save before any tight_layout
plt.savefig("model_efficiency_plot_loglog.png", dpi=300, bbox_inches='tight')

# Print model count
print(f"Total models plotted: {len(df)}")
print(f"Models by alignment method:")
print(df.groupby("alignment").size())