import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# File paths
base_dir = "/home/sean/diss/virtualize_llm/TraceStorm/tracestorm_results/chatbot_arena"
files = {
    "local": "results_local.json",
    "remote": "results_remote.json",
    "original": "results_original.json",
}

# Helper to load summary stats from each file
def load_stats(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    summary = data["summary"]
    return summary

# Helper to reconstruct sample data from summary stats
def reconstruct_samples(stats):
    # For TTFT, count is usually 1, so just use mean
    count = int(stats["count"])
    if count == 1:
        return [stats["mean"]]
    # For TPOT, generate synthetic data for CDF plot using mean and std
    # This is not perfect, but works for visualization if raw data is not available
    mean = stats["mean"]
    std = stats["std"] if not np.isnan(stats["std"]) else 0
    np.random.seed(0)
    samples = np.random.normal(loc=mean, scale=std, size=count)
    # Clamp to min/max
    samples = np.clip(samples, stats["min"], stats["max"])
    return samples

# Load and reconstruct data
ttft_data = {}
tpot_data = {}
for label, fname in files.items():
    path = os.path.join(base_dir, fname)
    stats = load_stats(path)
    # Convert to milliseconds
    ttft_data[label] = [x * 1000 for x in reconstruct_samples(stats["ttft"])]
    tpot_data[label] = [x * 1000 for x in reconstruct_samples(stats["tpot"])]

# Plot TTFT CDF
plt.figure(figsize=(8, 6))
for label, data in ttft_data.items():
    if len(data) > 1:
        sns.ecdfplot(data, label=label)
    else:
        plt.step([data[0], data[0]], [0, 1], where="post", label=label)
plt.title("CDF of Time to First Token (TTFT)")
plt.xlabel("TTFT (milliseconds)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "ttft_cdf_comparison.png"))
plt.close()

# Plot TPOT CDF
plt.figure(figsize=(8, 6))
for label, data in tpot_data.items():
    sns.ecdfplot(data, label=label)
plt.title("CDF of Time per Output Token (TPOT)")
plt.xlabel("TPOT (milliseconds)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "tpot_cdf_comparison.png"))
plt.close()

print("Plots saved to", base_dir)