"""
analyze_results_memory_aware.py

Loads the aggregated dataset from "memory_aware_dataset.csv" and produces
some minimal analysis or plots.

Environment Variables:
  - OUTPUT_DIR (default: './out')
"""

import os
import sys

# For plotting
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    output_dir = os.getenv("OUTPUT_DIR", "./out")
    dataset_csv = os.path.join(output_dir, "memory_aware_dataset.csv")

    if not os.path.isfile(dataset_csv):
        print(f"Dataset not found: {dataset_csv}. Exiting.")
        return

    df = pd.read_csv(dataset_csv)
    print("Analyzing memory-aware chunking dataset...")
    print(df.head())

    # Example: count how many times each chunk_mode / worker_count was used
    pivot = (
        df.groupby(["chunk_mode", "worker_count"])["session_id"]
        .count()
        .reset_index(name="runs")
    )
    print("\nNumber of runs by (chunk_mode, worker_count):")
    print(pivot)

    # Basic barplot
    sns.set_theme()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=pivot, x="worker_count", y="runs", hue="chunk_mode")
    plt.title("Number of Runs by Worker Count & Chunk Mode")
    plt.tight_layout()

    out_png = os.path.join(output_dir, "analysis_runs_count.png")
    plt.savefig(out_png)
    print(f"Saved analysis plot to {out_png}")

    print("Analysis complete.")


if __name__ == "__main__":
    sys.exit(main())
