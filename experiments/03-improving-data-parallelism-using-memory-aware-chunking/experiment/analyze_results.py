"""
Analyzes execution time and memory usage from summary/detail CSVs,
generates visualizations (execution time / memory usage vs. worker count),
and computes a leaderboard of best-performing chunking modes. Also includes
additional statistics such as OOM frequency and overall time/memory averages.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    # Load environment and paths
    output_dir = os.getenv("OUTPUT_DIR", "./out")
    results_dir = os.path.join(output_dir, "results")
    summary_csv = os.path.join(results_dir, "profiles_summary.csv")
    detail_csv = os.path.join(results_dir, "profiles_detail.csv")

    if not os.path.exists(summary_csv):
        print(f"[analyze_results] No summary file at {summary_csv}. Exiting.")
        sys.exit(1)

    summary_df = pd.read_csv(summary_csv)
    detail_df = (
        pd.read_csv(detail_csv) if os.path.exists(detail_csv) else pd.DataFrame()
    )

    # Convert columns to numeric/string as needed
    numeric_cols = [
        "inlines",
        "xlines",
        "samples",
        "worker_count",
        "timestamp",
        "execution_time_sec",
        "peak_memory_usage_bytes",
        "avg_memory_usage_bytes",
    ]
    string_cols = ["chunking_mode", "session_id", "random_id"]

    for col in summary_df.columns:
        if col in numeric_cols:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
        elif col in string_cols:
            summary_df[col] = summary_df[col].astype(str)

    if not detail_df.empty:
        for col in detail_df.columns:
            if col in numeric_cols:
                detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce")
            elif col in string_cols:
                detail_df[col] = detail_df[col].astype(str)

    # Compute total peak memory usage by summing peak usage across workers
    if not detail_df.empty:
        group_cols = [
            "inlines",
            "xlines",
            "samples",
            "chunking_mode",
            "worker_count",
            "timestamp",
            "session_id",
            "random_id",
        ]
        detail_agg = (
            detail_df.groupby(group_cols)["peak_memory_usage_bytes"]
            .sum()
            .reset_index(name="total_peak_memory_usage_bytes")
        )
        combined_df = pd.merge(summary_df, detail_agg, on=group_cols, how="left")
    else:
        combined_df = summary_df.copy()
        combined_df["total_peak_memory_usage_bytes"] = np.nan

    # Separate OOM runs from successful runs
    combined_df["is_oom"] = combined_df["oom_or_failed"] == True
    successful_df = combined_df[~combined_df["is_oom"]].copy()
    oom_df = combined_df[combined_df["is_oom"]].copy()

    # For OOM runs, clear out time/memory columns
    oom_df["execution_time_sec"] = np.nan
    oom_df["total_peak_memory_usage_bytes"] = np.nan

    # Identify shapes and create directory for charts
    shape_cols = ["inlines", "xlines", "samples"]
    shape_groups = combined_df[shape_cols].drop_duplicates()
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Generate charts per shape
    for _, shape_row in shape_groups.iterrows():
        inlines = shape_row["inlines"]
        xlines = shape_row["xlines"]
        samples = shape_row["samples"]
        shape_label = f"{int(inlines)}x{int(xlines)}x{int(samples)}"

        shape_success = successful_df[
            (successful_df["inlines"] == inlines)
            & (successful_df["xlines"] == xlines)
            & (successful_df["samples"] == samples)
        ]
        shape_oom = oom_df[
            (oom_df["inlines"] == inlines)
            & (oom_df["xlines"] == xlines)
            & (oom_df["samples"] == samples)
        ]

        if shape_success.empty and shape_oom.empty:
            continue

        # A) Execution Time vs Worker Count
        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(
            data=shape_success,
            x="worker_count",
            y="execution_time_sec",
            hue="chunking_mode",
            marker="o",
            hue_order=["auto", "evenly-split", "memaware"],
        )
        if not shape_oom.empty:
            sns.scatterplot(
                data=shape_oom,
                x="worker_count",
                y="execution_time_sec",
                hue="chunking_mode",
                marker="X",
                style="chunking_mode",
                palette="dark",
                ax=ax,
                s=80,
            )
            ytop = (
                shape_success["execution_time_sec"].max()
                if not shape_success.empty
                else 1.0
            )
            if pd.isna(ytop) or ytop <= 0:
                ytop = 1.0
            text_y = ytop * 1.1
            for _, row in shape_oom.iterrows():
                wcount = row["worker_count"]
                cmode = row["chunking_mode"]
                ax.text(wcount, text_y, f"OOM({cmode})", color="red", ha="center")

        plt.title(f"Execution Time vs Worker Count\nShape: {shape_label}")
        plt.xlabel("Worker Count")
        plt.ylabel("Execution Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{shape_label}_exec_time.png"), dpi=150)
        plt.close()

        # B) Total Peak Memory vs Worker Count
        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(
            data=shape_success,
            x="worker_count",
            y="total_peak_memory_usage_bytes",
            hue="chunking_mode",
            marker="o",
            hue_order=["auto", "evenly-split", "memaware"],
        )
        if not shape_oom.empty:
            sns.scatterplot(
                data=shape_oom,
                x="worker_count",
                y="total_peak_memory_usage_bytes",
                hue="chunking_mode",
                marker="X",
                style="chunking_mode",
                palette="dark",
                ax=ax,
                s=80,
            )
            ytop = (
                shape_success["total_peak_memory_usage_bytes"].max()
                if not shape_success.empty
                else 1.0
            )
            if pd.isna(ytop) or ytop <= 0:
                ytop = 1.0
            text_y = ytop * 1.1
            for _, row in shape_oom.iterrows():
                wcount = row["worker_count"]
                cmode = row["chunking_mode"]
                ax.text(wcount, text_y, f"OOM({cmode})", color="red", ha="center")

        plt.title(f"Total Peak Memory vs Worker Count\nShape: {shape_label}")
        plt.xlabel("Worker Count")
        plt.ylabel("Total Peak Memory (bytes)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(charts_dir, f"{shape_label}_total_memory.png"), dpi=150
        )
        plt.close()

        # C) Worker-level Boxplot from detail
        if not detail_df.empty:
            shape_detail = detail_df[
                (detail_df["inlines"] == inlines)
                & (detail_df["xlines"] == xlines)
                & (detail_df["samples"] == samples)
            ].copy()
            shape_detail = pd.merge(
                shape_detail,
                combined_df[["session_id", "random_id", "is_oom"]],
                on=["session_id", "random_id"],
                how="left",
            )
            shape_detail = shape_detail[~shape_detail["is_oom"]].copy()

            if not shape_detail.empty:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    data=shape_detail,
                    x="worker_count",
                    y="peak_memory_usage_bytes",
                    hue="chunking_mode",
                    order=sorted(shape_detail["worker_count"].unique()),
                    hue_order=["auto", "evenly-split", "memaware"],
                )
                plt.title(f"Worker-level Peak Memory Usage\nShape: {shape_label}")
                plt.xlabel("Worker Count")
                plt.ylabel("Peak Memory Usage (bytes)")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(charts_dir, f"{shape_label}_worker_box.png"), dpi=150
                )
                plt.close()

    # Generate a Leaderboard per shape & worker_count
    if not successful_df.empty:
        group_cols = ["inlines", "xlines", "samples", "worker_count"]
        leaders = []
        for _, grp in successful_df.groupby(group_cols):
            min_time = grp["execution_time_sec"].min()
            best_rows = grp[grp["execution_time_sec"] == min_time]
            best_row = best_rows.iloc[0]

            best_mode = best_row["chunking_mode"]
            best_time = best_row["execution_time_sec"]

            memaware_grp = grp[grp["chunking_mode"] == "memaware"]
            if not memaware_grp.empty:
                mem_time = memaware_grp["execution_time_sec"].min()
                diff_vs_best = (mem_time - best_time) / best_time * 100.0
                mem_diff_str = f"{diff_vs_best:.1f}%" if diff_vs_best != 0 else "tie"
            else:
                mem_time = None
                mem_diff_str = "not_run"

            leaders.append(
                {
                    "inlines": grp["inlines"].iloc[0],
                    "xlines": grp["xlines"].iloc[0],
                    "samples": grp["samples"].iloc[0],
                    "worker_count": grp["worker_count"].iloc[0],
                    "best_chunking_mode": best_mode,
                    "best_exec_time_sec": best_time,
                    "memaware_time_sec": mem_time,
                    "memaware_diff_percent": mem_diff_str,
                }
            )

        leader_df = pd.DataFrame(leaders)
        leader_csv = os.path.join(output_dir, "leaderboard.csv")
        leader_df.to_csv(leader_csv, index=False)
        print(f"[analyze_results] Leaderboard saved to {leader_csv}")
        print(leader_df)
    else:
        print("[analyze_results] No successful runs found. No leaderboard generated.")

    # Additional statistics: OOM counts and summary stats
    _compute_and_save_oom_stats(combined_df, output_dir)
    _compute_and_save_mode_summary(successful_df, output_dir)

    print(f"\n[analyze_results] Charts have been saved to: {charts_dir}")
    print("[analyze_results] Analysis complete.")


def _compute_and_save_oom_stats(combined_df, output_dir):
    """Summarizes how many OOMs occurred per chunking mode and saves to CSV."""
    oom_stats = (
        combined_df.groupby("chunking_mode")["is_oom"]
        .sum()
        .reset_index(name="oom_count")
    )
    total_counts = combined_df["chunking_mode"].value_counts().reset_index()
    total_counts.columns = ["chunking_mode", "total_count"]

    stats_df = pd.merge(oom_stats, total_counts, on="chunking_mode", how="right")
    stats_df["oom_ratio"] = stats_df["oom_count"] / stats_df["total_count"]
    oom_stats_csv = os.path.join(output_dir, "oom_stats_by_mode.csv")
    stats_df.to_csv(oom_stats_csv, index=False)
    print(f"\n[analyze_results] OOM stats by chunking mode saved to {oom_stats_csv}")
    print(stats_df)


def _compute_and_save_mode_summary(successful_df, output_dir):
    """Computes average/median time & memory usage, plus run counts, per chunk mode."""
    if successful_df.empty:
        return
    mode_summary = (
        successful_df.groupby("chunking_mode")
        .agg(
            avg_exec_time_sec=("execution_time_sec", "mean"),
            median_exec_time_sec=("execution_time_sec", "median"),
            avg_peak_memory_bytes=("total_peak_memory_usage_bytes", "mean"),
            median_peak_memory_bytes=("total_peak_memory_usage_bytes", "median"),
            count_runs=("chunking_mode", "count"),
        )
        .reset_index()
    )
    mode_summary_csv = os.path.join(output_dir, "mode_summary_stats.csv")
    mode_summary.to_csv(mode_summary_csv, index=False)
    print(
        f"\n[analyze_results] Summary stats (avg/median time & memory) saved to {mode_summary_csv}"
    )
    print(mode_summary)


if __name__ == "__main__":
    main()
