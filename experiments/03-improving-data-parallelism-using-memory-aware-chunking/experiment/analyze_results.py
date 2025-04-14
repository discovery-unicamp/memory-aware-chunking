#!/usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    # ---------------------------------------------------------------------
    # 0) Load environment config & CSVs
    # ---------------------------------------------------------------------
    output_dir = os.getenv("OUTPUT_DIR", "./out")
    results_dir = os.path.join(output_dir, "results")
    summary_csv_path = os.path.join(results_dir, "profiles_summary.csv")
    detail_csv_path = os.path.join(results_dir, "profiles_detail.csv")

    if not os.path.exists(summary_csv_path):
        print(f"[analyze_results] No summary file at {summary_csv_path}. Exiting.")
        sys.exit(1)

    summary_df = pd.read_csv(summary_csv_path)
    has_detail = os.path.exists(detail_csv_path)
    if has_detail:
        detail_df = pd.read_csv(detail_csv_path)
    else:
        detail_df = pd.DataFrame()

    # ---------------------------------------------------------------------
    # 1) Cast columns to consistent types to avoid merge conflicts
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 2) Compute total peak memory usage by summing worker peaks
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 3) Separate out OOM vs Successful runs
    # ---------------------------------------------------------------------
    combined_df["is_oom"] = combined_df["oom_or_failed"] == True
    successful_df = combined_df[~combined_df["is_oom"]].copy()
    oom_df = combined_df[combined_df["is_oom"]].copy()

    # For OOM runs, these columns aren't meaningful:
    oom_df["execution_time_sec"] = np.nan
    oom_df["total_peak_memory_usage_bytes"] = np.nan

    # ---------------------------------------------------------------------
    # 4) Basic shape grouping
    # ---------------------------------------------------------------------
    shape_cols = ["inlines", "xlines", "samples"]
    shape_groups = combined_df[shape_cols].drop_duplicates()

    charts_dir = os.path.join(output_dir, "analysis_charts")
    os.makedirs(charts_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # ---------------------------------------------------------------------
    # 5) Plot for each shape
    # ---------------------------------------------------------------------
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

        # Skip if no data
        if shape_success.empty and shape_oom.empty:
            continue

        # ----------------------------------------------
        # A) Execution Time vs Worker Count
        # ----------------------------------------------
        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(
            data=shape_success,
            x="worker_count",
            y="execution_time_sec",
            hue="chunking_mode",
            marker="o",
            hue_order=["auto", "evenly-split", "memaware"],  # consistent ordering
        )

        # Plot OOM as a scatter
        if not shape_oom.empty:
            sns.scatterplot(
                data=shape_oom,
                x="worker_count",
                y="execution_time_sec",  # NaN
                hue="chunking_mode",
                marker="X",
                style="chunking_mode",
                palette="dark",
                ax=ax,
                s=80,
            )
            # Annotate each OOM
            if not shape_success.empty:
                ytop = shape_success["execution_time_sec"].max()
                if pd.isna(ytop) or ytop <= 0:
                    ytop = 1.0
            else:
                ytop = 1.0
            text_y = ytop * 1.1
            for idx, row in shape_oom.iterrows():
                wcount = row["worker_count"]
                cmode = row["chunking_mode"]
                ax.text(wcount, text_y, f"OOM({cmode})", color="red", ha="center")

        plt.title(f"Execution Time vs Worker Count\nShape: {shape_label}")
        plt.xlabel("Worker Count")
        plt.ylabel("Execution Time (s)")
        plt.tight_layout()
        exec_time_png = os.path.join(charts_dir, f"{shape_label}_exec_time.png")
        plt.savefig(exec_time_png, dpi=150)
        plt.close()

        # ----------------------------------------------
        # B) Total Peak Memory vs Worker Count
        # ----------------------------------------------
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
                y="total_peak_memory_usage_bytes",  # NaN
                hue="chunking_mode",
                marker="X",
                style="chunking_mode",
                palette="dark",
                ax=ax,
                s=80,
            )
            if not shape_success.empty:
                ytop = shape_success["total_peak_memory_usage_bytes"].max()
                if pd.isna(ytop) or ytop <= 0:
                    ytop = 1.0
            else:
                ytop = 1.0
            text_y = ytop * 1.1
            for idx, row in shape_oom.iterrows():
                wcount = row["worker_count"]
                cmode = row["chunking_mode"]
                ax.text(wcount, text_y, f"OOM({cmode})", color="red", ha="center")

        plt.title(f"Total Peak Memory vs Worker Count\nShape: {shape_label}")
        plt.xlabel("Worker Count")
        plt.ylabel("Total Peak Memory (bytes)")
        plt.tight_layout()
        memory_png = os.path.join(charts_dir, f"{shape_label}_total_memory.png")
        plt.savefig(memory_png, dpi=150)
        plt.close()

        # ------------------------------------
        # C) Worker-level Boxplot from detail
        # ------------------------------------
        if not detail_df.empty:
            shape_detail = detail_df[
                (detail_df["inlines"] == inlines)
                & (detail_df["xlines"] == xlines)
                & (detail_df["samples"] == samples)
            ].copy()
            # Merge with combined_df to exclude OOM runs
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

                detail_boxplot_png = os.path.join(
                    charts_dir, f"{shape_label}_worker_box.png"
                )
                plt.savefig(detail_boxplot_png, dpi=150)
                plt.close()

    # ---------------------------------------------------------------------
    # 6) Generate a Leaderboard: best execution time per shape & worker_count
    # ---------------------------------------------------------------------
    if not successful_df.empty:
        group_cols = ["inlines", "xlines", "samples", "worker_count"]
        leaders = []
        for _, grp in successful_df.groupby(group_cols):
            min_time = grp["execution_time_sec"].min()
            best_rows = grp[grp["execution_time_sec"] == min_time]
            best_row = best_rows.iloc[0]

            best_mode = best_row["chunking_mode"]
            best_time = best_row["execution_time_sec"]

            # Compare memaware time to best time
            mem_row = grp[grp["chunking_mode"] == "memaware"]
            if not mem_row.empty:
                mem_time = mem_row["execution_time_sec"].min()
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

    # [ADDED INSIGHT 1] OOM statistics by chunk mode
    # ------------------------------------------------
    # Summarize how many OOMs occurred per chunking_mode
    # (or shape+mode) to understand which chunking strategy triggers OOM more often.
    oom_stats = (
        combined_df.groupby("chunking_mode")["is_oom"]
        .sum()
        .reset_index(name="oom_count")
    )
    total_per_mode = combined_df["chunking_mode"].value_counts().reset_index()
    total_per_mode.columns = ["chunking_mode", "total_count"]

    # Merge OOM stats with total counts
    mode_stats = pd.merge(oom_stats, total_per_mode, on="chunking_mode", how="right")
    mode_stats["oom_ratio"] = mode_stats["oom_count"] / mode_stats["total_count"]
    mode_stats_csv = os.path.join(output_dir, "oom_stats_by_mode.csv")
    mode_stats.to_csv(mode_stats_csv, index=False)
    print(f"\n[analyze_results] OOM stats by chunking mode saved to {mode_stats_csv}")
    print(mode_stats)

    # [ADDED INSIGHT 2] Overall memory usage & time statistics per chunk mode
    # ------------------------------------------------------------------------
    # For quick comparison, ignoring OOM runs
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

    print(f"\n[analyze_results] Charts have been saved to: {charts_dir}")
    print("[analyze_results] Analysis complete.")


if __name__ == "__main__":
    main()
