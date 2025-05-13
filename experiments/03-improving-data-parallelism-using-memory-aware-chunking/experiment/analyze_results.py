import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out")
RESULTS_DIR = os.getenv("RESULTS_DIR", f"{OUTPUT_DIR}/results")
CHARTS_DIR = os.getenv("CHARTS_DIR", f"{OUTPUT_DIR}/charts")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze summary/detail CSVs and output results."
    )
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR, help="Directory with summary/detail CSVs"
    )
    parser.add_argument(
        "--charts-dir", default=CHARTS_DIR, help="Output directory for charts"
    )
    return parser.parse_args()


def load_data(results_dir):
    summary = pd.read_csv(os.path.join(results_dir, "profiles_summary.csv"))
    detail_path = os.path.join(results_dir, "profiles_detail.csv")
    detail = pd.read_csv(detail_path) if os.path.exists(detail_path) else pd.DataFrame()
    return summary, detail


def prepare(summary, detail):
    num_cols = [
        "inlines",
        "xlines",
        "samples",
        "worker_count",
        "execution_time_sec",
        "peak_memory_usage_bytes",
        "avg_memory_usage_bytes",
    ]
    present = [c for c in num_cols if c in summary.columns]
    summary[present] = summary[present].apply(pd.to_numeric, errors="coerce")
    if not detail.empty:
        detail_present = [c for c in num_cols if c in detail.columns]
        detail[detail_present] = detail[detail_present].apply(
            pd.to_numeric, errors="coerce"
        )
    return summary, detail


def merge_totals(summary, detail):
    """Merge the per-process memory usage into a total for each session."""
    if detail.empty:
        summary["total_peak_memory_usage_bytes"] = np.nan
        return summary
    cols = [
        "inlines",
        "xlines",
        "samples",
        "chunking_mode",
        "worker_count",
        "timestamp",
        "session_id",
        "random_id",
    ]
    agg = (
        detail.groupby(cols)["peak_memory_usage_bytes"]
        .sum()
        .reset_index(name="total_peak_memory_usage_bytes")
    )
    return summary.merge(agg, on=cols, how="left")


def plot_by_shape(df, charts_dir):
    sns.set_theme(style="whitegrid")
    os.makedirs(charts_dir, exist_ok=True)
    shapes = df[["inlines", "xlines", "samples"]].drop_duplicates()
    for _, r in shapes.iterrows():
        inl, xl, smp = int(r.inlines), int(r.xlines), int(r.samples)
        sel = df[(df["inlines"] == inl) & (df["xlines"] == xl) & (df["samples"] == smp)]
        if sel.empty:
            continue
        label = f"{inl}x{xl}x{smp}"

        fig, ax = plt.subplots()
        sns.lineplot(
            data=sel[~sel["oom_or_failed"]],
            x="worker_count",
            y="execution_time_sec",
            hue="chunking_mode",
            marker="o",
            ax=ax,
        )
        ax.set(
            title=f"Execution Time vs Workers ({label})",
            xlabel="Workers",
            ylabel="Time (s)",
        )
        fig.savefig(os.path.join(charts_dir, f"{label}_time.pdf"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(
            data=sel[~sel["oom_or_failed"]],
            x="worker_count",
            y="total_peak_memory_usage_bytes",
            hue="chunking_mode",
            marker="o",
            ax=ax,
        )
        ax.set(
            title=f"Peak Memory vs Workers ({label})",
            xlabel="Workers",
            ylabel="Memory (bytes)",
        )
        fig.savefig(os.path.join(charts_dir, f"{label}_mem.pdf"), dpi=150)
        plt.close(fig)


def make_leaderboard(df, out_dir):
    ok = df[~df["oom_or_failed"]]
    if ok.empty:
        print("No successful runs for leaderboard.")
        return
    groups = ["inlines", "xlines", "samples", "worker_count"]
    rows = []
    for _, g in ok.groupby(groups):
        best = g.loc[g["execution_time_sec"].idxmin()]
        memaware = g[g["chunking_mode"] == "memaware"]
        diff = "N/A"
        if not memaware.empty:
            m = memaware["execution_time_sec"].min()
            diff = f"{(m - best.execution_time_sec)/best.execution_time_sec*100:.1f}%"
        rows.append(
            {
                **{c: best[c] for c in groups},
                "best_mode": best["chunking_mode"],
                "best_time": best["execution_time_sec"],
                "memaware_diff": diff,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)


def oom_stats(df, out_dir):
    stats = df.groupby("chunking_mode")["oom_or_failed"].sum().reset_index(name="ooms")
    totals = (
        df["chunking_mode"]
        .value_counts()
        .rename_axis("chunking_mode")
        .reset_index(name="count")
    )
    merged = stats.merge(totals, on="chunking_mode")
    merged["oom_ratio"] = merged["ooms"] / merged["count"]
    merged.to_csv(os.path.join(out_dir, "oom_stats_by_mode.csv"), index=False)


def mode_summary(df, out_dir):
    """
    Build mode-level summary but exclude memaware runs in any shape/worker_count
    config if either 'autom' or 'evenly_split' had OOM for that same config.
    """
    shape_cols = ["inlines", "xlines", "samples", "worker_count"]

    trouble = df[
        df["chunking_mode"].isin(["autom", "evenly_split"]) & (df["oom_or_failed"])
    ]
    trouble_cfgs = trouble[shape_cols].drop_duplicates()

    trouble_set = set(tuple(x) for x in trouble_cfgs.values)

    ok = df[~df["oom_or_failed"]].copy()

    def is_trouble(row):
        key = (row["inlines"], row["xlines"], row["samples"], row["worker_count"])
        return (row["chunking_mode"] == "memaware") and (key in trouble_set)

    ok = ok[~ok.apply(is_trouble, axis=1)]

    if ok.empty:
        return

    summary = (
        ok.groupby("chunking_mode")
        .agg(
            avg_exec_time_sec=("execution_time_sec", "mean"),
            median_exec_time_sec=("execution_time_sec", "median"),
            avg_peak_memory_bytes=("total_peak_memory_usage_bytes", "mean"),
            median_peak_memory_bytes=("total_peak_memory_usage_bytes", "median"),
            runs=("chunking_mode", "count"),
        )
        .reset_index()
    )

    summary.to_csv(os.path.join(out_dir, "mode_summary_stats.csv"), index=False)


def shape_summary(df, out_dir):
    cols = ["inlines", "xlines", "samples", "chunking_mode"]
    stats = (
        df.groupby(cols)
        .agg(
            avg_exec_time_sec=("execution_time_sec", "mean"),
            median_exec_time_sec=("execution_time_sec", "median"),
            avg_total_peak_memory_bytes=("total_peak_memory_usage_bytes", "mean"),
            median_total_peak_memory_bytes=("total_peak_memory_usage_bytes", "median"),
            total_runs=("oom_or_failed", "size"),
            oom_count=("oom_or_failed", "sum"),
        )
        .reset_index()
    )
    stats["oom_ratio"] = stats["oom_count"] / stats["total_runs"]
    stats.to_csv(os.path.join(out_dir, "shape_summary_stats.csv"), index=False)


def main():
    args = parse_args()
    summary, detail = load_data(args.results_dir)
    summary, detail = prepare(summary, detail)
    combined = merge_totals(summary, detail)

    plot_by_shape(combined, args.charts_dir)

    make_leaderboard(combined, args.results_dir)
    oom_stats(combined, args.results_dir)
    mode_summary(combined, args.results_dir)
    shape_summary(combined, args.results_dir)

    print(
        f"Analysis complete. Charts at {args.charts_dir}, CSV summaries in {args.results_dir}"
    )


if __name__ == "__main__":
    main()
