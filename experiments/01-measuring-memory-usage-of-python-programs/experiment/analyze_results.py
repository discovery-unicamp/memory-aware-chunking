"""
Analyzes memory usage from 'profiles_detail.csv', producing a series of charts & tables:
1) Memory usage history by non-traceq tools (all lines on one chart).
2) Memory usage history by tool (one chart per tool, including traceq).
3) Comparison of memory usage history (each base tool vs. its traceq).
4) Kernel density (KDE) for each base tool vs. traceq counterpart.
5) CDF for each base tool vs. traceq counterpart.
6) Table(s) comparing phases (min/max/mean).
7) Table comparing base_tool in a single merged CSV.
8) Bar chart: max memory usage (original vs. traceq).
9) Boxplot: memory distribution (base vs. traceq).
10) Chart: how many points collected (base vs. traceq).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")
mpl.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.edgecolor": "black",
        "grid.color": "gray",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.markeredgewidth": 1,
        "lines.markeredgecolor": "black",
        "lines.linewidth": 1.5,
    }
)


def main():
    out_dir = os.getenv("OUTPUT_DIR", "./out")
    results_dir = os.path.join(out_dir, "results")
    analysis_dir = os.path.join(out_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    detail_csv = os.path.join(results_dir, "profiles_detail.csv")
    if not os.path.isfile(detail_csv):
        print(f"Error: {detail_csv} not found. Make sure collect_results.py was run.")
        return

    # Load the detail data
    df = pd.read_csv(detail_csv)  # columns: [tool, step, memory_mb]

    # 1) Memory Usage History by Tool (excluding traceq_* tools)
    plot_memory_history_no_traceq(df, analysis_dir)

    # 2) Memory Usage per Tool (subplot for each tool, including traceq)
    plot_memory_usage_per_tool(df, analysis_dir)

    # 3) Comparison of memory usage history: original vs traceq
    #    (One chart per base tool.)
    compare_memory_usage_orig_vs_traceq(df, analysis_dir)

    # 4) Kernel Density for each base tool vs. traceq
    plot_kde_base_vs_traceq(df, analysis_dir)

    # 5) CDF for each base tool vs. traceq
    plot_cdf_base_vs_traceq(df, analysis_dir)

    # 6) Tables comparing phases (min / max / mean). We'll produce a CSV.
    produce_phase_tables(df, analysis_dir)

    # 7) Table comparing base_tool
    #    We'll build a DataFrame that merges original vs traceq stats
    #    and save as 'traceq_comparison.csv'.
    produce_traceq_comparison_table(df, analysis_dir)

    # 8) Max memory usage: original vs traceq bar chart
    plot_max_memory_orig_vs_traceq_bar(df, analysis_dir)

    # 9) Boxplot of memory distribution by tool comparing base with traceq
    boxplot_base_vs_traceq(df, analysis_dir)

    # 10) Chart with how many points collected (per base tool)
    plot_points_collected(df, analysis_dir)

    print("All analyses complete.")


# ---------------------------------------------------------------------------
# Utility: Distinguish base vs traceq
# ---------------------------------------------------------------------------
def is_traceq_tool(tool_name: str) -> bool:
    return tool_name.startswith("traceq_")


def get_base_tool(tool_name: str) -> str:
    """
    Return the base name for e.g. 'traceq_psutil' -> 'psutil', else same.
    """
    if is_traceq_tool(tool_name):
        return tool_name.replace("traceq_", "")
    return tool_name


# ---------------------------------------------------------------------------
# (1) Memory Usage History by Tool (excluding traceq)
# ---------------------------------------------------------------------------
def plot_memory_history_no_traceq(df: pd.DataFrame, analysis_dir: str):
    """
    Single figure with lines for each tool *not* starting with 'traceq_'.
    """
    out_path = os.path.join(analysis_dir, "memory_history_no_traceq.pdf")

    # Filter out traceq tools
    df_no_traceq = df[~df["tool"].str.startswith("traceq_")].copy()
    if df_no_traceq.empty:
        print("No non-traceq tools found. Skipping plot_memory_history_no_traceq.")
        return

    plt.figure(figsize=(10, 6))
    for tool_name in sorted(df_no_traceq["tool"].unique()):
        subset = df_no_traceq[df_no_traceq["tool"] == tool_name]
        plt.plot(subset["step"], subset["memory_mb"], marker="o", label=tool_name)

    plt.title("Memory Usage History by Tool (No TraceQ)")
    plt.xlabel("Sample/Step Index")
    plt.ylabel("Memory Usage (MB)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[1] Saved: {out_path}")


# ---------------------------------------------------------------------------
# (2) Memory Usage PER tool (one chart for each, including traceq)
# ---------------------------------------------------------------------------
def plot_memory_usage_per_tool(df: pd.DataFrame, analysis_dir: str):
    """
    For each tool in the DataFrame, produce a standalone PDF of its memory-usage line.
    """
    # Make sure the output dir exists
    os.makedirs(analysis_dir, exist_ok=True)

    for tool_name in sorted(df["tool"].unique()):
        subset = df[df["tool"] == tool_name].sort_values("step")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(subset["step"], subset["memory_mb"], marker="o", linewidth=1.5)
        ax.set_title(f"{tool_name} Memory Usage History")
        ax.set_xlabel("Sample/Step Index")
        ax.set_ylabel("Memory (MB)")
        ax.grid(True)

        out_path = os.path.join(analysis_dir, f"memory_usage_{tool_name}.pdf")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        print(f"[2] Saved: {out_path}")


# ---------------------------------------------------------------------------
# (3) Comparison of memory usage history: original vs traceq (one per base)
# ---------------------------------------------------------------------------
def compare_memory_usage_orig_vs_traceq(df: pd.DataFrame, analysis_dir: str):
    """
    For each 'base tool' that also exists as 'traceq_base', produce a single figure
    overlaying base vs. traceq memory usage lines.
    """
    out_path_template = os.path.join(analysis_dir, "compare_orig_vs_traceq_{}.pdf")

    base_tools = set()
    for t in df["tool"].unique():
        if not is_traceq_tool(t):
            # does 'traceq_t' exist?
            traceq_variant = "traceq_" + t
            if traceq_variant in df["tool"].values:
                base_tools.add(t)

    for base in sorted(base_tools):
        traceq_name = f"traceq_{base}"

        fig, ax = plt.subplots(figsize=(8, 5))
        # Base
        sub_base = df[df["tool"] == base].sort_values("step")
        ax.plot(sub_base["step"], sub_base["memory_mb"], marker="o", label=f"{base}")

        # TraceQ
        sub_tq = df[df["tool"] == traceq_name].sort_values("step")
        ax.plot(sub_tq["step"], sub_tq["memory_mb"], marker="s", label=f"{traceq_name}")

        ax.set_title(f"Original vs TraceQ: {base}")
        ax.set_xlabel("Sample/Step Index")
        ax.set_ylabel("Memory (MB)")
        ax.grid(True)
        ax.legend()

        out_path = out_path_template.format(base)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[3] Saved: {out_path}")


# ---------------------------------------------------------------------------
# (4) Kernel Density for each tool comparing with traceq (one chart per pair)
# ---------------------------------------------------------------------------
def plot_kde_base_vs_traceq(df: pd.DataFrame, analysis_dir: str):
    """
    For each base tool that also has a traceq_ version, produce a 2-line KDE plot.
    """
    out_path_template = os.path.join(analysis_dir, "kde_base_vs_traceq_{}.pdf")

    # Identify base tools that have a traceq version
    base_tools = set()
    for t in df["tool"].unique():
        if not is_traceq_tool(t):
            if "traceq_" + t in df["tool"].values:
                base_tools.add(t)

    for base in sorted(base_tools):
        base_data = df[df["tool"] == base]["memory_mb"]
        traceq_data = df[df["tool"] == f"traceq_{base}"]["memory_mb"]

        if len(base_data) == 0 or len(traceq_data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(base_data, label=base, fill=False)
        sns.kdeplot(traceq_data, label=f"traceq_{base}", fill=False)

        ax.set_title(f"KDE Memory Distribution: {base} vs TraceQ")
        ax.set_xlabel("Memory (MB)")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()

        out_path = out_path_template.format(base)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[4] Saved: {out_path}")


# ---------------------------------------------------------------------------
# (5) CDF for each tool comparing with traceq (one chart per pair)
# ---------------------------------------------------------------------------
def plot_cdf_base_vs_traceq(df: pd.DataFrame, analysis_dir: str):
    """
    Similar to the KDE chart, but we plot a CDF (cumulative distribution).
    For each base tool that has a traceq version, produce a figure.
    """
    out_path_template = os.path.join(analysis_dir, "cdf_base_vs_traceq_{}.pdf")

    base_tools = set()
    for t in df["tool"].unique():
        if not is_traceq_tool(t):
            if "traceq_" + t in df["tool"].values:
                base_tools.add(t)

    for base in sorted(base_tools):
        base_data = df[df["tool"] == base]["memory_mb"].values
        tq_data = df[df["tool"] == f"traceq_{base}"]["memory_mb"].values

        if len(base_data) == 0 or len(tq_data) == 0:
            continue

        # Compute CDF
        base_sorted = np.sort(base_data)
        base_cdf = np.arange(len(base_sorted)) / float(len(base_sorted))

        tq_sorted = np.sort(tq_data)
        tq_cdf = np.arange(len(tq_sorted)) / float(len(tq_sorted))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(base_sorted, base_cdf, marker=".", linestyle="none", label=base)
        ax.plot(tq_sorted, tq_cdf, marker=".", linestyle="none", label=f"traceq_{base}")

        ax.set_title(f"CDF Memory Distribution: {base} vs TraceQ")
        ax.set_xlabel("Memory (MB)")
        ax.set_ylabel("Cumulative Probability")
        ax.grid(True)
        ax.legend()

        out_path = out_path_template.format(base)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[5] Saved: {out_path}")


# ---------------------------------------------------------------------------
# (6) Tables comparing phases (min, max, mean).
# ---------------------------------------------------------------------------
def produce_phase_tables(df: pd.DataFrame, analysis_dir: str):
    """
    We'll define different step-phase splits for traceq vs original:
      - Original (non-traceq): phases = {1: steps 0..4, 2: 5..10, 3: 11+}
      - TraceQ: phases = {1: steps 0..9, 2: 10..30, 3: 31+}
    We'll produce a CSV with columns: [tool, phase, min_mem, max_mem, mean_mem, nPoints].
    """
    out_path = os.path.join(analysis_dir, "phase_comparison.csv")

    def phase_non_traceq(step):
        if step <= 4:
            return 1
        elif step <= 10:
            return 2
        else:
            return 3

    def phase_traceq(step):
        if step <= 9:
            return 1
        elif step <= 30:
            return 2
        else:
            return 3

    rows = []
    for tool_name in df["tool"].unique():
        sub = df[df["tool"] == tool_name].copy()
        if is_traceq_tool(tool_name):
            sub["phase"] = sub["step"].apply(phase_traceq)
        else:
            sub["phase"] = sub["step"].apply(phase_non_traceq)

        grp = sub.groupby("phase")["memory_mb"]
        for phase_id, group_vals in grp:
            row = {
                "tool": tool_name,
                "phase": phase_id,
                "min_mem": group_vals.min(),
                "max_mem": group_vals.max(),
                "mean_mem": group_vals.mean(),
                "n_points": len(group_vals),
            }
            rows.append(row)

    df_phase = pd.DataFrame(rows)
    df_phase.sort_values(["tool", "phase"], inplace=True)
    df_phase.to_csv(out_path, index=False)
    print(f"[6] Phase table saved to: {out_path}")


# ---------------------------------------------------------------------------
# (7) Table comparing base_tool
# ---------------------------------------------------------------------------
def produce_traceq_comparison_table(df: pd.DataFrame, analysis_dir: str):
    """
    We'll compute stats for each tool: (mean_memory, max_memory).
    Then for each base, merge with its traceq version.
    Output a CSV: 'traceq_comparison.csv' with columns:
       base_tool, mean_memory_orig, mean_memory_traceq, mean_diff_mb, ...
       max_memory_orig, max_memory_traceq, ...
    """
    out_path = os.path.join(analysis_dir, "traceq_comparison.csv")

    # 1) Compute stats per tool
    stats = (
        df.groupby("tool")
        .agg(mean_memory=("memory_mb", "mean"), max_memory=("memory_mb", "max"))
        .reset_index()
    )

    # 2) Add base_tool column
    stats["base_tool"] = stats["tool"].apply(get_base_tool)

    # 3) Split into original vs traceq
    df_orig = stats[~stats["tool"].str.startswith("traceq_")].copy()
    df_orig.rename(
        columns={
            "mean_memory": "mean_memory_orig",
            "max_memory": "max_memory_orig",
            "tool": "tool_orig",
        },
        inplace=True,
    )

    df_tq = stats[stats["tool"].str.startswith("traceq_")].copy()
    df_tq.rename(
        columns={
            "mean_memory": "mean_memory_traceq",
            "max_memory": "max_memory_traceq",
            "tool": "tool_traceq",
        },
        inplace=True,
    )

    # 4) Merge on base_tool
    comp = pd.merge(df_orig, df_tq, on="base_tool", how="inner")

    # 5) Compute diffs
    comp["mean_diff_mb"] = comp["mean_memory_traceq"] - comp["mean_memory_orig"]
    comp["mean_diff_pct"] = 100.0 * comp["mean_diff_mb"] / comp["mean_memory_orig"]

    comp["max_diff_mb"] = comp["max_memory_traceq"] - comp["max_memory_orig"]
    comp["max_diff_pct"] = 100.0 * comp["max_diff_mb"] / comp["max_memory_orig"]

    # 6) Rename columns to the desired human-friendly format
    comp_renamed = comp[
        [
            "base_tool",
            "tool_orig",
            "tool_traceq",
            "mean_memory_orig",
            "mean_memory_traceq",
            "mean_diff_mb",
            "mean_diff_pct",
            "max_memory_orig",
            "max_memory_traceq",
            "max_diff_mb",
            "max_diff_pct",
        ]
    ].copy()

    comp_renamed.columns = [
        "Tool",
        "Orig Tool Name",
        "TraceQ Tool Name",
        "Mean Orig (MB)",
        "Mean TraceQ (MB)",
        "Δ Mean (MB)",
        "Δ Mean (%)",
        "Max Orig (MB)",
        "Max TraceQ (MB)",
        "Δ Max (MB)",
        "Δ Max (%)",
    ]

    comp_renamed.to_csv(out_path, index=False)
    print(f"[7] TraceQ comparison table saved to: {out_path}")


# ---------------------------------------------------------------------------
# (8) Max memory usage: original vs traceq bar chart
# ---------------------------------------------------------------------------
def plot_max_memory_orig_vs_traceq_bar(df: pd.DataFrame, analysis_dir: str):
    """
    We'll re-derive the stats from #7 in code, then do a grouped bar chart:
      x-axis: base_tool
      two bars: (max_memory_orig, max_memory_traceq)
    """
    out_path = os.path.join(analysis_dir, "max_memory_orig_vs_traceq.pdf")

    # Gather stats
    stats = df.groupby("tool")["memory_mb"].max().reset_index()
    stats["base_tool"] = stats["tool"].apply(get_base_tool)
    stats_orig = stats[~stats["tool"].str.startswith("traceq_")].copy()
    stats_tq = stats[stats["tool"].str.startswith("traceq_")].copy()

    # rename columns
    stats_orig.rename(columns={"memory_mb": "max_orig"}, inplace=True)
    stats_tq.rename(columns={"memory_mb": "max_traceq"}, inplace=True)

    merged = pd.merge(stats_orig, stats_tq, on="base_tool", how="inner")
    merged = merged.sort_values("base_tool")

    x = np.arange(len(merged))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, merged["max_orig"], width=width, label="Original", zorder=3)
    ax.bar(x + width / 2, merged["max_traceq"], width=width, label="TraceQ", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["base_tool"], rotation=45, ha="right")
    ax.set_title("Max Memory Usage: Original vs TraceQ")
    ax.set_ylabel("Max Memory (MB)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[8] Saved: {out_path}")


# ---------------------------------------------------------------------------
# (9) Boxplot of memory distribution by tool comparing base with traceq
# ---------------------------------------------------------------------------
def boxplot_base_vs_traceq(df: pd.DataFrame, analysis_dir: str):
    """
    We want a single chart (or multiple) that puts, for each base tool,
    two boxplots side by side: base vs traceq.
    We'll gather each pair in a single DataFrame, then plot with Seaborn.
    """
    out_path = os.path.join(analysis_dir, "boxplot_base_vs_traceq.pdf")

    # Identify pairs
    pairs = []
    for t in df["tool"].unique():
        if not is_traceq_tool(t):
            if "traceq_" + t in df["tool"].values:
                pairs.append(t)

    if not pairs:
        print("No base+traceq pairs found. Skipping boxplot_base_vs_traceq.")
        return

    # We'll build a DataFrame with columns: [base_tool, is_traceq, memory_mb]
    # Then we can do a grouped boxplot
    recs = []
    for base in pairs:
        sub_orig = df[df["tool"] == base]["memory_mb"]
        for val in sub_orig:
            recs.append({"base_tool": base, "variant": base, "memory_mb": val})

        sub_tq = df[df["tool"] == f"traceq_{base}"]["memory_mb"]
        for val in sub_tq:
            recs.append(
                {"base_tool": base, "variant": f"traceq_{base}", "memory_mb": val}
            )

    plotdf = pd.DataFrame(recs)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=plotdf, x="base_tool", y="memory_mb", hue="variant", showfliers=False
    )
    plt.title("Memory Distribution: Base vs TraceQ")
    plt.xlabel("Base Tool")
    plt.ylabel("Memory (MB)")
    plt.grid(True, axis="y", linestyle=":", alpha=0.7)
    plt.legend(title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[9] Saved: {out_path}")


# ---------------------------------------------------------------------------
# (10) Chart with how many points collected (per base tool)
# ---------------------------------------------------------------------------
def plot_points_collected(df: pd.DataFrame, analysis_dir: str):
    """
    We'll count how many total rows are in df for each 'tool',
    then group them by base_tool for a side-by-side bar chart (original vs traceq).
    """
    out_path = os.path.join(analysis_dir, "points_collected.pdf")

    # Count rows per tool
    counts = df.groupby("tool")["step"].count().reset_index(name="num_points")
    counts["base_tool"] = counts["tool"].apply(get_base_tool)

    # split
    orig = counts[~counts["tool"].str.startswith("traceq_")].copy()
    tq = counts[counts["tool"].str.startswith("traceq_")].copy()

    orig.rename(columns={"num_points": "orig_points"}, inplace=True)
    tq.rename(columns={"num_points": "traceq_points"}, inplace=True)

    merged = pd.merge(orig, tq, on="base_tool", how="inner")
    merged.sort_values("base_tool", inplace=True)

    x = np.arange(len(merged))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2, merged["orig_points"], width=width, label="Original", zorder=3
    )
    ax.bar(
        x + width / 2, merged["traceq_points"], width=width, label="TraceQ", zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(merged["base_tool"], rotation=45, ha="right")
    ax.set_ylabel("Number of Data Points")
    ax.set_title("Number of Points Collected: Base vs TraceQ")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[10] Saved: {out_path}")


# ------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
