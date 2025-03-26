"""
Analyzes memory usage prediction experiment results for Memory-Aware Chunking.

1. Reads environment variables (OUTPUT_DIR, OPERATORS_DIR).
2. Finds operator directories and their CSV results.
3. Produces numerous plots based on memory usage, model performance,
   data reduction, feature selection, and additional insights.
4. Produces cross-operator plots aggregating data across all operators.

Newly Added Charts:
- actual_vs_predicted_by_model.pdf (all operators)
- feature_impact.pdf (all operators)
- memory_progression_envelope.pdf (per operator)
- metrics_evolution_by_number_of_features.pdf (cross)
- metrics_evolution_by_sample_size.pdf (cross)
- residual_metrics_by_number_of_features.pdf (cross)
- residual_metrics_by_sample_size.pdf (cross)
- residual_vs_predicted.pdf (cross)

Additional Extended Analyses (Section 9):
- Time-to-Peak Memory Plots
- Linear Scaling-Factor (Slope) Analysis
- Feature Correlation Heatmaps
- Residual vs. Single Feature
- Best Model Summary (Cross-Operator)
- Data Reduction “Breakdown Point” (Inflection)
- HPC Resource “Safety Margin” Plots
"""

import ast
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------------------------
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

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out")
OPERATORS_DIR = os.getenv("OPERATORS_DIR", f"{OUTPUT_DIR}/results/operators")


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    """Coordinates the entire analysis."""
    print("Analyzing results...\n")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"OPERATORS_DIR: {OPERATORS_DIR}")
    print()

    # 1. Gather operator data
    results = load_all_operators(OPERATORS_DIR)

    # 2. Run single-operator analyses
    analyze_profile(results)
    analyze_model(results)
    analyze_data_reduction(results)
    analyze_feature_selection(results)
    analyze_additional_insights(results)

    # 3. Generate cross-operator charts
    analyze_profile_cross(results)
    analyze_model_cross(results)
    analyze_data_reduction_cross(results)
    analyze_feature_selection_cross(results)
    analyze_additional_insights_cross(results)

    # 9. New enhancements (extended analyses)
    analyze_new_enhancements(results)


# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
def load_all_operators(operators_dir):
    """
    Loads result CSV files for each operator found in `operators_dir`.
    Returns a dict: { operator_name: {csv_name_without_ext: DataFrame}, ... }
    """
    print("---- STEP 1: Loading all operator data ----")
    if not os.path.isdir(operators_dir):
        print(f"Operators directory not found: {operators_dir}")
        return {}

    operators = [op for op in os.listdir(operators_dir) if not op.startswith(".")]
    print(f"Found {len(operators)} operators: {operators}\n")

    all_results = {}
    for operator in operators:
        operator_folder = os.path.join(operators_dir, operator, "results")
        if os.path.isdir(operator_folder):
            operator_data = {}
            for file in os.listdir(operator_folder):
                if file.lower().endswith(".csv"):
                    key = os.path.splitext(file)[0]
                    csv_path = os.path.join(operator_folder, file)
                    operator_data[key] = pd.read_csv(csv_path)
            all_results[operator] = operator_data

    return all_results


# ------------------------------------------------------------------------------
# Analysis Section (Profile) - Single Operator
# ------------------------------------------------------------------------------
def analyze_profile(results):
    """
    Per-operator analysis of memory/time profiling, saved to <out>/charts/profile/.
    """
    print("---- STEP 2: Analyzing Profile (Memory & Time) ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "profile_summary" not in dfs or "profile_history" not in dfs:
            print("  -> Missing 'profile_summary' or 'profile_history'. Skipping.\n")
            continue

        summary = dfs["profile_summary"]
        history = dfs["profile_history"]

        # Directory for profile charts
        profile_dir = os.path.join(OUTPUT_DIR, "charts", "profile")
        os.makedirs(profile_dir, exist_ok=True)

        # Existing standard memory/time profile plots
        plot_peak_memory_usage_per_volume(summary, operator, profile_dir)
        plot_memory_usage_distribution(history, operator, profile_dir)
        plot_inline_xline_progression(history, operator, profile_dir)
        plot_memory_usage_heatmap_by_time(history, operator, profile_dir)
        plot_memory_usage_by_configuration(history, operator, profile_dir)
        plot_inlines_xlines_heatmap(history, operator, profile_dir)
        plot_inlines_xlines_samples_3d(history, operator, profile_dir)
        plot_execution_time_by_volume(summary, operator, profile_dir)
        plot_execution_time_distribution(summary, operator, profile_dir)
        plot_execution_time_distribution_by_volume(history, operator, profile_dir)

        # NEW: memory_progression_envelope.pdf (per operator)
        plot_memory_progression(history, operator, profile_dir)

        print()
    print()


def plot_peak_memory_usage_per_volume(df, operator, out_dir):
    print(f"  -> Peak memory usage per volume for {operator}")
    fig, ax1 = plt.subplots()
    ax1.plot(
        df["volume"],
        df["peak_memory_usage_avg"],
        marker="o",
        zorder=3,
        label="Avg Mem Usage",
    )

    # Fill std dev area
    if "peak_memory_usage_std_dev" in df.columns:
        low = df["peak_memory_usage_avg"] - df["peak_memory_usage_std_dev"]
        high = df["peak_memory_usage_avg"] + df["peak_memory_usage_std_dev"]
        ax1.fill_between(df["volume"], low, high, alpha=0.2, zorder=2)

    ax1.set_xlabel("Volume")
    ax1.set_ylabel("Peak Memory (GB)")

    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )
    plt.xticks(rotation=45, ha="right")

    # Plot Coefficient of Variation on second axis if available
    if "peak_memory_usage_cv" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            df["volume"],
            df["peak_memory_usage_cv"],
            marker="s",
            linestyle="--",
            zorder=3,
            color="tab:orange",
            label="Coeff Variation",
        )
        ax2.set_ylabel("Coefficient of Variation (CV)")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.title(f"Peak Memory Usage + Variability (GB) - {operator}")
    filename = f"peak_memory_by_volume_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_memory_usage_distribution(df, operator, out_dir):
    print(f"  -> Memory usage distribution for {operator}")
    fig, ax = plt.subplots()
    sns.violinplot(
        data=df,
        x="volume",
        y="captured_memory_usage",
        inner="quartile",
        cut=0,
        density_norm="width",
        bw_adjust=0.8,
        ax=ax,
        zorder=3,
    )
    ax.set_title(f"Memory Usage Distribution by Volume - {operator}")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Memory Usage (GB)")

    volumes = df["volume"].unique()
    ax.set_xticks(range(len(volumes)))
    ax.set_xticklabels(
        [format_volume_label(v) for v in volumes], rotation=45, ha="right"
    )

    filename = f"memory_usage_distribution_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_inline_xline_progression(df, operator, out_dir):
    print(f"  -> Inline/Xline memory progression for {operator}")
    g = sns.FacetGrid(
        df,
        col="inlines",
        row="xlines",
        margin_titles=True,
        height=3,
        aspect=2,
        despine=False,
    )
    g.map_dataframe(
        sns.lineplot, x="relative_time", y="captured_memory_usage", marker="o"
    )
    g.set_axis_labels("Relative Time", "Captured Memory (GB)")
    g.set_titles(col_template="Inlines={col_name}", row_template="Xlines={row_name}")

    for ax in g.axes.flatten():
        sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
        ax.set_axisbelow(True)

    fig = g.fig
    fig.suptitle(f"Inline/Xline Memory Progression - {operator}", y=1.05)

    filename = f"inline_xline_memory_usage_progression_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_memory_usage_heatmap_by_time(df, operator, out_dir):
    print(f"  -> Memory usage heatmap by time for {operator}")
    ph = df.copy()
    if "relative_time" not in ph.columns or "volume" not in ph.columns:
        print("  -> Skipping heatmap by time (columns missing).")
        return

    ph["time_bin"] = pd.cut(ph["relative_time"], bins=50, labels=False)
    # Use quantile bins for volume to avoid collisions
    ph["volume_bin"] = pd.qcut(ph["volume"], q=10, labels=False, duplicates="drop")

    pivoted = ph.pivot_table(
        index="volume_bin",
        columns="time_bin",
        values="captured_memory_usage",
        aggfunc="mean",
    )

    fig, ax = plt.subplots()
    sns.heatmap(pivoted, cmap="viridis", ax=ax)
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Volume Bin")
    ax.set_title(f"Memory Usage Over Time (Mean) - {operator}")

    filename = f"memory_usage_heatmap_by_time_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_memory_usage_by_configuration(df, operator, out_dir):
    print(f"  -> Memory usage by configuration (3D) for {operator}")
    if not all(
        col in df.columns
        for col in ["session_id", "volume", "inlines", "xlines", "samples"]
    ):
        print("  -> Skipping 3D config plot (columns missing).")
        return

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    grouped = df.groupby(["session_id", "volume", "inlines", "xlines", "samples"])
    for _, subset in grouped:
        ax.plot(
            subset["relative_time"],
            subset["volume"],
            subset["captured_memory_usage"],
            zorder=3,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Volume")
    ax.set_zlabel("Mem Usage (GB)")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )
    ax.view_init(elev=20, azim=140)
    ax.set_title(f"Memory Usage Over Time by Config - {operator}")

    filename = f"memory_usage_by_configuration_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_inlines_xlines_heatmap(df, operator, out_dir):
    print(f"  -> Memory usage inlines/xlines heatmap for {operator}")
    if not all(
        col in df.columns for col in ["inlines", "xlines", "captured_memory_usage"]
    ):
        print("  -> Skipping inlines/xlines heatmap (columns missing).")
        return

    pivoted = df.groupby(["inlines", "xlines"])["captured_memory_usage"].max().unstack()

    fig, ax = plt.subplots()
    sns.heatmap(pivoted, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_xlabel("Xlines")
    ax.set_ylabel("Inlines")
    ax.set_title(f"Peak Memory Usage Heatmap - {operator}")

    filename = f"memory_usage_inlines_xlines_heatmap_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_inlines_xlines_samples_3d(df, operator, out_dir):
    print(f"  -> 3D memory usage (inlines/xlines/samples) for {operator}")
    needed = ["inlines", "xlines", "samples", "captured_memory_usage"]
    if not all(col in df.columns for col in needed):
        print("  -> Skipping 3D memory usage plot (columns missing).")
        return

    grouped = (
        df.groupby(["inlines", "xlines", "samples"])["captured_memory_usage"]
        .max()
        .reset_index()
    )

    X = grouped["inlines"].values
    Y = grouped["xlines"].values
    Z = grouped["samples"].values
    C = grouped["captured_memory_usage"].values

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(X, Y, Z, c=C, cmap="viridis", s=50)
    ax.set_xlabel("Inlines")
    ax.set_ylabel("Xlines")
    ax.set_zlabel("Samples")
    ax.set_title(f"3D Peak Memory Usage - {operator}")

    cbar = fig.colorbar(sc, shrink=0.5, aspect=5)
    cbar.set_label("Memory Usage (GB)")

    filename = f"memory_usage_inlines_xlines_samples_heatmap_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_execution_time_by_volume(df, operator, out_dir):
    print(f"  -> Execution time by volume for {operator}")
    fig, ax = plt.subplots()
    ax.plot(
        df["volume"], df["execution_time_avg"], marker="o", label="Avg Time", zorder=3
    )

    if all(k in df.columns for k in ["execution_time_min", "execution_time_max"]):
        ax.fill_between(
            df["volume"],
            df["execution_time_min"],
            df["execution_time_max"],
            alpha=0.2,
            zorder=2,
        )

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )
    ax.set_xlabel("Volume")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title(f"Execution Time by Volume - {operator}")
    ax.legend()

    filename = f"execution_time_by_volume_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_execution_time_distribution(df, operator, out_dir):
    print(f"  -> Execution time distribution for {operator}")
    if "execution_time_avg" not in df.columns:
        print("  -> Skipping execution time distribution (column missing).")
        return

    fig, ax = plt.subplots()
    sns.histplot(df["execution_time_avg"], bins=10, kde=True, ax=ax, zorder=3)
    ax.set_xlabel("Total Execution Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Execution Time Distribution - {operator}")

    filename = f"execution_time_distribution_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_execution_time_distribution_by_volume(df, operator, out_dir):
    print(f"  -> Execution time distribution by volume for {operator}")
    needed = ["session_id", "timestamp", "volume"]
    if not all(col in df.columns for col in needed):
        print("  -> Skipping execution time distribution by volume (columns missing).")
        return

    grouped = (
        df.groupby("session_id")
        .agg(
            total_execution_time=("timestamp", lambda x: (x.max() - x.min()) / 1e9),
            volume=("volume", "first"),
        )
        .reset_index()
    )
    grouped["volume_label"] = grouped["volume"].apply(format_volume_label)

    fig, ax = plt.subplots()
    sns.boxplot(
        data=grouped,
        hue="volume",
        x="volume_label",
        y="total_execution_time",
        ax=ax,
        zorder=3,
    )
    ax.set_xlabel("Volume")
    ax.set_ylabel("Total Execution Time (s)")
    ax.set_title(f"Execution Time Distribution by Volume - {operator}")

    filename = f"execution_time_distribution_by_volume_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_memory_progression(df, operator, out_dir):
    """
    Generates a memory-usage progression chart for a single operator, faceted by xlines/inlines,
    with lines colored/styled by 'samples'.

    If operator == 'gst3d', it applies smoothing/downsampling to the data.
    Otherwise, it plots data as-is.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 1) Optional smoothing/downsampling function
    def smooth_and_downsample(sub_df):
        sub_df = sub_df.sort_values("relative_time").copy()
        # Rolling average
        sub_df["captured_memory_usage"] = (
            sub_df["captured_memory_usage"]
            .rolling(window=5, center=True, min_periods=1)
            .mean()
        )
        # Downsampling: keep every 5th point
        sub_df = sub_df.iloc[::5, :]
        return sub_df

    # 2) If operator == "gst3d", group by shape columns and apply smoothing
    if operator.lower() == "gst3d":
        df = (
            df.groupby(["xlines", "inlines", "samples"], group_keys=True)
            .apply(smooth_and_downsample)
            .reset_index(drop=True)
        )

    # 3) Convert columns to strings for Seaborn faceting
    df["xlines"] = df["xlines"].astype(str)
    df["inlines"] = df["inlines"].astype(str)
    df["samples"] = df["samples"].astype(str)

    # 4) Use Seaborn relplot
    g = sns.relplot(
        data=df,
        x="relative_time",
        y="captured_memory_usage",
        row="xlines",
        col="inlines",
        hue="samples",
        style="samples",
        kind="line",
        estimator=None,  # we want raw data, not an aggregator
        facet_kws={"sharex": False, "sharey": False},
        height=3,
        aspect=1.4,
    )

    # 5) Cosmetic adjustments
    g.set_axis_labels("Relative Time", "Memory Usage (GB)")
    g.set_titles(row_template="Xlines={row_name}", col_template="Inlines={col_name}")
    g.fig.suptitle(
        f"Memory Usage Progression - Operator: {operator}", fontsize=21, y=1.01
    )

    # Draw black borders around each facet
    for ax in g.axes.flat:
        rect = ax.patch
        rect.set_edgecolor("black")
        rect.set_linewidth(1.5)

    # Access Seaborn's automatic legend
    legend = g._legend
    if legend:
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_linestyle("--")
        # Move legend if desired
        legend.set_bbox_to_anchor((1.0, 0.5))

    g.tight_layout()
    out_path = os.path.join(out_dir, f"memory_progression_{operator}.pdf")
    g.fig.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


# ------------------------------------------------------------------------------
# Analysis Section (Model) - Single Operator
# ------------------------------------------------------------------------------
def analyze_model(results):
    """
    Per-operator analysis of model metrics, saved to <out>/charts/model/.
    """
    print("---- STEP 3: Analyzing Model Metrics ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "model_metrics" not in dfs:
            print("  -> Missing 'model_metrics'. Skipping.\n")
            continue

        metrics = dfs["model_metrics"]
        model_dir = os.path.join(OUTPUT_DIR, "charts", "model")
        os.makedirs(model_dir, exist_ok=True)

        plot_model_performance(metrics, operator, model_dir)
        plot_model_score(metrics, operator, model_dir)
        plot_model_acc_vs_rmse(metrics, operator, model_dir)
        plot_residual_distribution(metrics, operator, model_dir)
        plot_actual_vs_predicted(metrics, operator, model_dir)
        print()
    print()


def plot_model_performance(df, operator, out_dir):
    print(f"  -> Model performance for {operator}")
    models = df["model_name"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(x - width * 1.5, df["rmse"], width, label="RMSE", zorder=3)
    ax.bar(x - width * 0.5, df["mae"], width, label="MAE", zorder=3)
    ax.bar(x + width * 0.5, df["r2"], width, label="R²", zorder=3)
    ax.bar(x + width * 1.5, df["accuracy"], width, label="Accuracy", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title(f"Comparison of Model Performance - {operator}")
    ax.legend()

    filename = f"performance_by_model_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_model_score(df, operator, out_dir):
    print(f"  -> Model score for {operator}")
    models = df["model_name"]
    scores = df["score"]
    max_score = scores.max()

    fig, ax = plt.subplots()
    ax.bar(models, scores, zorder=3)
    ax.axhline(max_score, linestyle="--", label="Top Score", zorder=4)
    ax.text(
        x=len(models) - 1,
        y=max_score + 0.01 * max_score,
        s=f"Top Score: {max_score:.3f}",
        ha="right",
        va="top",
        zorder=5,
    )

    ax.set_title(f"Model Ranking by Score - {operator}")
    ax.legend()
    plt.xticks(rotation=45, ha="right")

    filename = f"score_by_model_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_model_acc_vs_rmse(df, operator, out_dir):
    print(f"  -> Model Accuracy vs RMSE for {operator}")
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        ax.scatter(
            row["rmse"],
            row["accuracy"],
            label=row["model_name"],
            s=100,
            zorder=3,
        )
    ax.set_xlabel("RMSE (Lower is Better)")
    ax.set_ylabel("Accuracy (Higher is Better)")
    ax.set_title(f"Accuracy vs. RMSE - {operator}")
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.1))

    filename = f"accuracy_by_rmse_per_model_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_residual_distribution(df, operator, out_dir):
    print(f"  -> Residual distribution for {operator}")
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        # Use ast.literal_eval
        residuals = ast.literal_eval(row["residuals"])
        sns.kdeplot(residuals, fill=True, alpha=0.3, label=row["model_name"], ax=ax)

    ax.axvline(0, linestyle="dashed")
    ax.set_xlabel("Residual Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution by Model - {operator}")
    ax.legend()

    filename = f"residuals_distribution_by_model_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_actual_vs_predicted(df, operator, out_dir):
    print(f"  -> Actual vs. Predicted for {operator}")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in df.iterrows():
        if i >= len(axes):
            break
        y_test = ast.literal_eval(row["y_test"])
        y_pred = ast.literal_eval(row["y_pred"])
        sns.regplot(
            x=y_test,
            y=y_pred,
            ax=axes[i],
            scatter_kws={"zorder": 3},
            line_kws={"zorder": 4},
        )
        axes[i].set_title(row["model_name"])
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")

    plt.tight_layout()
    filename = f"actual_vs_predicted_by_model_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


# ------------------------------------------------------------------------------
# Analysis Section (Data Reduction) - Single Operator
# ------------------------------------------------------------------------------
def analyze_data_reduction(results):
    """
    Per-operator analysis of data reduction, saved to <out>/charts/data_reduction/.
    """
    print("---- STEP 4: Analyzing Data Reduction ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "data_reduction" not in dfs:
            print("  -> Missing 'data_reduction'. Skipping.\n")
            continue

        dr = dfs["data_reduction"]
        dr_dir = os.path.join(OUTPUT_DIR, "charts", "data_reduction")
        os.makedirs(dr_dir, exist_ok=True)

        plot_metrics_by_sample_size(dr, operator, dr_dir)
        plot_score_by_sample_size(dr, operator, dr_dir)
        plot_rmse_mae_ratio_by_sample_size(dr, operator, dr_dir)
        plot_residual_distribution_by_sample_size(dr, operator, dr_dir)
        print()
    print()


def plot_metrics_by_sample_size(df, operator, out_dir):
    print(f"  -> Metrics by sample size for {operator}")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics_map = {
        "RMSE": ("rmse", axes[0, 0]),
        "MAE": ("mae", axes[0, 1]),
        "R²": ("r2", axes[1, 0]),
        "Accuracy": ("accuracy", axes[1, 1]),
    }
    for title, (col, ax) in metrics_map.items():
        ax.plot(df["num_samples"], df[col], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Num Samples")

    plt.tight_layout()
    filename = f"metrics_evolution_by_sample_size_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_score_by_sample_size(df, operator, out_dir):
    print(f"  -> Score by sample size for {operator}")
    fig, ax = plt.subplots()
    ax.plot(df["num_samples"], df["score"], marker="o", zorder=3)
    ax.set_title(f"Model Score vs. Number of Samples - {operator}")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Score")

    filename = f"score_by_sample_size_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_rmse_mae_ratio_by_sample_size(df, operator, out_dir):
    print(f"  -> RMSE/MAE ratio by sample size for {operator}")
    ratio = df["rmse"] / df["mae"]
    fig, ax = plt.subplots()
    ax.plot(df["num_samples"], ratio, marker="o", zorder=3)
    ax.set_title(f"RMSE/MAE Ratio Over Data Reduction - {operator}")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE/MAE")

    filename = f"rmse_mae_ratio_by_sample_size_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_residual_distribution_by_sample_size(df, operator, out_dir):
    print(f"  -> Residual distribution by sample size for {operator}")
    data = df.copy()
    data["residuals"] = data["residuals"].apply(ast.literal_eval)
    data["mean"] = data["residuals"].apply(np.mean)
    data["std"] = data["residuals"].apply(np.std)
    data["mae_calc"] = data["residuals"].apply(lambda x: np.mean(np.abs(x)))
    data["rmse_calc"] = data["residuals"].apply(
        lambda x: np.sqrt(np.mean(np.square(x)))
    )

    fig, ax = plt.subplots()
    ax.plot(data["num_samples"], data["mae_calc"], marker="o", label="MAE", zorder=3)
    ax.plot(data["num_samples"], data["rmse_calc"], marker="s", label="RMSE", zorder=3)

    ax.fill_between(
        data["num_samples"],
        data["mae_calc"] - data["std"],
        data["mae_calc"] + data["std"],
        alpha=0.2,
        zorder=2,
    )

    ax.set_title(f"Error Metrics vs Dataset Size - {operator}")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Error")
    ax.legend()

    filename = f"residual_metrics_by_sample_size_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


# ------------------------------------------------------------------------------
# Analysis Section (Feature Selection) - Single Operator
# ------------------------------------------------------------------------------
def analyze_feature_selection(results):
    """
    Per-operator analysis of feature selection, saved to <out>/charts/feature_selection/.
    """
    print("---- STEP 5: Analyzing Feature Selection ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "feature_selection" not in dfs:
            print("  -> Missing 'feature_selection'. Skipping.\n")
            continue

        fs = dfs["feature_selection"]
        fs_dir = os.path.join(OUTPUT_DIR, "charts", "feature_selection")
        os.makedirs(fs_dir, exist_ok=True)

        plot_metrics_by_feature_count(fs, operator, fs_dir)
        plot_score_by_feature_count(fs, operator, fs_dir)
        plot_rmse_mae_ratio_by_feature_count(fs, operator, fs_dir)
        plot_residual_by_feature_count(fs, operator, fs_dir)
        plot_feature_performance(fs, operator, fs_dir)
        print()
    print()


def plot_metrics_by_feature_count(df, operator, out_dir):
    print(f"  -> Metrics by feature count for {operator}")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics_map = {
        "RMSE": ("rmse", axes[0, 0]),
        "MAE": ("mae", axes[0, 1]),
        "R²": ("r2", axes[1, 0]),
        "Accuracy": ("accuracy", axes[1, 1]),
    }
    for title, (col, ax) in metrics_map.items():
        ax.plot(df["num_features"], df[col], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Num Features")

    save_chart(
        fig,
        os.path.join(
            out_dir, f"metrics_evolution_by_number_of_features_{operator}.pdf"
        ),
    )


def plot_score_by_feature_count(df, operator, out_dir):
    print(f"  -> Score by number of features for {operator}")
    fig, ax = plt.subplots()
    ax.plot(df["num_features"], df["score"], marker="o", zorder=3)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Score vs. Number of Features - {operator}")

    filename = f"score_by_number_of_features_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_rmse_mae_ratio_by_feature_count(df, operator, out_dir):
    print(f"  -> RMSE/MAE ratio by number of features for {operator}")
    ratio = df["rmse"] / df["mae"]
    fig, ax = plt.subplots()
    ax.plot(df["num_features"], ratio, marker="o", zorder=3)
    ax.set_title(f"RMSE/MAE Ratio Over Feature Selection - {operator}")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RMSE/MAE")

    filename = f"rmse_mae_ratio_by_number_of_features_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_residual_by_feature_count(df, operator, out_dir):
    print(f"  -> Residual distribution by feature count for {operator}")
    data = df.copy()
    data["residuals"] = data["residuals"].apply(ast.literal_eval)
    data["std"] = data["residuals"].apply(np.std)
    data["mae_calc"] = data["residuals"].apply(lambda x: np.mean(np.abs(x)))
    data["rmse_calc"] = data["residuals"].apply(
        lambda x: np.sqrt(np.mean(np.square(x)))
    )

    fig, ax = plt.subplots()
    ax.plot(data["num_features"], data["mae_calc"], marker="o", label="MAE", zorder=3)
    ax.plot(data["num_features"], data["rmse_calc"], marker="s", label="RMSE", zorder=3)

    ax.fill_between(
        data["num_features"],
        data["mae_calc"] - data["std"],
        data["mae_calc"] + data["std"],
        alpha=0.2,
        zorder=2,
    )

    ax.set_title(f"Residual Metrics vs Number of Features - {operator}")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Error")
    ax.legend()

    filename = f"residual_metrics_by_number_of_features_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_feature_performance(df, operator, out_dir):
    print(f"  -> Feature performance impact for {operator}")
    import ast

    data = df.copy()
    data["selected_features"] = data["selected_features"].apply(ast.literal_eval)
    data.sort_values("num_features", ascending=False, inplace=True, ignore_index=True)

    # Attempt to see single-feature removal impact
    impact_records = []
    for i in range(len(data) - 1):
        current_feats = set(data.loc[i, "selected_features"])
        next_feats = set(data.loc[i + 1, "selected_features"])
        removed = current_feats - next_feats

        if len(removed) == 1:
            [removed_feat] = removed
            delta_rmse = data.loc[i + 1, "rmse"] - data.loc[i, "rmse"]
            impact_records.append(
                {
                    "removed_feature": removed_feat,
                    "delta_rmse": delta_rmse,
                }
            )

    if not impact_records:
        print("  -> No single-feature removal steps found. Skipping.\n")
        return

    impact_df = pd.DataFrame(impact_records)
    avg_impact = (
        impact_df.groupby("removed_feature")["delta_rmse"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()
    sns.barplot(x=avg_impact.index, y=avg_impact.values, ax=ax, zorder=3)
    ax.set_xlabel("Removed Feature")
    ax.set_ylabel("Avg ΔRMSE")
    ax.set_title(f"Impact of Removing Each Feature on RMSE - {operator}")
    plt.xticks(rotation=90)

    filename = f"feature_impact_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


# ------------------------------------------------------------------------------
# EXTRA INSIGHTS - Single Operator
# ------------------------------------------------------------------------------
def analyze_additional_insights(results):
    """
    Additional per-operator analyses, saved to <out>/charts/insights/.
    """
    print("---- EXTRA: Additional Explorations & Plots ----")
    for operator, dfs in results.items():
        print(f"Additional insights for operator: {operator}")
        insights_dir = os.path.join(OUTPUT_DIR, "charts", "insights")
        os.makedirs(insights_dir, exist_ok=True)

        if "profile_summary" in dfs:
            psum = dfs["profile_summary"]
            plot_memory_vs_volume_regression(psum, operator, insights_dir)

        if "profile_history" in dfs:
            phist = dfs["profile_history"]
            # This requires columns: "inlines", "xlines", "samples"
            if all(col in phist.columns for col in ["inlines", "xlines", "samples"]):
                plot_memory_vs_dimensions(phist, operator, insights_dir)

        if "model_metrics" in dfs:
            mmetrics = dfs["model_metrics"]
            plot_residual_vs_predicted(mmetrics, operator, insights_dir)
            plot_residual_qq(mmetrics, operator, insights_dir)

        if "profile_summary" in dfs and all(
            c in dfs["profile_summary"].columns
            for c in ["peak_memory_usage_avg", "execution_time_avg"]
        ):
            plot_execution_time_vs_memory(
                dfs["profile_summary"], operator, insights_dir
            )


def plot_memory_vs_volume_regression(df, operator, out_dir):
    """
    Plots memory usage vs volume with a simple linear regression overlay.
    Expects columns: "volume" and "peak_memory_usage_avg".
    """
    if not all(col in df.columns for col in ["volume", "peak_memory_usage_avg"]):
        print(
            f"  -> Missing columns in profile_summary for memory vs volume. Skipping."
        )
        return

    print(f"  -> Memory vs. Volume (Regression) for {operator}")
    x = df["volume"].values
    y = df["peak_memory_usage_avg"].values

    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Observed", zorder=3)
    ax.plot(x, m * x + b, color="red", label=f"Lin Fit: y={m:.4f}x+{b:.4f}", zorder=4)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Avg Peak Memory (GB)")
    ax.set_title(f"Memory vs. Volume (Linear Fit) - {operator}")
    ax.legend()

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: format_volume_label(v))
    )

    filename = f"memory_vs_volume_regression_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_memory_vs_dimensions(df, operator, out_dir):
    """
    Creates a pairplot to see how memory usage (captured_memory_usage)
    correlates with inlines, xlines, samples.
    """
    needed_cols = ["inlines", "xlines", "samples", "captured_memory_usage"]
    if not all(col in df.columns for col in needed_cols):
        print(
            f"  -> Missing needed columns in profile_history for dimension pairplot. Skipping."
        )
        return

    print(f"  -> Memory vs. Dimensions Pairplot for {operator}")
    subset = df[needed_cols].copy()

    g = sns.pairplot(
        subset,
        kind="reg",
        plot_kws={"line_kws": {"color": "red"}},
        diag_kind="kde",
    )
    g.fig.suptitle(f"Memory vs. Dimensions (Pairplot) - {operator}", y=1.02)

    out_path = os.path.join(out_dir, f"memory_vs_dimensions_pairplot_{operator}.pdf")
    g.fig.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def plot_residual_vs_predicted(mmetrics, operator, out_dir):
    """
    Plots residual (y_pred - y_test) vs. predicted for each model.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not all(
        col in mmetrics.columns for col in ["model_name", "residuals", "y_pred"]
    ):
        print(
            "  -> Missing needed columns in model_metrics for residual vs. predicted. Skipping."
        )
        return

    print(f"  -> Residual vs. Predicted for {operator}")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in mmetrics.iterrows():
        if i >= len(axes):
            break

        model_name = row["model_name"]
        residuals = np.array(ast.literal_eval(row["residuals"]))
        y_pred = np.array(ast.literal_eval(row["y_pred"]))

        if len(residuals) != len(y_pred):
            continue

        ax = axes[i]
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(0, linestyle="--", color="red")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residual")
        ax.set_title(f"{model_name}")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Residual vs. Predicted - {operator}", fontsize=16)

    out_path = os.path.join(out_dir, f"residual_vs_predicted_{operator}.pdf")
    fig.savefig(out_path)
    plt.close(fig)


def plot_residual_qq(mmetrics, operator, out_dir):
    """
    Creates a QQ-Plot for each model's residual distribution to check normality.
    """
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    if not all(col in mmetrics.columns for col in ["model_name", "residuals"]):
        print("  -> Missing needed columns in model_metrics for QQ-plot. Skipping.")
        return

    print(f"  -> Residual QQ-Plot for {operator}")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in mmetrics.iterrows():
        if i >= len(axes):
            break
        model_name = row["model_name"]
        residuals = np.array(ast.literal_eval(row["residuals"]))

        ax = axes[i]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"QQ-Plot: {model_name}")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.suptitle(f"QQ-Plots of Residuals - {operator}", fontsize=16)

    out_path = os.path.join(out_dir, f"residual_qq_plots_{operator}.pdf")
    fig.savefig(out_path)
    plt.close(fig)


def plot_execution_time_vs_memory(df, operator, out_dir):
    """
    Scatter of execution_time_avg vs. peak_memory_usage_avg
    to see if there's a relationship between time and memory usage.
    """
    print(f"  -> Execution Time vs. Memory for {operator}")
    fig, ax = plt.subplots()
    ax.scatter(df["peak_memory_usage_avg"], df["execution_time_avg"], alpha=0.7)

    ax.set_xlabel("Avg Peak Memory (GB)")
    ax.set_ylabel("Avg Execution Time (s)")
    ax.set_title(f"Execution Time vs. Memory Usage - {operator}")

    out_path = os.path.join(out_dir, f"execution_time_vs_memory_{operator}.pdf")
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Cross-Operator Analyses (Profile)
# ------------------------------------------------------------------------------
def analyze_profile_cross(results):
    """
    Creates cross-operator charts using 'profile_summary' or 'profile_history'
    from each operator. Saved to <out>/charts/cross/profile/.
    """
    print("---- CROSS-OPERATOR: Profile ----")
    cross_dir = os.path.join(OUTPUT_DIR, "charts", "cross", "profile")
    os.makedirs(cross_dir, exist_ok=True)

    # Gather 'profile_summary' from all operators
    summary_data = []
    for operator, dfs in results.items():
        if "profile_summary" in dfs:
            df = dfs["profile_summary"].copy()
            df["operator"] = operator
            summary_data.append(df)
    if summary_data:
        summary_all = pd.concat(summary_data, ignore_index=True)
        plot_cross_peak_memory_by_volume(summary_all, cross_dir)
        plot_cross_execution_time_by_volume(summary_all, cross_dir)
    else:
        print("  -> No operator has 'profile_summary' for cross-operator charts.")


def plot_cross_peak_memory_by_volume(df, out_dir):
    """
    Plots a cross-operator line chart of peak_memory_usage_avg vs volume.
    Each operator is a separate line.
    """
    fig, ax = plt.subplots()
    operators = df["operator"].unique()
    for op in operators:
        subset = df[df["operator"] == op].sort_values("volume")
        ax.plot(
            subset["volume"],
            subset["peak_memory_usage_avg"],
            marker="o",
            label=op,
            zorder=3,
        )

    ax.set_xlabel("Volume")
    ax.set_ylabel("Peak Memory Usage (GB)")
    ax.set_title("Cross-Operator: Peak Memory Usage vs Volume")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )
    ax.legend()

    filename = "cross_peak_memory_by_volume.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_cross_execution_time_by_volume(df, out_dir):
    """
    Plots a cross-operator line chart of execution_time_avg vs volume.
    Each operator is a separate line.
    """
    fig, ax = plt.subplots()
    operators = df["operator"].unique()
    for op in operators:
        subset = df[df["operator"] == op].sort_values("volume")
        if "execution_time_avg" not in subset.columns:
            continue
        ax.plot(
            subset["volume"],
            subset["execution_time_avg"],
            marker="s",
            label=op,
            zorder=3,
        )

    ax.set_xlabel("Volume")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Cross-Operator: Execution Time vs Volume")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )
    ax.legend()

    filename = "cross_execution_time_by_volume.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


# ------------------------------------------------------------------------------
# Cross-Operator Analyses (Model)
# ------------------------------------------------------------------------------
def analyze_model_cross(results):
    """
    Creates cross-operator charts using 'model_metrics' from each operator.
    Saved to <out>/charts/cross/model/.
    """
    print("---- CROSS-OPERATOR: Model ----")
    cross_dir = os.path.join(OUTPUT_DIR, "charts", "cross", "model")
    os.makedirs(cross_dir, exist_ok=True)

    model_data = []
    for operator, dfs in results.items():
        if "model_metrics" in dfs:
            df = dfs["model_metrics"].copy()
            df["operator"] = operator
            model_data.append(df)

    if not model_data:
        print("  -> No operator has 'model_metrics' for cross-operator charts.")
        return

    all_metrics = pd.concat(model_data, ignore_index=True)
    plot_cross_model_performance(all_metrics, cross_dir)
    plot_cross_model_rmse(all_metrics, cross_dir)

    # NEW CROSS CHARTS
    # actual_vs_predicted_by_model.pdf (all operators)
    plot_cross_actual_vs_predicted_by_model(all_metrics, cross_dir)
    # residual_vs_predicted.pdf (all operators)
    plot_cross_residual_vs_predicted(all_metrics, cross_dir)


def plot_cross_model_performance(df, out_dir):
    """
    A grouped bar chart showing each operator's model_name vs. R² (example).
    """
    fig, ax = plt.subplots()
    grouped = df.groupby(["operator", "model_name"])["r2"].mean().reset_index()

    pivoted = grouped.pivot(index="model_name", columns="operator", values="r2")
    pivoted.plot(kind="bar", ax=ax, zorder=3)
    ax.set_title("Cross-Operator: Average R² per Model")
    ax.set_ylabel("R²")
    ax.legend(title="Operator", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")

    filename = "cross_model_r2_bar.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_cross_model_rmse(df, out_dir):
    """
    Scatter: average RMSE for each operator-model combination.
    """
    grouped = df.groupby(["operator", "model_name"])["rmse"].mean().reset_index()

    fig, ax = plt.subplots()
    for op in grouped["operator"].unique():
        sub = grouped[grouped["operator"] == op]
        ax.scatter(sub["model_name"], sub["rmse"], label=op, s=100, zorder=3)

    ax.set_title("Cross-Operator: RMSE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.legend()
    plt.xticks(rotation=45, ha="right")

    filename = "cross_model_rmse_scatter.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_cross_actual_vs_predicted_by_model(df, out_dir):
    """
    Creates a multi-panel figure, one panel per model_name.
    Within each panel, it overlays actual vs predicted lines for each operator.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = df["model_name"].unique()
    operators = df["operator"].unique()
    out_path = os.path.join(out_dir, "actual_vs_predicted_by_model.pdf")
    print("  -> Cross: actual_vs_predicted_by_model.pdf")

    # Prepare color map for operators
    color_map = dict(zip(operators, sns.color_palette(n_colors=len(operators))))

    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        sub_model = df[df["model_name"] == model]
        for op in sub_model["operator"].unique():
            sub_op = sub_model[sub_model["operator"] == op]
            for _, row in sub_op.iterrows():
                y_test = ast.literal_eval(row["y_test"])
                y_pred = ast.literal_eval(row["y_pred"])
                # Actual => dashed
                ax.plot(y_test, "--", color=color_map[op])
                # Predicted => solid
                ax.plot(y_pred, "-", color=color_map[op])
        ax.set_title(model)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")

    # Build legends
    # Operator legend
    operator_handles = [
        Line2D([], [], color=color_map[op], marker="o", linestyle="None", label=op)
        for op in operators
    ]
    # Style legend
    style_actual = Line2D([], [], color="black", linestyle="--", label="Actual")
    style_pred = Line2D([], [], color="black", linestyle="-", label="Predicted")

    # Place legends
    fig.legend(
        handles=operator_handles + [style_actual, style_pred],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.96),
        title="Operators / Type",
    )
    fig.suptitle("All Operators - Actual vs. Predicted by Model", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_residual_vs_predicted(df, out_dir):
    """
    Plots residual vs predicted for all operators/models in one chart or multiple subplots.
    """
    out_path = os.path.join(out_dir, "residual_vs_predicted.pdf")
    print("  -> Cross: residual_vs_predicted.pdf")

    # We'll flatten all points and color by operator, style by model.
    all_points = []
    for _, row in df.iterrows():
        if "residuals" not in row or "y_pred" not in row:
            continue
        res = ast.literal_eval(row["residuals"])
        y_pred = ast.literal_eval(row["y_pred"])
        if len(res) != len(y_pred):
            continue
        for r_val, p_val in zip(res, y_pred):
            all_points.append(
                {
                    "operator": row["operator"],
                    "model_name": row["model_name"],
                    "residual": r_val,
                    "predicted": p_val,
                }
            )
    if not all_points:
        print("  -> No residual/predicted data available.")
        return

    df_points = pd.DataFrame(all_points)
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_points,
        x="predicted",
        y="residual",
        hue="operator",
        style="model_name",
        ax=ax,
        alpha=0.6,
    )
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residual")
    ax.set_title("Cross-Operator: Residual vs. Predicted")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_chart(fig, out_path)


# ------------------------------------------------------------------------------
# Cross-Operator Analyses (Data Reduction)
# ------------------------------------------------------------------------------
def analyze_data_reduction_cross(results):
    """
    Creates cross-operator charts using 'data_reduction' from each operator.
    Saved to <out>/charts/cross/data_reduction/.
    """
    print("---- CROSS-OPERATOR: Data Reduction ----")
    cross_dir = os.path.join(OUTPUT_DIR, "charts", "cross", "data_reduction")
    os.makedirs(cross_dir, exist_ok=True)

    dr_data = []
    for operator, dfs in results.items():
        if "data_reduction" in dfs:
            df = dfs["data_reduction"].copy()
            df["operator"] = operator
            dr_data.append(df)

    if not dr_data:
        print("  -> No operator has 'data_reduction' for cross-operator charts.")
        return

    all_dr = pd.concat(dr_data, ignore_index=True)
    plot_cross_data_reduction(all_dr, cross_dir)

    # NEW CROSS charts:
    plot_cross_metrics_evolution_by_sample_size(all_dr, cross_dir)
    plot_cross_residual_metrics_by_sample_size(all_dr, cross_dir)


def plot_cross_data_reduction(df, out_dir):
    """
    Example cross-operator plot: average RMSE vs num_samples for each operator.
    """
    fig, ax = plt.subplots()
    operators = df["operator"].unique()
    for op in operators:
        sub = df[df["operator"] == op].sort_values("num_samples")
        ax.plot(sub["num_samples"], sub["rmse"], marker="o", label=op, zorder=3)

    ax.set_title("Cross-Operator: RMSE vs Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE")
    ax.legend()

    filename = "cross_data_reduction_rmse.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_cross_metrics_evolution_by_sample_size(df, out_dir):
    """
    2x2 chart showing RMSE, MAE, R², Accuracy vs. num_samples for each operator in separate lines.
    """
    filename = "metrics_evolution_by_sample_size.pdf"
    out_path = os.path.join(out_dir, filename)
    print(f"  -> Cross: {filename}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics_map = {
        "RMSE": ("rmse", axes[0, 0]),
        "MAE": ("mae", axes[0, 1]),
        "R²": ("r2", axes[1, 0]),
        "Accuracy": ("accuracy", axes[1, 1]),
    }

    for metric_name, (col, ax) in metrics_map.items():
        for op in df["operator"].unique():
            sub = df[df["operator"] == op].sort_values("num_samples")
            ax.plot(sub["num_samples"], sub[col], marker="o", label=op)
        ax.set_title(metric_name)
        ax.set_xlabel("Number of Samples")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.suptitle("Cross-Operator: Metrics Evolution by Sample Size")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_residual_metrics_by_sample_size(df, out_dir):
    """
    Similar to the single-operator version: plots residual-based metrics (MAE, RMSE) vs sample_size,
    lines for each operator.
    """
    filename = "residual_metrics_by_sample_size.pdf"
    out_path = os.path.join(out_dir, filename)
    print(f"  -> Cross: {filename}")

    # If "residuals" exist, compute metrics from them. Otherwise, fallback to 'mae','rmse'.
    df = df.copy()
    if "residuals" in df.columns:
        df["residuals_eval"] = df["residuals"].apply(
            lambda x: np.array(ast.literal_eval(x)) if pd.notna(x) else []
        )
        df["mae_calc"] = df["residuals_eval"].apply(
            lambda x: np.mean(np.abs(x)) if len(x) else np.nan
        )
        df["rmse_calc"] = df["residuals_eval"].apply(
            lambda x: np.sqrt(np.mean(x**2)) if len(x) else np.nan
        )
    else:
        df["mae_calc"] = df["mae"]
        df["rmse_calc"] = df["rmse"]

    fig, ax = plt.subplots()
    ops = df["operator"].unique()
    for op in ops:
        sub = df[df["operator"] == op].sort_values("num_samples")
        ax.plot(sub["num_samples"], sub["mae_calc"], marker="o", label=f"{op} MAE")
        ax.plot(sub["num_samples"], sub["rmse_calc"], marker="s", label=f"{op} RMSE")

    ax.set_title("Cross-Operator: Residual Metrics vs. Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Error")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Cross-Operator Analyses (Feature Selection)
# ------------------------------------------------------------------------------
def analyze_feature_selection_cross(results):
    """
    Creates cross-operator charts using 'feature_selection' from each operator.
    Saved to <out>/charts/cross/feature_selection/.
    """
    print("---- CROSS-OPERATOR: Feature Selection ----")
    cross_dir = os.path.join(OUTPUT_DIR, "charts", "cross", "feature_selection")
    os.makedirs(cross_dir, exist_ok=True)

    fs_data = []
    for operator, dfs in results.items():
        if "feature_selection" in dfs:
            df = dfs["feature_selection"].copy()
            df["operator"] = operator
            fs_data.append(df)

    if not fs_data:
        print("  -> No operator has 'feature_selection' for cross-operator charts.")
        return

    all_fs = pd.concat(fs_data, ignore_index=True)
    plot_cross_feature_selection(all_fs, cross_dir)

    # NEW CROSS: feature_impact.pdf, metrics_evolution_by_number_of_features.pdf,
    #            residual_metrics_by_number_of_features.pdf
    plot_cross_feature_impact(all_fs, cross_dir)
    plot_cross_metrics_evolution_by_number_of_features(all_fs, cross_dir)
    plot_cross_residual_metrics_by_number_of_features(all_fs, cross_dir)


def plot_cross_feature_selection(df, out_dir):
    """
    Example cross-operator chart: line plot of average R² vs num_features, per operator.
    """
    fig, ax = plt.subplots()
    operators = df["operator"].unique()
    for op in operators:
        sub = (
            df[df["operator"] == op].groupby("num_features")["r2"].mean().reset_index()
        )
        ax.plot(sub["num_features"], sub["r2"], marker="o", label=op, zorder=3)

    ax.set_title("Cross-Operator: R² vs Number of Features")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("R²")
    ax.legend()

    filename = "cross_feature_selection_r2.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_cross_feature_impact(df, out_dir):
    """
    Aggregates single-feature removal steps for all operators in one chart.
    """
    filename = "feature_impact.pdf"
    out_path = os.path.join(out_dir, filename)
    print(f"  -> Cross: {filename}")

    df = df.copy()
    df["selected_features"] = df["selected_features"].apply(ast.literal_eval)
    # Sort by operator, then descending num_features
    df.sort_values(["operator", "num_features"], ascending=[True, False], inplace=True)

    records = []
    for op in df["operator"].unique():
        sub_op = df[df["operator"] == op].reset_index(drop=True)
        for i in range(len(sub_op) - 1):
            feats_now = set(sub_op.loc[i, "selected_features"])
            feats_next = set(sub_op.loc[i + 1, "selected_features"])
            removed = feats_now - feats_next
            if len(removed) == 1:
                [removed_feat] = removed
                d_rmse = sub_op.loc[i + 1, "rmse"] - sub_op.loc[i, "rmse"]
                records.append(
                    {
                        "operator": op,
                        "removed_feature": removed_feat,
                        "delta_rmse": d_rmse,
                    }
                )

    if not records:
        print("No single-feature removal steps found in cross feature_selection data.")
        return

    rec_df = pd.DataFrame(records)
    fig, ax = plt.subplots()
    sns.barplot(data=rec_df, x="removed_feature", y="delta_rmse", hue="operator", ax=ax)
    ax.set_xlabel("Removed Feature")
    ax.set_ylabel("Avg ΔRMSE")
    ax.set_title("Cross-Operator: Impact of Removing Each Feature on RMSE")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_metrics_evolution_by_number_of_features(df, out_dir):
    """
    2x2 chart: RMSE, MAE, R², Accuracy vs. num_features for each operator.
    """
    filename = "metrics_evolution_by_number_of_features.pdf"
    out_path = os.path.join(out_dir, filename)
    print(f"  -> Cross: {filename}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics_map = {
        "RMSE": ("rmse", axes[0, 0]),
        "MAE": ("mae", axes[0, 1]),
        "R²": ("r2", axes[1, 0]),
        "Accuracy": ("accuracy", axes[1, 1]),
    }

    for metric_name, (col, ax) in metrics_map.items():
        for op in df["operator"].unique():
            sub = df[df["operator"] == op].sort_values("num_features")
            ax.plot(sub["num_features"], sub[col], marker="o", label=op)
        ax.set_title(metric_name)
        ax.set_xlabel("Number of Features")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.suptitle("Cross-Operator: Metrics Evolution by Number of Features")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_residual_metrics_by_number_of_features(df, out_dir):
    """
    Similar approach: plot lines for each operator, x=num_features, y=MAE or RMSE from residuals.
    """
    filename = "residual_metrics_by_number_of_features.pdf"
    out_path = os.path.join(out_dir, filename)
    print(f"  -> Cross: {filename}")

    data = df.copy()
    if "residuals" in data.columns:
        data["residuals_eval"] = data["residuals"].apply(
            lambda x: np.array(ast.literal_eval(x)) if pd.notna(x) else []
        )
        data["mae_calc"] = data["residuals_eval"].apply(
            lambda x: np.mean(np.abs(x)) if len(x) else np.nan
        )
        data["rmse_calc"] = data["residuals_eval"].apply(
            lambda x: np.sqrt(np.mean(x**2)) if len(x) else np.nan
        )
    else:
        data["mae_calc"] = data["mae"]
        data["rmse_calc"] = data["rmse"]

    fig, ax = plt.subplots()
    for op in data["operator"].unique():
        sub = data[data["operator"] == op].sort_values("num_features")
        ax.plot(sub["num_features"], sub["mae_calc"], marker="o", label=f"{op} MAE")
        ax.plot(sub["num_features"], sub["rmse_calc"], marker="s", label=f"{op} RMSE")

    ax.set_title("Cross-Operator: Residual Metrics vs. Number of Features")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Error")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Cross-Operator Additional Insights
# ------------------------------------------------------------------------------
def analyze_additional_insights_cross(results):
    """
    Creates cross-operator charts that might combine multiple CSV sources.
    Saved to <out>/charts/cross/insights/.
    """
    print("---- CROSS-OPERATOR: Additional Insights ----")
    cross_dir = os.path.join(OUTPUT_DIR, "charts", "cross", "insights")
    os.makedirs(cross_dir, exist_ok=True)
    # Extend as needed for your cross-operator "insights".


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def format_volume_label(val):
    val = float(val)
    if val >= 1e6:
        return f"{int(val/1e6)}M"
    if val >= 1e3:
        return f"{int(val/1e3)}K"
    return str(int(val))


def save_chart(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------------------
# 9. Additional Extended Analyses
# ------------------------------------------------------------------------------
def analyze_new_enhancements(results):
    """
    Section 9: Additional analyses & plots not covered in previous sections.
    Calls specialized functions below for each operator or across operators.
    """
    print("---- STEP 9: New Enhancements (Extended Analyses) ----")

    # 9.1 Time-to-Peak Memory Analysis
    for operator, dfs in results.items():
        if "profile_history" in dfs:
            ph = dfs["profile_history"]
            out_dir = os.path.join(OUTPUT_DIR, "charts", "insights")
            os.makedirs(out_dir, exist_ok=True)
            plot_time_to_peak(ph, operator, out_dir)

    # 9.2 HPC Memory Scaling-Factor (Slope) - CROSS
    #    We can do a cross-operator approach if each operator has 'profile_summary'.
    summaries = []
    for operator, dfs in results.items():
        if "profile_summary" in dfs:
            psum = dfs["profile_summary"].copy()
            psum["operator"] = operator
            summaries.append(psum)
    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        out_dir = os.path.join(OUTPUT_DIR, "charts", "cross", "insights")
        os.makedirs(out_dir, exist_ok=True)
        plot_memory_scaling_factor(all_summary, out_dir)

    # 9.3 Feature Correlation Heatmaps (Per Operator)
    for operator, dfs in results.items():
        if "profile_summary" in dfs:
            psum = dfs["profile_summary"].copy()
            out_dir = os.path.join(OUTPUT_DIR, "charts", "insights")
            os.makedirs(out_dir, exist_ok=True)
            plot_feature_correlation(psum, operator, out_dir)

    # 9.4 Best Model Summary (Cross-Operator)
    #    We'll use the cross-operator model_metrics to find top model per operator
    model_data = []
    for operator, dfs in results.items():
        if "model_metrics" in dfs:
            mm = dfs["model_metrics"].copy()
            mm["operator"] = operator
            model_data.append(mm)
    if model_data:
        all_mm = pd.concat(model_data, ignore_index=True)
        out_dir = os.path.join(OUTPUT_DIR, "charts", "cross", "model")
        os.makedirs(out_dir, exist_ok=True)
        plot_best_model_per_operator(all_mm, out_dir)

    # 9.5 HPC Resource “Safety Margin” Plots (Per Operator)
    for operator, dfs in results.items():
        if "profile_history" in dfs:
            ph = dfs["profile_history"].copy()
            out_dir = os.path.join(OUTPUT_DIR, "charts", "insights")
            os.makedirs(out_dir, exist_ok=True)
            plot_hpc_safety_margin(ph, operator, out_dir, percentile=0.95)

    # 9.6 Residual vs. Each Feature (Optional) - If you have merged data to link residuals to shape.
    #    (Placeholder, see the docstring in the snippet from earlier suggestions.)
    #    If your pipeline merges them, you could call something like:
    #    plot_residuals_vs_features(merged_df, operator, out_dir)
    #    after you have a merged DataFrame of shape + model residual info.


def plot_time_to_peak(df, operator, out_dir):
    """
    For each session_id, find the time (relative_time) when memory usage is max.
    Then plot a distribution or boxplot across volumes.
    """
    needed_cols = ["session_id", "captured_memory_usage", "relative_time", "volume"]
    if not all(col in df.columns for col in needed_cols):
        print(
            f"  -> Missing columns for time-to-peak analysis in {operator}. Skipping."
        )
        return

    print(f"  -> Time-to-Peak Memory for {operator}")
    grouped = df.groupby("session_id")
    peak_times = []
    for sid, sub in grouped:
        max_mem = sub["captured_memory_usage"].max()
        idx = sub["captured_memory_usage"].idxmax()
        t_peak = sub.loc[idx, "relative_time"]
        volume = sub.loc[idx, "volume"]
        peak_times.append({"session_id": sid, "time_to_peak": t_peak, "volume": volume})

    ptdf = pd.DataFrame(peak_times)

    fig, ax = plt.subplots()
    sns.boxplot(data=ptdf, x="volume", y="time_to_peak", ax=ax)
    ax.set_title(f"Time to Peak Memory Usage - {operator}")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Time to Peak (s)")
    volumes = sorted(ptdf["volume"].unique())
    ax.set_xticklabels(
        [format_volume_label(v) for v in volumes], rotation=45, ha="right"
    )

    filename = f"time_to_peak_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_memory_scaling_factor(summary_all, out_dir):
    """
    For each operator, fit peak_memory_usage_avg = m*volume + b,
    then plot bar chart of slope (m).
    """
    print("  -> Cross-Operator Memory Scaling Factor")
    operators = summary_all["operator"].unique()
    slope_data = []

    for op in operators:
        sub = summary_all[summary_all["operator"] == op].dropna(
            subset=["volume", "peak_memory_usage_avg"]
        )
        if len(sub) < 2:
            continue
        x = sub["volume"].values
        y = sub["peak_memory_usage_avg"].values
        m, b = np.polyfit(x, y, 1)
        slope_data.append({"operator": op, "slope": m})

    if not slope_data:
        print("  -> Not enough data for memory scaling factor.")
        return

    sdf = pd.DataFrame(slope_data)

    fig, ax = plt.subplots()
    sns.barplot(data=sdf, x="operator", y="slope", ax=ax)
    ax.set_title("Memory Scaling Factor (Slope) by Operator")
    ax.set_xlabel("Operator")
    ax.set_ylabel("Slope (GB per volume unit)")

    filename = "cross_operator_memory_scaling_factor.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_feature_correlation(summary, operator, out_dir):
    """
    Create a correlation heatmap among shape-based columns and memory/time metrics.
    Expects columns like: 'inlines', 'xlines', 'samples', 'volume',
    'peak_memory_usage_avg', 'execution_time_avg'.
    """
    needed = [
        "inlines",
        "xlines",
        "samples",
        "volume",
        "peak_memory_usage_avg",
        "execution_time_avg",
    ]
    existing_cols = [c for c in needed if c in summary.columns]
    if len(existing_cols) < 3:
        print(f"  -> Skipping correlation heatmap for {operator} (not enough columns).")
        return

    print(f"  -> Feature Correlation Heatmap for {operator}")
    corr_df = summary[existing_cols].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(f"Feature Correlations - {operator}")

    filename = f"feature_correlation_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_best_model_per_operator(all_metrics, out_dir):
    """
    Creates a bar chart of best model (max score) for each operator,
    using a simpler approach that doesn't require 'include_group_keys'.
    """
    print("  -> Cross-Operator Best Model per Operator")

    # Instead of groupby.apply(...), do this one-liner:
    # 1) group your df by "operator"
    # 2) find the index of the max "score" for each group
    # 3) select those rows from all_metrics
    best_idxs = all_metrics.groupby("operator")["score"].idxmax()
    best_rows = all_metrics.loc[best_idxs].reset_index(drop=True)

    fig, ax = plt.subplots()
    x = np.arange(len(best_rows))

    ax.bar(x, best_rows["score"], color="skyblue", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(best_rows["operator"], rotation=45, ha="right")

    # Optionally label each bar with the best model name
    for i, row in best_rows.iterrows():
        ax.text(
            i,
            row["score"],
            row["model_name"],
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9,
            zorder=4,
        )

    ax.set_title("Best Model Score per Operator")
    ax.set_ylabel("Score")

    filename = "best_model_per_operator.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


def plot_hpc_safety_margin(df, operator, out_dir, percentile=0.95):
    needed = ["session_id", "captured_memory_usage", "volume"]
    if not all(col in df.columns for col in needed):
        print(f"  -> Missing columns for HPC safety margin in {operator}. Skipping.")
        return

    print(f"  -> HPC Safety Margin for {operator} (pct={percentile})")

    grouped = df.groupby("volume")["captured_memory_usage"]

    # Step 1: compute mean usage
    stats_df = grouped.agg(avg_usage="mean").reset_index()

    # Step 2: compute the p95 usage
    stats_df["p95_usage"] = grouped.apply(
        lambda x: np.percentile(x, percentile * 100)
    ).values

    fig, ax = plt.subplots()
    x = np.arange(len(stats_df))

    ax.bar(x - 0.2, stats_df["avg_usage"], width=0.4, label="Avg", zorder=3)
    ax.bar(
        x + 0.2,
        stats_df["p95_usage"],
        width=0.4,
        label=f"{int(percentile*100)}th pct",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_volume_label(v) for v in stats_df["volume"]], rotation=45, ha="right"
    )
    ax.set_title(f"Memory Safety Margin - {operator}")
    ax.set_ylabel("Memory Usage (GB)")
    ax.legend()

    filename = f"memory_safety_margin_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


# ------------------------------------------------------------------------------
# (Optional) Residual vs Single Feature Stub
# ------------------------------------------------------------------------------
def plot_residuals_vs_features(
    merged_df, operator, out_dir, features=["inlines", "xlines", "samples"]
):
    """
    EXAMPLE (not automatically called by default).

    For each model, plots residuals vs. each specified feature.

    NOTE: This requires that you have a single DataFrame `merged_df`
    that contains columns:
      - 'model_name'
      - 'residual'
      - 'inlines', 'xlines', 'samples' (or your shape features)
      - optional: 'session_id'
    If you haven't merged the shape info with the model residuals, do so first.
    """
    pass


# ------------------------------------------------------------------------------
# Run the Script
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
