"""
Analyzes memory usage prediction experiment results for Memory-Aware Chunking.

1. Reads environment variables (OUTPUT_DIR, OPERATORS_DIR).
2. Finds operator directories and their CSV results.
3. Produces numerous plots based on memory usage, model performance,
   data reduction, feature selection, and additional insights.
4. Produces cross-operator plots aggregating data across all operators.
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
# Analysis Section (Profile)
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


# ------------------------------------------------------------------------------
# Analysis Section (Model)
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
        # Use safe eval or literal_eval
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
# Analysis Section (Data Reduction)
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
# Analysis Section (Feature Selection)
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

    plt.tight_layout()
    filename = f"metrics_evolution_by_number_of_features_{operator}.pdf"
    save_chart(fig, os.path.join(out_dir, filename))


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
# EXTRA INSIGHTS
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
    Plots residual (y_pred - y_test) vs predicted for each model.
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
# Cross-Operator Analyses
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
        print("  -> No operator has 'profile_summary' to build cross-operator charts.")


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
        print("  -> No operator has 'model_metrics' to build cross-operator charts.")
        return

    all_metrics = pd.concat(model_data, ignore_index=True)
    plot_cross_model_performance(all_metrics, cross_dir)
    plot_cross_model_rmse(all_metrics, cross_dir)


def plot_cross_model_performance(df, out_dir):
    """
    A grouped bar chart showing each operator's model_name vs. R² or Accuracy, etc.
    """
    fig, ax = plt.subplots()
    # Example: group by operator, then plot average R2
    grouped = df.groupby(["operator", "model_name"])["r2"].mean().reset_index()

    # Pivot for easier plotting
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
        print("  -> No operator has 'data_reduction' to build cross-operator charts.")
        return

    all_dr = pd.concat(dr_data, ignore_index=True)
    plot_cross_data_reduction(all_dr, cross_dir)


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
        print(
            "  -> No operator has 'feature_selection' to build cross-operator charts."
        )
        return

    all_fs = pd.concat(fs_data, ignore_index=True)
    plot_cross_feature_selection(all_fs, cross_dir)


def plot_cross_feature_selection(df, out_dir):
    """
    Example cross-operator chart: line plot of average R² vs number_of_features, per operator.
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
# Run the Script
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
