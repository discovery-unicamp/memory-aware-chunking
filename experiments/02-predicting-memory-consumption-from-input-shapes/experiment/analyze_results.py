"""
Analyzes memory usage prediction experiment results for Memory-Aware Chunking.

1. Reads environment variables (OUTPUT_DIR, OPERATORS_DIR).
2. Finds operator directories and their CSV results.
3. Produces numerous plots based on memory usage, model performance,
   data reduction, and feature selection.
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
    print("Analyzing results for MSc project...\n")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"OPERATORS_DIR: {OPERATORS_DIR}")
    print()

    # 1. Gather operator data
    results = load_all_operators(OPERATORS_DIR)

    # 2. Run the various analyses
    analyze_profile(results)
    analyze_model(results)
    analyze_data_reduction(results)
    analyze_feature_selection(results)


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

    # Filter out any hidden junk like .DS_Store
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
    For each operator, analyzes and plots memory usage and execution time
    using 'profile_summary' and 'profile_history' CSVs.
    """
    print("---- STEP 2: Analyzing Profile (Memory & Time) ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "profile_summary" not in dfs or "profile_history" not in dfs:
            print("  -> Missing 'profile_summary' or 'profile_history'. Skipping.\n")
            continue

        summary = dfs["profile_summary"]
        history = dfs["profile_history"]
        out_dir = os.path.join(OPERATORS_DIR, operator, "charts")

        plot_peak_memory_usage_per_volume(summary, operator, out_dir)
        plot_memory_usage_distribution(history, operator, out_dir)
        plot_inline_xline_progression(history, operator, out_dir)
        plot_memory_usage_heatmap_by_time(history, operator, out_dir)
        plot_memory_usage_by_configuration(history, operator, out_dir)
        plot_inlines_xlines_heatmap(history, operator, out_dir)
        plot_inlines_xlines_samples_3d(history, operator, out_dir)
        plot_execution_time_by_volume(summary, operator, out_dir)
        plot_execution_time_distribution(summary, operator, out_dir)
        plot_execution_time_distribution_by_volume(history, operator, out_dir)
        print()
    print()


def plot_peak_memory_usage_per_volume(df, operator, out_dir):
    """Plots average peak memory usage vs volume, with std dev + CV on 2nd axis."""
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
    low = df["peak_memory_usage_avg"] - df["peak_memory_usage_std_dev"]
    high = df["peak_memory_usage_avg"] + df["peak_memory_usage_std_dev"]
    ax1.fill_between(df["volume"], low, high, alpha=0.2, zorder=2)

    ax1.set_xlabel("Volume")
    ax1.set_ylabel("Peak Memory (GB)")

    # Format volumes nicely
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )
    plt.xticks(rotation=45, ha="right")

    # Plot Coefficient of Variation on second axis
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

    plt.title("Peak Memory Usage + Variability (GB)")
    save_chart(fig, os.path.join(out_dir, "peak_memory_by_volume.pdf"))


def plot_memory_usage_distribution(df, operator, out_dir):
    """Plots a violin distribution of memory usage by volume."""
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
    ax.set_title("Memory Usage Distribution by Volume")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Memory Usage (GB)")

    # Format volumes
    volumes = df["volume"].unique()
    ax.set_xticks(range(len(volumes)))
    ax.set_xticklabels(
        [format_volume_label(v) for v in volumes], rotation=45, ha="right"
    )

    save_chart(fig, os.path.join(out_dir, "memory_usage_distribution.pdf"))


def plot_inline_xline_progression(df, operator, out_dir):
    """FacetGrid: memory usage over time, grouped by inlines/xlines."""
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

    # Tweak styling in each subplot
    for ax in g.axes.flatten():
        sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
        ax.set_axisbelow(True)

    fig = g.fig
    save_chart(fig, os.path.join(out_dir, "inline_xline_memory_usage_progression.pdf"))


def plot_memory_usage_heatmap_by_time(df, operator, out_dir):
    """Heatmap: average memory usage over binned time & volume bins."""
    print(f"  -> Memory usage heatmap by time for {operator}")
    ph = df.copy()
    ph["time_bin"] = pd.cut(ph["relative_time"], bins=50, labels=False)
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
    ax.set_title("Memory Usage Over Time (Mean)")

    save_chart(fig, os.path.join(out_dir, "memory_usage_heatmap_by_time.pdf"))


def plot_memory_usage_by_configuration(df, operator, out_dir):
    """3D line plot of memory usage over time, colored by volume."""
    print(f"  -> Memory usage by configuration (3D) for {operator}")
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
    ax.set_title("Memory Usage Over Time by Config")

    save_chart(fig, os.path.join(out_dir, "memory_usage_by_configuration.pdf"))


def plot_inlines_xlines_heatmap(df, operator, out_dir):
    """2D heatmap of peak memory usage by inlines/xlines."""
    print(f"  -> Memory usage inlines/xlines heatmap for {operator}")
    pivoted = df.groupby(["inlines", "xlines"])["captured_memory_usage"].max().unstack()

    fig, ax = plt.subplots()
    sns.heatmap(pivoted, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_xlabel("Xlines")
    ax.set_ylabel("Inlines")
    ax.set_title("Peak Memory Usage Heatmap")

    save_chart(fig, os.path.join(out_dir, "memory_usage_inlines_xlines_heatmap.pdf"))


def plot_inlines_xlines_samples_3d(df, operator, out_dir):
    """3D scatter of peak memory usage across inlines/xlines/samples."""
    print(f"  -> 3D memory usage (inlines/xlines/samples) for {operator}")
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
    ax.set_title("3D Peak Memory Usage")

    cbar = fig.colorbar(sc, shrink=0.5, aspect=5)
    cbar.set_label("Memory Usage (GB)")

    save_chart(
        fig, os.path.join(out_dir, "memory_usage_inlines_xlines_samples_heatmap.pdf")
    )


def plot_execution_time_by_volume(df, operator, out_dir):
    """Plots average execution time vs. volume with min/max fill."""
    print(f"  -> Execution time by volume for {operator}")
    fig, ax = plt.subplots()
    ax.plot(
        df["volume"], df["execution_time_avg"], marker="o", label="Avg Time", zorder=3
    )

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
    ax.set_title("Execution Time by Volume")
    ax.legend()

    save_chart(fig, os.path.join(out_dir, "execution_time_by_volume.pdf"))


def plot_execution_time_distribution(df, operator, out_dir):
    """Histogram of average execution times."""
    print(f"  -> Execution time distribution for {operator}")
    fig, ax = plt.subplots()
    sns.histplot(df["execution_time_avg"], bins=10, kde=True, ax=ax, zorder=3)
    ax.set_xlabel("Total Execution Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Execution Time Distribution")

    save_chart(fig, os.path.join(out_dir, "execution_time_distribution.pdf"))


def plot_execution_time_distribution_by_volume(df, operator, out_dir):
    """Boxplot of total execution time grouped by volume."""
    print(f"  -> Execution time distribution by volume for {operator}")
    # Convert timestamps to total seconds
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
    ax.set_title("Execution Time Distribution by Volume")

    save_chart(fig, os.path.join(out_dir, "execution_time_distribution_by_volume.pdf"))


# ------------------------------------------------------------------------------
# Analysis Section (Model)
# ------------------------------------------------------------------------------
def analyze_model(results):
    """
    For each operator, looks at 'model_metrics' and plots metrics like RMSE, MAE,
    Accuracy, and more.
    """
    print("---- STEP 3: Analyzing Model Metrics ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "model_metrics" not in dfs:
            print("  -> Missing 'model_metrics'. Skipping.\n")
            continue

        metrics = dfs["model_metrics"]
        out_dir = os.path.join(OPERATORS_DIR, operator, "charts")

        plot_model_performance(metrics, operator, out_dir)
        plot_model_score(metrics, operator, out_dir)
        plot_model_acc_vs_rmse(metrics, operator, out_dir)
        plot_residual_distribution(metrics, operator, out_dir)
        plot_actual_vs_predicted(metrics, operator, out_dir)
        print()
    print()


def plot_model_performance(df, operator, out_dir):
    """Bar chart comparing RMSE, MAE, R², and Accuracy side by side per model."""
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
    ax.set_title("Comparison of Model Performance")
    ax.legend()

    save_chart(fig, os.path.join(out_dir, "performance_by_model.pdf"))


def plot_model_score(df, operator, out_dir):
    """Bar chart of a 'score' field, marking the top score with a dashed line."""
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

    ax.set_title("Model Ranking by Score")
    ax.legend()
    plt.xticks(rotation=45, ha="right")

    save_chart(fig, os.path.join(out_dir, "score_by_model.pdf"))


def plot_model_acc_vs_rmse(df, operator, out_dir):
    """Scatter plot of Accuracy vs. RMSE for each model."""
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
    ax.set_title("Accuracy vs. RMSE")
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.1))

    save_chart(fig, os.path.join(out_dir, "accuracy_by_rmse_per_model.pdf"))


def plot_residual_distribution(df, operator, out_dir):
    """KDE distribution of residual errors for each model."""
    print(f"  -> Residual distribution for {operator}")
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        residuals = eval(row["residuals"])  # convert string "[...]" to list
        sns.kdeplot(residuals, fill=True, alpha=0.3, label=row["model_name"], ax=ax)

    ax.axvline(0, linestyle="dashed")
    ax.set_xlabel("Residual Error")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution by Model")
    ax.legend()

    save_chart(fig, os.path.join(out_dir, "residuals_distribution_by_model.pdf"))


def plot_actual_vs_predicted(df, operator, out_dir):
    """Plots actual vs. predicted values for each model in a 3x3 grid."""
    print(f"  -> Actual vs. Predicted for {operator}")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in df.iterrows():
        if i >= len(axes):
            break  # In case there's more than 9 models
        y_test = eval(row["y_test"])
        y_pred = eval(row["y_pred"])
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

    save_chart(fig, os.path.join(out_dir, "actual_vs_predicted_by_model.pdf"))


# ------------------------------------------------------------------------------
# Analysis Section (Data Reduction)
# ------------------------------------------------------------------------------
def analyze_data_reduction(results):
    """
    Looks at 'data_reduction' for each operator and plots metrics across
    different sample sizes.
    """
    print("---- STEP 4: Analyzing Data Reduction ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "data_reduction" not in dfs:
            print("  -> Missing 'data_reduction'. Skipping.\n")
            continue

        dr = dfs["data_reduction"]
        out_dir = os.path.join(OPERATORS_DIR, operator, "charts")

        plot_metrics_by_sample_size(dr, operator, out_dir)
        plot_score_by_sample_size(dr, operator, out_dir)
        plot_rmse_mae_ratio_by_sample_size(dr, operator, out_dir)
        plot_residual_distribution_by_sample_size(dr, operator, out_dir)
        print()
    print()


def plot_metrics_by_sample_size(df, operator, out_dir):
    """Plots RMSE, MAE, R², Accuracy vs. num_samples."""
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

    save_chart(fig, os.path.join(out_dir, "metrics_evolution_by_sample_size.pdf"))


def plot_score_by_sample_size(df, operator, out_dir):
    """Line plot of 'score' vs num_samples."""
    print(f"  -> Score by sample size for {operator}")
    fig, ax = plt.subplots()
    ax.plot(df["num_samples"], df["score"], marker="o", zorder=3)
    ax.set_title("Model Score vs. Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Score")

    save_chart(fig, os.path.join(out_dir, "score_by_sample_size.pdf"))


def plot_rmse_mae_ratio_by_sample_size(df, operator, out_dir):
    """Line plot of the ratio (RMSE/MAE) across different sample sizes."""
    print(f"  -> RMSE/MAE ratio by sample size for {operator}")
    ratio = df["rmse"] / df["mae"]
    fig, ax = plt.subplots()
    ax.plot(df["num_samples"], ratio, marker="o", zorder=3)
    ax.set_title("RMSE/MAE Ratio Over Data Reduction")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE/MAE")

    save_chart(fig, os.path.join(out_dir, "rmse_mae_ratio_by_sample_size.pdf"))


def plot_residual_distribution_by_sample_size(df, operator, out_dir):
    """Plots MAE and RMSE vs. num_samples, plus a fill between for standard dev."""
    print(f"  -> Residual distribution by sample size for {operator}")
    data = df.copy()
    data["residuals"] = data["residuals"].apply(lambda x: eval(x))
    data["mean"] = data["residuals"].apply(np.mean)
    data["std"] = data["residuals"].apply(np.std)
    data["mae_calc"] = data["residuals"].apply(lambda x: np.mean(np.abs(x)))
    data["rmse_calc"] = data["residuals"].apply(
        lambda x: np.sqrt(np.mean(np.square(x)))
    )

    fig, ax = plt.subplots()
    ax.plot(data["num_samples"], data["mae_calc"], marker="o", label="MAE", zorder=3)
    ax.plot(data["num_samples"], data["rmse_calc"], marker="s", label="RMSE", zorder=3)

    # Fill ± std around MAE just as an example
    ax.fill_between(
        data["num_samples"],
        data["mae_calc"] - data["std"],
        data["mae_calc"] + data["std"],
        alpha=0.2,
        zorder=2,
    )

    ax.set_title(f"Error Metrics vs Dataset Size\nOperator: {operator}")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Error")
    ax.legend()

    save_chart(fig, os.path.join(out_dir, "residual_metrics_by_sample_size.pdf"))


# ------------------------------------------------------------------------------
# Analysis Section (Feature Selection)
# ------------------------------------------------------------------------------
def analyze_feature_selection(results):
    """
    For each operator, uses 'feature_selection' to analyze how metrics change
    when features are added/removed.
    """
    print("---- STEP 5: Analyzing Feature Selection ----")
    for operator, dfs in results.items():
        print(f"Analyzing operator: {operator}")
        if "feature_selection" not in dfs:
            print("  -> Missing 'feature_selection'. Skipping.\n")
            continue

        fs = dfs["feature_selection"]
        out_dir = os.path.join(OPERATORS_DIR, operator, "charts")

        plot_metrics_by_feature_count(fs, operator, out_dir)
        plot_score_by_feature_count(fs, operator, out_dir)
        plot_rmse_mae_ratio_by_feature_count(fs, operator, out_dir)
        plot_residual_by_feature_count(fs, operator, out_dir)
        plot_feature_performance(fs, operator, out_dir)
        print()
    print()


def plot_metrics_by_feature_count(df, operator, out_dir):
    """Plots RMSE, MAE, R², Accuracy vs num_features."""
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
        fig, os.path.join(out_dir, "metrics_evolution_by_number_of_features.pdf")
    )


def plot_score_by_feature_count(df, operator, out_dir):
    """Line plot of model 'score' vs. number of features."""
    print(f"  -> Score by number of features for {operator}")
    fig, ax = plt.subplots()
    ax.plot(df["num_features"], df["score"], marker="o", zorder=3)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score")
    ax.set_title("Model Score vs. Number of Features")

    save_chart(fig, os.path.join(out_dir, "score_by_number_of_features.pdf"))


def plot_rmse_mae_ratio_by_feature_count(df, operator, out_dir):
    """Line plot of RMSE/MAE ratio across feature counts."""
    print(f"  -> RMSE/MAE ratio by number of features for {operator}")
    ratio = df["rmse"] / df["mae"]
    fig, ax = plt.subplots()
    ax.plot(df["num_features"], ratio, marker="o", zorder=3)
    ax.set_title("RMSE/MAE Ratio Over Feature Selection")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RMSE/MAE")

    save_chart(fig, os.path.join(out_dir, "rmse_mae_ratio_by_number_of_features.pdf"))


def plot_residual_by_feature_count(df, operator, out_dir):
    """
    Plots MAE & RMSE vs. num_features, plus a fill for standard deviation
    on the residual distribution.
    """
    print(f"  -> Residual distribution by feature count for {operator}")
    data = df.copy()
    data["residuals"] = data["residuals"].apply(lambda x: eval(x))
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

    ax.set_title(f"Residual Metrics vs Number of Features\nOperator: {operator}")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Error")
    ax.legend()

    save_chart(fig, os.path.join(out_dir, "residual_metrics_by_number_of_features.pdf"))


def plot_feature_performance(df, operator, out_dir):
    """
    Attempts to see how removing features affects metrics. Looks for places
    where exactly one feature was removed from one row to the next
    (based on sorted order).
    """
    print(f"  -> Feature performance impact for {operator}")
    data = df.copy()
    data["selected_features"] = data["selected_features"].apply(ast.literal_eval)
    data.sort_values("num_features", ascending=False, inplace=True, ignore_index=True)

    impact_records = []
    for i in range(len(data) - 1):
        current_feats = set(data.loc[i, "selected_features"])
        next_feats = set(data.loc[i + 1, "selected_features"])
        removed = current_feats - next_feats

        # Only handle single-feature removals
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
    ax.set_title("Impact of Removing Each Feature on RMSE")
    plt.xticks(rotation=90)

    save_chart(fig, os.path.join(out_dir, "feature_impact.pdf"))


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def format_volume_label(val):
    """Helper to format numeric volume to a K/M label."""
    val = float(val)
    if val >= 1e6:
        return f"{int(val/1e6)}M"
    if val >= 1e3:
        return f"{int(val/1e3)}K"
    return str(int(val))


def save_chart(fig, out_path):
    """Saves the given matplotlib figure to out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Run the Script
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
