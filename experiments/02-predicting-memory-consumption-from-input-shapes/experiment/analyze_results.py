"""
Analyzes memory usage prediction experiment results for Memory-Aware Chunking,
including extended/new analyses for deeper insights.

1. Reads environment variables (OUTPUT_DIR, OPERATORS_DIR).
2. Finds operator directories and their CSV results.
3. Produces numerous plots based on memory usage, model performance,
   data reduction, feature selection, and additional insights.
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

    # 2. Run the various analyses
    analyze_profile(results)
    analyze_model(results)
    analyze_data_reduction(results)
    analyze_feature_selection(results)

    # 3. New/Extra Explorations (optional)
    analyze_additional_insights(results)


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

    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )
    plt.xticks(rotation=45, ha="right")

    # Plot Coefficient of Variation on second axis
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

    plt.title("Peak Memory Usage + Variability (GB)")
    save_chart(fig, os.path.join(out_dir, "peak_memory_by_volume.pdf"))


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
    ax.set_title("Memory Usage Distribution by Volume")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Memory Usage (GB)")

    volumes = df["volume"].unique()
    ax.set_xticks(range(len(volumes)))
    ax.set_xticklabels(
        [format_volume_label(v) for v in volumes], rotation=45, ha="right"
    )

    save_chart(fig, os.path.join(out_dir, "memory_usage_distribution.pdf"))


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
    save_chart(fig, os.path.join(out_dir, "inline_xline_memory_usage_progression.pdf"))


def plot_memory_usage_heatmap_by_time(df, operator, out_dir):
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
    print(f"  -> Memory usage inlines/xlines heatmap for {operator}")
    pivoted = df.groupby(["inlines", "xlines"])["captured_memory_usage"].max().unstack()

    fig, ax = plt.subplots()
    sns.heatmap(pivoted, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_xlabel("Xlines")
    ax.set_ylabel("Inlines")
    ax.set_title("Peak Memory Usage Heatmap")

    save_chart(fig, os.path.join(out_dir, "memory_usage_inlines_xlines_heatmap.pdf"))


def plot_inlines_xlines_samples_3d(df, operator, out_dir):
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
    ax.set_title("Execution Time by Volume")
    ax.legend()

    save_chart(fig, os.path.join(out_dir, "execution_time_by_volume.pdf"))


def plot_execution_time_distribution(df, operator, out_dir):
    print(f"  -> Execution time distribution for {operator}")
    fig, ax = plt.subplots()
    sns.histplot(df["execution_time_avg"], bins=10, kde=True, ax=ax, zorder=3)
    ax.set_xlabel("Total Execution Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Execution Time Distribution")

    save_chart(fig, os.path.join(out_dir, "execution_time_distribution.pdf"))


def plot_execution_time_distribution_by_volume(df, operator, out_dir):
    print(f"  -> Execution time distribution by volume for {operator}")
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
    print(f"  -> Residual distribution for {operator}")
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        residuals = eval(row["residuals"])
        sns.kdeplot(residuals, fill=True, alpha=0.3, label=row["model_name"], ax=ax)

    ax.axvline(0, linestyle="dashed")
    ax.set_xlabel("Residual Error")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution by Model")
    ax.legend()

    save_chart(fig, os.path.join(out_dir, "residuals_distribution_by_model.pdf"))


def plot_actual_vs_predicted(df, operator, out_dir):
    print(f"  -> Actual vs. Predicted for {operator}")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in df.iterrows():
        if i >= len(axes):
            break
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
    print(f"  -> Score by sample size for {operator}")
    fig, ax = plt.subplots()
    ax.plot(df["num_samples"], df["score"], marker="o", zorder=3)
    ax.set_title("Model Score vs. Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Score")

    save_chart(fig, os.path.join(out_dir, "score_by_sample_size.pdf"))


def plot_rmse_mae_ratio_by_sample_size(df, operator, out_dir):
    print(f"  -> RMSE/MAE ratio by sample size for {operator}")
    ratio = df["rmse"] / df["mae"]
    fig, ax = plt.subplots()
    ax.plot(df["num_samples"], ratio, marker="o", zorder=3)
    ax.set_title("RMSE/MAE Ratio Over Data Reduction")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE/MAE")

    save_chart(fig, os.path.join(out_dir, "rmse_mae_ratio_by_sample_size.pdf"))


def plot_residual_distribution_by_sample_size(df, operator, out_dir):
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
    print(f"  -> Score by number of features for {operator}")
    fig, ax = plt.subplots()
    ax.plot(df["num_features"], df["score"], marker="o", zorder=3)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score")
    ax.set_title("Model Score vs. Number of Features")

    save_chart(fig, os.path.join(out_dir, "score_by_number_of_features.pdf"))


def plot_rmse_mae_ratio_by_feature_count(df, operator, out_dir):
    print(f"  -> RMSE/MAE ratio by number of features for {operator}")
    ratio = df["rmse"] / df["mae"]
    fig, ax = plt.subplots()
    ax.plot(df["num_features"], ratio, marker="o", zorder=3)
    ax.set_title("RMSE/MAE Ratio Over Feature Selection")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RMSE/MAE")

    save_chart(fig, os.path.join(out_dir, "rmse_mae_ratio_by_number_of_features.pdf"))


def plot_residual_by_feature_count(df, operator, out_dir):
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
    print(f"  -> Feature performance impact for {operator}")
    data = df.copy()
    data["selected_features"] = data["selected_features"].apply(ast.literal_eval)
    data.sort_values("num_features", ascending=False, inplace=True, ignore_index=True)

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
    ax.set_title("Impact of Removing Each Feature on RMSE")
    plt.xticks(rotation=90)

    save_chart(fig, os.path.join(out_dir, "feature_impact.pdf"))


# ------------------------------------------------------------------------------
# EXTRA INSIGHTS / NEW ANALYSIS
# ------------------------------------------------------------------------------
def analyze_additional_insights(results):
    """
    OPTIONAL: Additional analyses suggested to deepen insight.
    Feel free to comment out or refine these depending on data availability.
    """
    print("---- EXTRA: Additional Explorations & Plots ----")
    for operator, dfs in results.items():
        print(f"Additional insights for operator: {operator}")
        out_dir = os.path.join(OPERATORS_DIR, operator, "charts")

        # 1. If 'profile_summary' is available, we can do a memory vs volume regression:
        if "profile_summary" in dfs:
            psum = dfs["profile_summary"]
            plot_memory_vs_volume_regression(psum, operator, out_dir)

        # 2. If 'profile_history' has inlines, xlines, samples, do a pairplot:
        if "profile_history" in dfs:
            phist = dfs["profile_history"]
            # This requires that you have columns "inlines", "xlines", "samples"
            if all(col in phist.columns for col in ["inlines", "xlines", "samples"]):
                plot_memory_vs_dimensions(phist, operator, out_dir)

        # 3. If 'model_metrics' is available, we can do residual vs. predicted or QQ
        if "model_metrics" in dfs:
            mmetrics = dfs["model_metrics"]
            plot_residual_vs_predicted(mmetrics, operator, out_dir)
            plot_residual_qq(mmetrics, operator, out_dir)

        # 4. If we want to see execution time vs. memory usage in 'profile_summary'
        if "profile_summary" in dfs and all(
            c in dfs["profile_summary"].columns
            for c in ["peak_memory_usage_avg", "execution_time_avg"]
        ):
            plot_execution_time_vs_memory(dfs["profile_summary"], operator, out_dir)


def plot_memory_vs_volume_regression(df, operator, out_dir):
    """
    Plots memory usage vs. volume with a simple linear regression overlay.
    Expects columns: "volume" and "peak_memory_usage_avg".
    """
    if not all(col in df.columns for col in ["volume", "peak_memory_usage_avg"]):
        print(
            f"  -> Missing needed columns in profile_summary for memory vs. volume. Skipping."
        )
        return

    print(f"  -> Memory vs. Volume (Regression) for {operator}")
    # Fit a simple linear model: y = m*x + b
    x = df["volume"].values
    y = df["peak_memory_usage_avg"].values

    # np.polyfit returns [slope, intercept] if deg=1, but let's do it carefully
    # Actually, polyfit returns [m, b] in a different order with 'deg=1' for polynomial
    # So let's store them carefully:
    m, b = np.polyfit(x, y, 1)  # slope, intercept

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Observed", zorder=3)
    ax.plot(x, m * x + b, color="red", label=f"Lin Fit: y={m:.4f}x+{b:.4f}", zorder=4)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Avg Peak Memory (GB)")
    ax.set_title("Memory vs. Volume with Linear Fit")
    ax.legend()

    # Format volume axis
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: format_volume_label(v))
    )

    save_chart(fig, os.path.join(out_dir, "memory_vs_volume_regression.pdf"))


def plot_memory_vs_dimensions(df, operator, out_dir):
    """
    Creates a pairplot to see how memory usage (captured_memory_usage)
    correlates with inlines, xlines, samples.
    Expects columns: "inlines", "xlines", "samples", "captured_memory_usage".
    """
    needed_cols = ["inlines", "xlines", "samples", "captured_memory_usage"]
    if not all(col in df.columns for col in needed_cols):
        print(
            f"  -> Missing needed columns in profile_history for dimension pairplot. Skipping."
        )
        return

    print(f"  -> Memory vs. Dimensions Pairplot for {operator}")
    subset = df[needed_cols].copy()

    # Large pairplots can be slow for big data; consider sampling if huge
    # subset = subset.sample(n=500, random_state=42)  # if needed

    # Create pairplot
    g = sns.pairplot(
        subset,
        kind="reg",  # includes a regression line in each scatter
        plot_kws={"line_kws": {"color": "red"}},
        diag_kind="kde",
    )
    g.fig.suptitle(f"Memory vs. Dimensions (Pairplot) - {operator}", y=1.02)

    out_path = os.path.join(out_dir, "memory_vs_dimensions_pairplot.pdf")
    g.fig.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def plot_residual_vs_predicted(mmetrics, operator, out_dir):
    """
    Plots residual (y_pred - y_test) vs. predicted for each model.
    Shows if residuals grow with predictions.
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # Check necessary columns
    if not all(
        col in mmetrics.columns for col in ["model_name", "residuals", "y_pred"]
    ):
        print(
            "  -> Missing needed columns in model_metrics for residual vs. predicted. Skipping."
        )
        return

    print(f"  -> Residual vs. Predicted for {operator}")

    # Create subplots (3x3)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in mmetrics.iterrows():
        if i >= len(axes):
            break  # If there are more than 9 models, only plot the first 9

        model_name = row["model_name"]
        residuals = np.array(eval(row["residuals"]))
        y_pred = np.array(eval(row["y_pred"]))

        # Ensure array lengths match
        if len(residuals) != len(y_pred):
            continue

        ax = axes[i]
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(0, linestyle="--", color="red")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residual")
        ax.set_title(f"{model_name}")

    # Adjust layout so the main title doesn't overlap subplots
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Residual vs. Predicted", fontsize=16)

    # Save & close
    out_path = os.path.join(out_dir, "residual_vs_predicted.pdf")
    fig.savefig(out_path)
    plt.close(fig)


def plot_residual_qq(mmetrics, operator, out_dir):
    """
    Creates a QQ-Plot for each model's residual distribution to check normality.
    """
    import numpy as np
    import os
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    # Quick columns check
    if not all(col in mmetrics.columns for col in ["model_name", "residuals"]):
        print("  -> Missing needed columns in model_metrics for QQ-plot. Skipping.")
        return

    print(f"  -> Residual QQ-Plot for {operator}")

    # Create 3x3 subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    # Iterate through rows, each row is a model
    for i, row in mmetrics.iterrows():
        if i >= len(axes):
            break  # If there are more than 9 models, we only plot the first 9
        model_name = row["model_name"]
        residuals = np.array(eval(row["residuals"]))

        ax = axes[i]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"QQ-Plot: {model_name}")

    # Adjust layout so titles won't overlap subplots
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.suptitle("QQ-Plots of Residuals", fontsize=16)

    # Save & close
    out_path = os.path.join(out_dir, "residual_qq_plots.pdf")
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
    ax.set_title("Execution Time vs. Memory Usage")

    out_path = os.path.join(out_dir, "execution_time_vs_memory.pdf")
    fig.savefig(out_path)
    plt.close(fig)


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
