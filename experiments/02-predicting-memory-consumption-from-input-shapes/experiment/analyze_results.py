import ast
import os

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out")
OPERATORS_DIR = os.getenv("OPERATORS_DIR", f"{OUTPUT_DIR}/results/operators")


def main():
    print("Analyzing results...")
    print("Using args:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  OPERATORS_DIR: {OPERATORS_DIR}")
    print()

    results = __get_results()

    __analyze_profile(results)
    __analyze_model(results)
    __analyze_data_reduction(results)
    __analyze_feature_selection(results)


def __get_results():
    print("---------- STEP 1: Getting results")
    operators = list(set(os.listdir(OPERATORS_DIR)) - {".DS_Store"})
    print(f"Found {len(operators)} operators in {OPERATORS_DIR}")
    print(f"Operators: {operators}")

    results = {}
    for operator in operators:
        results[operator] = __get_operator_results(operator)

    print("Finished getting results")
    print()

    return results


def __get_operator_results(operator: str):
    print(f"Getting results for operator {operator}")
    operator_results_dir = f"{OPERATORS_DIR}/{operator}/results"
    operator_result_paths = os.listdir(operator_results_dir)
    print(
        f"Found {len(operator_result_paths)} results for operator {operator} in {operator_results_dir}"
    )
    print(f"Result paths: {operator_result_paths}")

    return {
        result_path.split(".")[0]: pd.read_csv(
            os.path.join(operator_results_dir, result_path)
        )
        for result_path in operator_result_paths
    }


def __analyze_profile(results: dict):
    print("---------- STEP 2: Analyzing profile")

    for operator, operator_results in results.items():
        print(f"Analyzing operator {operator}")
        profile_summary = operator_results["profile_summary"]
        profile_history = operator_results["profile_history"]
        output_dir = f"{OPERATORS_DIR}/{operator}/charts"

        __analyze_peak_memory_usage_per_volume(profile_summary, operator, output_dir)
        __analyze_memory_usage_distribution(profile_history, operator, output_dir)
        __analyze_inline_xline_memory_usage_progression(
            profile_history,
            operator,
            output_dir,
        )
        __analyze_memory_usage_heatmap_by_time(
            profile_history,
            operator,
            output_dir,
        )
        __analyze_memory_usage_by_configuration(profile_history, operator, output_dir)
        __analyze_memory_usage_inlines_xlines_heatmap(
            profile_history, operator, output_dir
        )
        __analyze_memory_usage_inlines_xlines_samples_heatmap(
            profile_history, operator, output_dir
        )
        __analyze_execution_time_by_volume(profile_summary, operator, output_dir)
        __analyze_execution_time_distribution(profile_summary, operator, output_dir)
        __analyze_execution_time_distribution_by_volume(
            profile_history, operator, output_dir
        )

    print()


def __analyze_model(results: dict):
    print("---------- STEP 2: Analyzing model")

    for operator, operator_results in results.items():
        print(f"Analyzing operator {operator}")
        model_metrics = operator_results["model_metrics"]
        output_dir = f"{OPERATORS_DIR}/{operator}/charts"

        __analyze_model_performance(model_metrics, operator, output_dir)
        __analyze_model_score(model_metrics, operator, output_dir)
        __analyze_model_acc_rmse(model_metrics, operator, output_dir)
        __analyze_residual_distribution(model_metrics, operator, output_dir)
        __analyze_actual_vs_predicted(model_metrics, operator, output_dir)

    print()


def __analyze_data_reduction(results: dict):
    print("---------- STEP 3: Analyzing data reduction")

    for operator, operator_results in results.items():
        print(f"Analyzing operator {operator}")
        data_reduction = operator_results["data_reduction"]
        output_dir = f"{OPERATORS_DIR}/{operator}/charts"

        __analyze_metrics_by_sample_size(data_reduction, operator, output_dir)
        __analyze_score_by_sample_size(data_reduction, operator, output_dir)
        __analyze_rmse_mae_ratio_by_sample_size(data_reduction, operator, output_dir)
        __analyze_residual_distribution_by_sample_size(
            data_reduction,
            operator,
            output_dir,
        )


def __analyze_feature_selection(results: dict):
    print("---------- STEP 4: Analyzing feature selection")

    for operator, operator_results in results.items():
        print(f"Analyzing operator {operator}")
        feature_selection = operator_results["feature_selection"]
        output_dir = f"{OPERATORS_DIR}/{operator}/charts"

        __analyze_metrics_by_number_of_features(feature_selection, operator, output_dir)
        __analyze_score_by_number_of_features(feature_selection, operator, output_dir)
        __analyze_rmse_mae_ratio_by_number_of_features(
            feature_selection, operator, output_dir
        )
        __analyze_residual_distribution_by_number_of_features(
            feature_selection,
            operator,
            output_dir,
        )
        __analyze_feature_performance(feature_selection, operator, output_dir)


################
# CHARTS
################


def __analyze_peak_memory_usage_per_volume(
    profile_summary: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing peak memory usage per volume for operator {operator}")
    print("Using data:")
    print(profile_summary.head())

    plt.figure(figsize=(15, 7))
    ax1 = plt.gca()

    ax1.plot(
        profile_summary["volume"],
        profile_summary["peak_memory_usage_avg"],
        marker="o",
        linestyle="-",
        label="Average Memory Usage",
        color="steelblue",
    )
    ax1.fill_between(
        profile_summary["volume"],
        profile_summary["peak_memory_usage_avg"]
        - profile_summary["peak_memory_usage_std_dev"],
        profile_summary["peak_memory_usage_avg"]
        + profile_summary["peak_memory_usage_std_dev"],
        alpha=0.2,
        color="steelblue",
        label="1 Std Dev Range",
    )
    ax1.set_xlabel("Volume")
    ax1.set_ylabel("Peak Memory Usage (GB)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: (
                f"{int(x / 1e6)}M"
                if x >= 1e6
                else f"{int(x / 1e3)}K" if x >= 1e3 else str(int(x))
            )
        )
    )
    plt.xticks(rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(
        profile_summary["volume"],
        profile_summary["peak_memory_usage_cv"],
        marker="s",
        linestyle="--",
        color="red",
        label="Coefficient of Variation (CV)",
    )
    ax2.set_ylabel("Coefficient of Variation (CV)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Peak Memory Usage and Variability Per Volume")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"peak_memory_by_volume.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_memory_usage_distribution(
    profile_history: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing memory usage distribution {operator}")
    print("Using data:")
    print(profile_history.head())

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=profile_history,
        x="volume",
        y="captured_memory_usage",
        density_norm="width",
        inner="quartile",
    )

    plt.xlabel("Volume")
    plt.ylabel("Memory Usage (GB)")
    plt.title("Memory Usage Distribution Across Volume Configurations")

    ax = plt.gca()
    ax.xaxis.set_ticks(range(len(profile_history["volume"].unique())))
    ax.set_xticklabels(
        [
            f"{int(float(x) / 1e6)}M" if float(x) >= 1e6 else f"{int(float(x) / 1e3)}K"
            for x in profile_history["volume"].unique()
        ]
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"memory_usage_distribution.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_inline_xline_memory_usage_progression(
    profile_history: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing inline xline memory usage progression {operator}")
    print("Using data:")
    print(profile_history.head())

    g = sns.FacetGrid(
        profile_history,
        col="inlines",
        row="xlines",
        margin_titles=True,
        height=3,
        aspect=2,
    )
    g.map_dataframe(
        sns.lineplot, x="relative_time", y="captured_memory_usage", marker="o"
    )
    g.set_axis_labels("Relative Time (Index)", "Captured Memory Usage (GB)")
    g.set_titles(col_template="Inlines={col_name}", row_template="Xlines={row_name}")
    plt.xticks(rotation=45)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"inline_xline_memory_usage_progression.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_memory_usage_heatmap_by_time(
    profile_history: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing memory usage heatmap by time {operator}")
    print("Using data:")
    print(profile_history.head())

    profile_history = profile_history.copy()
    profile_history["relative_time_bin"] = pd.cut(
        profile_history["relative_time"], bins=50, labels=False
    )

    profile_history["volume_bin"] = pd.qcut(
        profile_history["volume"], q=10, labels=False, duplicates="drop"
    )

    table = profile_history.pivot_table(
        index="volume_bin",
        columns="relative_time_bin",
        values="captured_memory_usage",
        aggfunc="mean",
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(table, cmap="coolwarm", linewidths=0.5)
    plt.xlabel("Relative Time Binned")
    plt.ylabel("Volume Group")
    plt.title("Memory Usage Over Time (Grouped by Volume)")
    plt.xticks([])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"memory_usage_heatmap_by.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_memory_usage_by_configuration(
    profile_history: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing memory usage by configuration {operator}")
    print("Using data:")
    print(profile_history.head())

    norm = mcolors.Normalize(
        vmin=profile_history["volume"].min(), vmax=profile_history["volume"].max()
    )
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection="3d")
    legend_patches = {}

    for (sid, vol, inl, xln, smp), subset in profile_history.groupby(
        ["session_id", "volume", "inlines", "xlines", "samples"]
    ):
        color = cmap(norm(vol))
        ax.plot(
            subset["relative_time"],
            subset["volume"],
            subset["captured_memory_usage"],
            color=color,
        )
        if vol not in legend_patches:
            legend_patches[vol] = mpatches.Patch(color=color, label=f"Volume {vol}")

    ax.set_xlabel("Relative Time", labelpad=25)
    ax.set_ylabel("Volume")
    ax.set_zlabel("Memory Usage (GB)")
    plt.title("Memory Usage Over Time by Configuration")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: (
                f"{int(x / 1e6)}M"
                if x >= 1e6
                else f"{int(x / 1e3)}K" if x >= 1e3 else str(int(x))
            )
        )
    )

    plt.legend(
        handles=legend_patches.values(),
        title="Volume Groups",
        loc="upper left",
        ncol=3,
        fontsize=10,
    )

    ax.view_init(elev=20, azim=140)
    ax.set_box_aspect([2.5, 1, 1], zoom=0.95)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"memory_usage_by_configuration.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_memory_usage_inlines_xlines_heatmap(
    profile_history: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing memory usage inlines/xlines heatmap {operator}")
    print("Using data:")
    print(profile_history.head())

    df_heatmap = (
        profile_history.groupby(["inlines", "xlines"])["captured_memory_usage"]
        .max()
        .unstack()
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(df_heatmap, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)

    plt.xlabel("Xlines")
    plt.ylabel("Inlines")
    plt.title("Peak Memory Usage Heatmap")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"memory_usage_inlines_xlines_heatmap.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_memory_usage_inlines_xlines_samples_heatmap(
    profile_history: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing memory usage inlines/xlines/samples heatmap {operator}")
    print("Using data:")
    print(profile_history.head())

    df_3d = (
        profile_history.groupby(["inlines", "xlines", "samples"])[
            "captured_memory_usage"
        ]
        .max()
        .reset_index()
    )
    X = df_3d["inlines"].values
    Y = df_3d["xlines"].values
    Z = df_3d["samples"].values
    C = df_3d["captured_memory_usage"].values

    norm = plt.Normalize(C.min(), C.max())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(X, Y, Z, c=C, cmap="coolwarm", s=50)

    ax.set_xlabel("Inlines")
    ax.set_ylabel("Xlines")
    ax.set_zlabel("Samples")
    ax.set_title("3D Heatmap of Peak Memory Usage")

    cbar = fig.colorbar(sc, shrink=0.5, aspect=5)
    cbar.set_label("Memory Usage (GB)")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"memory_usage_inlines_xlines_samples_heatmap.pdf"
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_execution_time_by_volume(
    profile_summary: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing execution time by volume {operator}")
    print("Using data:")
    print(profile_summary.head())

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=profile_summary,
        x="volume",
        y="execution_time_avg",
        marker="o",
        color="steelblue",
        label="Average Execution Time",
    )
    plt.fill_between(
        profile_summary["volume"],
        profile_summary["execution_time_min"],
        profile_summary["execution_time_max"],
        color="steelblue",
        alpha=0.2,
        label="Min-Max Range",
    )

    plt.xlabel("Volume")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time by Volume with Variability")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: (
                f"{int(x / 1e6)}M"
                if x >= 1e6
                else f"{int(x / 1e3)}K" if x >= 1e3 else str(int(x))
            )
        )
    )

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"execution_time_by_volume.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_execution_time_distribution(
    profile_summary: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing execution time distribution {operator}")
    print("Using data:")
    print(profile_summary.head())

    plt.figure(figsize=(12, 6))
    sns.histplot(
        profile_summary["execution_time_avg"], bins=10, kde=True, color="darkorange"
    )

    plt.xlabel("Total Execution Time (s)")
    plt.ylabel("Frequency")
    plt.title("Execution Time Distribution Across Sessions")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"execution_time_distribution.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_execution_time_distribution_by_volume(
    profile_history: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing execution time distribution by volume {operator}")
    print("Using data:")
    print(profile_history.head())

    plt.figure(figsize=(12, 6))

    df_exec_time = (
        profile_history.groupby("session_id")
        .agg(
            total_execution_time=("timestamp", lambda x: (x.max() - x.min()) / 10**9),
            volume=("volume", "first"),
        )
        .reset_index()
    )
    df_exec_time["volume_label"] = df_exec_time["volume"].apply(
        lambda v: (
            f"{int(v / 1e6)}M"
            if v >= 1e6
            else f"{int(v / 1e3)}K" if v >= 1e3 else str(v)
        )
    )

    sns.boxplot(
        data=df_exec_time,
        hue="volume",
        x="volume_label",
        y="total_execution_time",
        palette="coolwarm",
        legend=False,
    )

    plt.xlabel("Volume")
    plt.ylabel("Total Execution Time (s)")
    plt.title("Execution Time Distribution Across Volumes")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"execution_time_distribution_by_volume.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_model_performance(
    model_metrics: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing model performance for {operator}")
    print("Using data:")
    print(model_metrics.head())

    models = model_metrics["model_name"]
    x = np.arange(len(models))

    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width * 1.5, model_metrics["rmse"], width, label="RMSE")
    ax.bar(x - width * 0.5, model_metrics["mae"], width, label="MAE")
    ax.bar(x + width * 0.5, model_metrics["r2"], width, label="R²")
    ax.bar(x + width * 1.5, model_metrics["accuracy"], width, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title("Comparison of Model Performance Metrics")
    ax.set_xlabel("Models")
    ax.set_ylabel("Metric Value")
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"performance_by_model.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_model_score(
    model_metrics: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing model score for {operator}")
    print("Using data:")
    print(model_metrics.head())

    models = model_metrics["model_name"]
    scores = model_metrics["score"]
    max_score = scores.max()

    plt.figure(figsize=(10, 5))
    plt.bar(models, scores, color="blue", alpha=0.7)

    plt.axhline(
        max_score, linestyle="--", color="red", linewidth=1.5, label="Top Score"
    )
    plt.text(
        x=len(models) - 1,
        y=max_score + 0.01 * max_score,
        s=f"Top Score: {max_score:.3f}",
        color="red",
        ha="right",
        va="top",
        fontsize=10,
    )

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Models")
    plt.ylabel("Model Score")
    plt.title("Model Ranking Based on Weighted Score")
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"score_by_model.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_model_acc_rmse(
    model_metrics: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing model accuracy by RMSE for {operator}")
    print("Using data:")
    print(model_metrics.head())

    plt.figure(figsize=(8, 5))

    for _, row in model_metrics.iterrows():
        plt.scatter(
            row["rmse"],
            row["accuracy"],
            label=row["model_name"],
            s=100,  # size of the point
        )

    plt.xlabel("RMSE (Lower is Better)")
    plt.ylabel("Accuracy (Higher is Better)")
    plt.title("Accuracy vs. RMSE for Each Model")
    plt.legend(loc="lower left", fontsize="small", bbox_to_anchor=(1.0, 0.1))
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"accuracy_by_rmse_per_model.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_residual_distribution(
    model_metrics: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing residual distribution for {operator}")
    print("Using data:")
    print(model_metrics.head())

    plt.figure(figsize=(10, 6))
    for _, row in model_metrics.iterrows():
        sns.kdeplot(
            eval(row["residuals"]), label=row["model_name"], fill=True, alpha=0.3
        )
    plt.axvline(0, color="red", linestyle="dashed")
    plt.xlabel("Residual Error")
    plt.ylabel("Density")
    plt.title("Residual Distribution Across Models")
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"residuals_distribution_by_model.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_actual_vs_predicted(
    model_metrics: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing actual vs. predicted for {operator}")
    print("Using data:")
    print(model_metrics.head())

    colors = sns.color_palette("tab10", len(model_metrics["model_name"]))

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in model_metrics.iterrows():
        sns.regplot(
            x=eval(row["y_test"]),
            y=eval(row["y_pred"]),
            ax=axes[i],
            scatter_kws={"alpha": 0.5, "color": colors[i]},
            line_kws={"color": colors[i]},
        )

        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")
        axes[i].set_title(f"{row["model_name"]}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"actual_vs_predicted_by_model.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_metrics_by_sample_size(
    data_reduction: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing metrics by sample size for {operator}")
    print("Using data:")
    print(data_reduction.head())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_ranges = {
        "RMSE": (0, max(data_reduction["rmse"]) * 1.1),
        "MAE": (0, max(data_reduction["mae"]) * 1.1),
        "R²": (0, 1.2),
        "Accuracy": (0, 1.2),
    }
    metrics = {
        "RMSE": "rmse",
        "MAE": "mae",
        "R²": "r2",
        "Accuracy": "accuracy",
    }

    for ax, (metric, col) in zip(axes.flatten(), metrics.items()):
        ax.plot(
            data_reduction["num_samples"],
            data_reduction[col],
            marker="o",
            linestyle="-",
        )
        ax.set_title(f"{metric} Evolution")
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel(metric)
        ax.set_ylim(metric_ranges[metric])

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metrics_evolution_by_sample_size.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_score_by_sample_size(
    data_reduction: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing score by sample size for {operator}")
    print("Using data:")
    print(data_reduction.head())

    plt.figure(figsize=(8, 5))
    plt.plot(
        data_reduction["num_samples"],
        data_reduction["score"],
        marker="o",
        linestyle="-",
    )
    plt.title("Model Score vs. Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel("Model Score")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"score_by_sample_size.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_rmse_mae_ratio_by_sample_size(
    data_reduction: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing RMSE/MAE ratio by sample size for {operator}")
    print("Using data:")
    print(data_reduction.head())

    plt.figure(figsize=(8, 5))
    plt.plot(
        data_reduction["num_samples"],
        data_reduction["rmse"] / data_reduction["mae"],
        marker="o",
        linestyle="-",
    )
    plt.title("RMSE/MAE Ratio Over Data Reduction")
    plt.xlabel("Number of Samples")
    plt.ylabel("RMSE / MAE Ratio")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rmse_mae_ratio_by_sample_size.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_residual_distribution_by_sample_size(
    data_reduction: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing residual distribution by sample size for {operator}")
    print("Using data:")
    print(data_reduction.head())

    df = data_reduction.copy()
    df["residuals"] = df["residuals"].apply(lambda x: eval(x))
    df["mean"] = df["residuals"].apply(np.mean)
    df["std"] = df["residuals"].apply(np.std)
    df["max"] = df["residuals"].apply(np.max)
    df["min"] = df["residuals"].apply(np.min)
    df["median"] = df["residuals"].apply(np.median)
    df["mae"] = df["residuals"].apply(lambda x: np.mean(np.abs(x)))
    df["rmse"] = df["residuals"].apply(lambda x: np.sqrt(np.mean(np.square(x))))

    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 6))
    plt.plot(df["num_samples"], df["mae"], marker="o", label="MAE")
    plt.plot(df["num_samples"], df["rmse"], marker="s", label="RMSE")

    plt.fill_between(
        df["num_samples"],
        df["mae"] - df["std"],
        df["mae"] + df["std"],
        color="blue",
        alpha=0.1,
        label="MAE ± STD",
    )

    plt.title(f"Error Metrics vs Dataset Size\nOperator: {operator}", fontsize=16)
    plt.xlabel("Number of Samples", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"residual_metrics_by_sample_size.pdf",
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_metrics_by_number_of_features(
    feature_selection: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing metrics by number of features for {operator}")
    print("Using data:")
    print(feature_selection.head())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_ranges = {
        "RMSE": (0, max(feature_selection["rmse"]) * 1.1),
        "MAE": (0, max(feature_selection["mae"]) * 1.1),
        "R²": (0, 1.2),
        "Accuracy": (0, 1.2),
    }
    metrics = {
        "RMSE": "rmse",
        "MAE": "mae",
        "R²": "r2",
        "Accuracy": "accuracy",
    }

    for ax, (metric, col) in zip(axes.flatten(), metrics.items()):
        ax.plot(
            feature_selection["num_features"],
            feature_selection[col],
            marker="o",
            linestyle="-",
        )
        ax.set_title(f"{metric} Evolution")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(metric)
        ax.set_ylim(metric_ranges[metric])

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"metrics_evolution_by_number_of_features.pdf"
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_score_by_number_of_features(
    feature_selection: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing score by number of features for {operator}")
    print("Using data:")
    print(feature_selection.head())

    plt.figure(figsize=(8, 5))
    plt.plot(
        feature_selection["num_features"],
        feature_selection["score"],
        marker="o",
        linestyle="-",
    )
    plt.title("Model Score vs. Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Model Score")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"score_by_number_of_features.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_rmse_mae_ratio_by_number_of_features(
    feature_selection: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing RMSE/MAE ratio by number of features for {operator}")
    print("Using data:")
    print(feature_selection.head())

    plt.figure(figsize=(8, 5))
    plt.plot(
        feature_selection["num_features"],
        feature_selection["rmse"] / feature_selection["mae"],
        marker="o",
        linestyle="-",
    )
    plt.title("RMSE/MAE Ratio Over Data Reduction")
    plt.xlabel("Number of Features")
    plt.ylabel("RMSE / MAE Ratio")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rmse_mae_ratio_by_number_of_features.pdf")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_residual_distribution_by_number_of_features(
    feature_selection: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing residual distribution by number of features for {operator}")
    print("Using data:")
    print(feature_selection.head())

    df = feature_selection.copy()
    df["residuals"] = df["residuals"].apply(lambda x: eval(x))
    df["mean"] = df["residuals"].apply(np.mean)
    df["std"] = df["residuals"].apply(np.std)
    df["max"] = df["residuals"].apply(np.max)
    df["min"] = df["residuals"].apply(np.min)
    df["median"] = df["residuals"].apply(np.median)
    df["mae"] = df["residuals"].apply(lambda x: np.mean(np.abs(x)))
    df["rmse"] = df["residuals"].apply(lambda x: np.sqrt(np.mean(np.square(x))))

    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 6))
    plt.plot(df["num_features"], df["mae"], marker="o", label="MAE")
    plt.plot(df["num_features"], df["rmse"], marker="s", label="RMSE")

    plt.fill_between(
        df["num_features"],
        df["mae"] - df["std"],
        df["mae"] + df["std"],
        color="blue",
        alpha=0.1,
        label="MAE ± STD",
    )

    plt.title(f"Error Metrics vs Dataset Size\nOperator: {operator}", fontsize=16)
    plt.xlabel("Number of Features", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"residual_metrics_by_number_of_features.pdf",
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


def __analyze_feature_performance(
    feature_selection: pd.DataFrame,
    operator: str,
    output_dir: str,
):
    print(f"Analyzing feature performance for {operator}")
    print("Using data:")
    print(feature_selection.head())

    df = feature_selection.copy()
    df["selected_features"] = df["selected_features"].apply(ast.literal_eval)
    df_sorted = df.sort_values(by="num_features", ascending=False).reset_index(
        drop=True
    )

    impact_records = []

    for i in range(len(df_sorted) - 1):
        current_features = set(df_sorted.loc[i, "selected_features"])
        next_features = set(df_sorted.loc[i + 1, "selected_features"])

        removed = current_features - next_features
        if len(removed) != 1:
            continue
        removed_feature = list(removed)[0]

        delta_rmse = df_sorted.loc[i + 1, "rmse"] - df_sorted.loc[i, "rmse"]
        delta_mae = df_sorted.loc[i + 1, "mae"] - df_sorted.loc[i, "mae"]
        delta_r2 = df_sorted.loc[i + 1, "r2"] - df_sorted.loc[i, "r2"]

        impact_records.append(
            {
                "removed_feature": removed_feature,
                "delta_rmse": delta_rmse,
                "delta_mae": delta_mae,
                "delta_r2": delta_r2,
                "from_features": len(current_features),
                "to_features": len(next_features),
            }
        )

    impact_df = pd.DataFrame(impact_records)

    avg_impact = (
        impact_df.groupby("removed_feature")[["delta_rmse"]]
        .mean()
        .sort_values("delta_rmse", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_impact.index, y="delta_rmse", data=avg_impact)
    plt.xticks(rotation=90)
    plt.ylabel("Average ΔRMSE")
    plt.xlabel("Removed Feature")
    plt.title("Impact of Removing Each Feature on RMSE")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"feature_impact.pdf",
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
