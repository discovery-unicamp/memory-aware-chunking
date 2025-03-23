import ast
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
            profile_history, operator, output_dir
        )
        __analyze_memory_usage_heatmap_by_time(profile_history, operator, output_dir)
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
            data_reduction, operator, output_dir
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
            feature_selection, operator, output_dir
        )
        __analyze_feature_performance(feature_selection, operator, output_dir)


def __analyze_peak_memory_usage_per_volume(
    profile_summary: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing peak memory usage per volume for operator {operator}")
    print("Using data:")
    print(profile_summary.head())

    fig, ax1 = plt.subplots()

    ax1.plot(
        profile_summary["volume"],
        profile_summary["peak_memory_usage_avg"],
        marker="o",
        zorder=3,
        label="Average Memory Usage",
    )

    ax1.fill_between(
        profile_summary["volume"],
        profile_summary["peak_memory_usage_avg"]
        - profile_summary["peak_memory_usage_std_dev"],
        profile_summary["peak_memory_usage_avg"]
        + profile_summary["peak_memory_usage_std_dev"],
        alpha=0.2,
        zorder=2,
    )

    ax1.set_xlabel("Volume")
    ax1.set_ylabel("Peak Memory Usage (GB)")

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
        zorder=3,
        label="Coefficient of Variation (CV)",
        color="tab:orange",
    )

    ax2.set_ylabel("Coefficient of Variation (CV)")

    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax2.yaxis.grid(False)

    plt.title("Peak Memory Usage and Variability Per Volume")

    out = os.path.join(output_dir, f"peak_memory_by_volume.pdf")
    __save_chart(out)


def __analyze_memory_usage_distribution(
    profile_history: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing memory usage distribution {operator}")
    print("Using data:")
    print(profile_history.head())

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(
        data=profile_history,
        x="volume",
        y="captured_memory_usage",
        inner="quartile",
        cut=0,
        zorder=3,
        scale="width",
        bw_adjust=0.8,
        ax=ax,
    )

    ax.set_xlabel("Volume")
    ax.set_ylabel("Memory Usage (GB)")
    ax.set_title("Memory Usage Distribution Across Volume Configurations")

    ax.xaxis.set_ticks(range(len(profile_history["volume"].unique())))
    ax.set_xticklabels(
        [
            f"{int(float(x) / 1e6)}M" if float(x) >= 1e6 else f"{int(float(x) / 1e3)}K"
            for x in profile_history["volume"].unique()
        ]
    )

    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="both")
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"memory_usage_distribution.pdf")
    __save_chart(out)


def __analyze_inline_xline_memory_usage_progression(
    profile_history: pd.DataFrame, operator: str, output_dir: str
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
        despine=False,
    )

    g.map_dataframe(
        sns.lineplot, x="relative_time", y="captured_memory_usage", marker="o", zorder=3
    )
    g.set_axis_labels("Relative Time (Index)", "Captured Memory Usage (GB)")
    g.set_titles(col_template="Inlines={col_name}", row_template="Xlines={row_name}")

    for ax in g.axes.flatten():
        sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
        ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"inline_xline_memory_usage_progression.pdf")
    __save_chart(out)


def __analyze_memory_usage_heatmap_by_time(
    profile_history: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing memory usage heatmap by time {operator}")
    print("Using data:")
    print(profile_history.head())

    ph = profile_history.copy()
    ph["relative_time_bin"] = pd.cut(ph["relative_time"], bins=50, labels=False)
    ph["volume_bin"] = pd.qcut(ph["volume"], q=10, labels=False, duplicates="drop")

    table = ph.pivot_table(
        index="volume_bin",
        columns="relative_time_bin",
        values="captured_memory_usage",
        aggfunc="mean",
    )

    fig, ax = plt.subplots()

    sns.heatmap(table, cmap="viridis", ax=ax, zorder=1)
    plt.xlabel("Relative Time Binned")
    plt.ylabel("Volume Group")
    plt.title("Memory Usage Over Time (Grouped by Volume)")

    ax.grid(True, which="both", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
    sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"memory_usage_heatmap_by_time.pdf")
    __save_chart(out)


def __analyze_memory_usage_by_configuration(
    profile_history: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing memory usage by configuration {operator}")
    print("Using data:")
    print(profile_history.head())

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for _, subset in profile_history.groupby(
        ["session_id", "volume", "inlines", "xlines", "samples"]
    ):
        ax.plot(
            subset["relative_time"],
            subset["volume"],
            subset["captured_memory_usage"],
            zorder=3,
        )

    ax.set_xlabel("Relative Time")
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

    ax.grid(True)
    ax.set_axisbelow(True)
    ax.view_init(elev=20, azim=140)

    out = os.path.join(output_dir, f"memory_usage_by_configuration.pdf")
    __save_chart(out)


def __analyze_memory_usage_inlines_xlines_heatmap(
    profile_history: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing memory usage inlines/xlines heatmap {operator}")
    print("Using data:")
    print(profile_history.head())

    df_heatmap = (
        profile_history.groupby(["inlines", "xlines"])["captured_memory_usage"]
        .max()
        .unstack()
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        df_heatmap,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        zorder=1,
    )
    plt.xlabel("Xlines")
    plt.ylabel("Inlines")
    plt.title("Peak Memory Usage Heatmap")

    for spine in ax.spines.values():
        spine.set_visible(True)
    sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)
    ax.set_axisbelow(True)
    ax.grid(False)

    out = os.path.join(output_dir, f"memory_usage_inlines_xlines_heatmap.pdf")
    __save_chart(out)


def __analyze_memory_usage_inlines_xlines_samples_heatmap(
    profile_history: pd.DataFrame, operator: str, output_dir: str
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

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    sc = ax.scatter(X, Y, Z, c=C, cmap="viridis", s=50, zorder=3)

    ax.set_xlabel("Inlines")
    ax.set_ylabel("Xlines")
    ax.set_zlabel("Samples")
    ax.set_title("3D Heatmap of Peak Memory Usage")

    cbar = fig.colorbar(sc, shrink=0.5, aspect=5)
    cbar.set_label("Memory Usage (GB)")

    ax.grid(True)
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"memory_usage_inlines_xlines_samples_heatmap.pdf")
    __save_chart(out)


def __analyze_execution_time_by_volume(
    profile_summary: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing execution time by volume {operator}")
    print("Using data:")
    print(profile_summary.head())

    fig, ax = plt.subplots()

    ax.plot(
        profile_summary["volume"],
        profile_summary["execution_time_avg"],
        marker="o",
        zorder=3,
        label="Average Execution Time",
    )

    ax.fill_between(
        profile_summary["volume"],
        profile_summary["execution_time_min"],
        profile_summary["execution_time_max"],
        alpha=0.2,
        zorder=2,
    )

    ax.set_xlabel("Volume")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Execution Time by Volume with Variability")

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
    ax.set_axisbelow(True)
    plt.legend()

    out = os.path.join(output_dir, f"execution_time_by_volume.pdf")
    __save_chart(out)


def __analyze_execution_time_distribution(
    profile_summary: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing execution time distribution {operator}")
    print("Using data:")
    print(profile_summary.head())

    fig, ax = plt.subplots()

    sns.histplot(
        profile_summary["execution_time_avg"], bins=10, kde=True, ax=ax, zorder=3
    )

    ax.set_xlabel("Total Execution Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Execution Time Distribution Across Sessions")
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"execution_time_distribution.pdf")
    __save_chart(out)


def __analyze_execution_time_distribution_by_volume(
    profile_history: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing execution time distribution by volume {operator}")
    print("Using data:")
    print(profile_history.head())

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

    fig, ax = plt.subplots()

    sns.boxplot(
        data=df_exec_time,
        hue="volume",
        x="volume_label",
        y="total_execution_time",
        ax=ax,
        zorder=3,
    )

    ax.set_xlabel("Volume")
    ax.set_ylabel("Total Execution Time (s)")
    ax.set_title("Execution Time Distribution Across Volumes")
    ax.set_axisbelow(True)

    plt.xticks(rotation=45)
    plt.grid(True, axis="both")

    out = os.path.join(output_dir, f"execution_time_distribution_by_volume.pdf")
    __save_chart(out)


def __analyze_model_performance(
    model_metrics: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing model performance for {operator}")
    print("Using data:")
    print(model_metrics.head())

    models = model_metrics["model_name"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots()

    ax.bar(x - width * 1.5, model_metrics["rmse"], width, label="RMSE", zorder=3)
    ax.bar(x - width * 0.5, model_metrics["mae"], width, label="MAE", zorder=3)
    ax.bar(x + width * 0.5, model_metrics["r2"], width, label="R²", zorder=3)
    ax.bar(
        x + width * 1.5, model_metrics["accuracy"], width, label="Accuracy", zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title("Comparison of Model Performance Metrics")
    ax.set_xlabel("Models")
    ax.set_ylabel("Metric Value")

    ax.set_axisbelow(True)
    ax.legend()

    out = os.path.join(output_dir, f"performance_by_model.pdf")
    __save_chart(out)


def __analyze_model_score(model_metrics: pd.DataFrame, operator: str, output_dir: str):
    print(f"Analyzing model score for {operator}")
    print("Using data:")
    print(model_metrics.head())

    models = model_metrics["model_name"]
    scores = model_metrics["score"]
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

    ax.set_xlabel("Models")
    ax.set_ylabel("Model Score")
    ax.set_title("Model Ranking Based on Weighted Score")

    ax.set_axisbelow(True)
    ax.legend()

    plt.xticks(rotation=45, ha="right")

    out = os.path.join(output_dir, f"score_by_model.pdf")
    __save_chart(out)


def __analyze_model_acc_rmse(
    model_metrics: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing model accuracy by RMSE for {operator}")
    print("Using data:")
    print(model_metrics.head())

    plt.figure()

    for _, row in model_metrics.iterrows():
        plt.scatter(
            row["rmse"],
            row["accuracy"],
            label=row["model_name"],
            s=100,
            zorder=3,
        )

    plt.xlabel("RMSE (Lower is Better)")
    plt.ylabel("Accuracy (Higher is Better)")
    plt.title("Accuracy vs. RMSE for Each Model")
    plt.legend(loc="lower left", fontsize="small", bbox_to_anchor=(1.0, 0.1))

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"accuracy_by_rmse_per_model.pdf")
    __save_chart(out)


def __analyze_residual_distribution(
    model_metrics: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing residual distribution for {operator}")
    print("Using data:")
    print(model_metrics.head())

    plt.figure()

    for _, row in model_metrics.iterrows():
        sns.kdeplot(
            eval(row["residuals"]),
            label=row["model_name"],
            fill=True,
            alpha=0.3,
            zorder=3,
        )

    plt.axvline(0, linestyle="dashed", zorder=4)
    plt.xlabel("Residual Error")
    plt.ylabel("Density")
    plt.title("Residual Distribution Across Models")
    plt.legend()

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"residuals_distribution_by_model.pdf")
    __save_chart(out)


def __analyze_actual_vs_predicted(
    model_metrics: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing actual vs. predicted for {operator}")
    print("Using data:")
    print(model_metrics.head())

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    for i, row in model_metrics.iterrows():
        sns.regplot(
            x=eval(row["y_test"]),
            y=eval(row["y_pred"]),
            ax=axes[i],
            scatter_kws={"zorder": 3},
            line_kws={"zorder": 4},
        )
        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")
        axes[i].set_title(f"{row['model_name']}")
        axes[i].set_axisbelow(True)

    out = os.path.join(output_dir, f"actual_vs_predicted_by_model.pdf")
    __save_chart(out)


def __analyze_metrics_by_sample_size(
    data_reduction: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing metrics by sample size for {operator}")
    print("Using data:")
    print(data_reduction.head())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

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
            zorder=3,
        )
        ax.set_title(f"{metric} Evolution")
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel(metric)
        ax.set_ylim(metric_ranges[metric])
        ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"metrics_evolution_by_sample_size.pdf")
    __save_chart(out)


def __analyze_score_by_sample_size(
    data_reduction: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing score by sample size for {operator}")
    print("Using data:")
    print(data_reduction.head())

    plt.figure()

    plt.plot(
        data_reduction["num_samples"],
        data_reduction["score"],
        marker="o",
        zorder=3,
    )

    plt.title("Model Score vs. Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel("Model Score")

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"score_by_sample_size.pdf")
    __save_chart(out)


def __analyze_rmse_mae_ratio_by_sample_size(
    data_reduction: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing RMSE/MAE ratio by sample size for {operator}")
    print("Using data:")
    print(data_reduction.head())

    plt.figure()

    plt.plot(
        data_reduction["num_samples"],
        data_reduction["rmse"] / data_reduction["mae"],
        marker="o",
        zorder=3,
    )

    plt.title("RMSE/MAE Ratio Over Data Reduction")
    plt.xlabel("Number of Samples")
    plt.ylabel("RMSE / MAE Ratio")

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"rmse_mae_ratio_by_sample_size.pdf")
    __save_chart(out)


def __analyze_residual_distribution_by_sample_size(
    data_reduction: pd.DataFrame, operator: str, output_dir: str
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

    plt.figure()

    plt.plot(df["num_samples"], df["mae"], marker="o", label="MAE", zorder=3)
    plt.plot(df["num_samples"], df["rmse"], marker="s", label="RMSE", zorder=3)

    plt.fill_between(
        df["num_samples"],
        df["mae"] - df["std"],
        df["mae"] + df["std"],
        alpha=0.2,
        zorder=2,
    )

    plt.title(f"Error Metrics vs Dataset Size\nOperator: {operator}")
    plt.xlabel("Number of Samples")
    plt.ylabel("Error")
    plt.xticks(rotation=45)
    plt.legend()

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"residual_metrics_by_sample_size.pdf")
    __save_chart(out)


def __analyze_metrics_by_number_of_features(
    feature_selection: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing metrics by number of features for {operator}")
    print("Using data:")
    print(feature_selection.head())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

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
            zorder=3,
        )
        ax.set_title(f"{metric} Evolution")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(metric)
        ax.set_ylim(metric_ranges[metric])
        ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"metrics_evolution_by_number_of_features.pdf")
    __save_chart(out)


def __analyze_score_by_number_of_features(
    feature_selection: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing score by number of features for {operator}")
    print("Using data:")
    print(feature_selection.head())

    plt.figure()

    plt.plot(
        feature_selection["num_features"],
        feature_selection["score"],
        marker="o",
        zorder=3,
    )

    plt.title("Model Score vs. Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Model Score")

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"score_by_number_of_features.pdf")
    __save_chart(out)


def __analyze_rmse_mae_ratio_by_number_of_features(
    feature_selection: pd.DataFrame, operator: str, output_dir: str
):
    print(f"Analyzing RMSE/MAE ratio by number of features for {operator}")
    print("Using data:")
    print(feature_selection.head())

    plt.figure()

    plt.plot(
        feature_selection["num_features"],
        feature_selection["rmse"] / feature_selection["mae"],
        marker="o",
        zorder=3,
    )

    plt.title("RMSE/MAE Ratio Over Data Reduction")
    plt.xlabel("Number of Features")
    plt.ylabel("RMSE / MAE Ratio")

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"rmse_mae_ratio_by_number_of_features.pdf")
    __save_chart(out)


def __analyze_residual_distribution_by_number_of_features(
    feature_selection: pd.DataFrame, operator: str, output_dir: str
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

    plt.figure()

    plt.plot(df["num_features"], df["mae"], marker="o", label="MAE", zorder=3)
    plt.plot(df["num_features"], df["rmse"], marker="s", label="RMSE", zorder=3)

    plt.fill_between(
        df["num_features"],
        df["mae"] - df["std"],
        df["mae"] + df["std"],
        alpha=0.2,
        zorder=2,
    )

    plt.title(f"Error Metrics vs Dataset Size\nOperator: {operator}")
    plt.xlabel("Number of Features")
    plt.ylabel("Error")
    plt.xticks(rotation=45)
    plt.legend()

    ax = plt.gca()
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"residual_metrics_by_number_of_features.pdf")
    __save_chart(out)


def __analyze_feature_performance(
    feature_selection: pd.DataFrame, operator: str, output_dir: str
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

    fig, ax = plt.subplots()

    sns.barplot(x=avg_impact.index, y="delta_rmse", data=avg_impact, ax=ax, zorder=3)

    plt.xticks(rotation=90)
    plt.ylabel("Average ΔRMSE")
    plt.xlabel("Removed Feature")
    plt.title("Impact of Removing Each Feature on RMSE")
    plt.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)

    out = os.path.join(output_dir, f"feature_impact.pdf")
    __save_chart(out)


def __save_chart(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
