import os

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def __get_results():
    print("---------- STEP 1: Getting results")
    operators = os.listdir(OPERATORS_DIR)
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
    operator_results_dir = f"{OPERATORS_DIR}/{operator}/results/data"
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
        output_dir = f"{OPERATORS_DIR}/{operator}/results/charts"

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


if __name__ == "__main__":
    main()
