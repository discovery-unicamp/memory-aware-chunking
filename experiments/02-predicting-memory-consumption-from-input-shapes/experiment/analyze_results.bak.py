import ast
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(
    context="paper",
    style="whitegrid",
)
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

CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def main():
    print("Analyzing results...")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"OPERATORS_DIR: {OPERATORS_DIR}")
    print()

    # 1. Load all operator data
    results = load_all_operators(OPERATORS_DIR)
    if not results:
        print("No valid operator data found. Exiting.")
        return

    # 2. Build combined DataFrames for each type of data
    profile_summaries = []
    profile_histories = []
    model_metrics = []
    data_reductions = []
    feature_selections = []

    for operator, dfs in results.items():
        if "profile_summary" in dfs:
            tmp = dfs["profile_summary"].copy()
            tmp["operator"] = operator
            profile_summaries.append(tmp)

        if "profile_history" in dfs:
            tmp = dfs["profile_history"].copy()
            tmp["operator"] = operator
            profile_histories.append(tmp)

        if "model_metrics" in dfs:
            tmp = dfs["model_metrics"].copy()
            tmp["operator"] = operator
            model_metrics.append(tmp)

        if "data_reduction" in dfs:
            tmp = dfs["data_reduction"].copy()
            tmp["operator"] = operator
            data_reductions.append(tmp)

        if "feature_selection" in dfs:
            tmp = dfs["feature_selection"].copy()
            tmp["operator"] = operator
            feature_selections.append(tmp)

    # Convert to single combined DataFrame (where applicable)
    profile_summaries = (
        pd.concat(profile_summaries, ignore_index=True) if profile_summaries else None
    )
    profile_histories = (
        pd.concat(profile_histories, ignore_index=True) if profile_histories else None
    )
    model_metrics = (
        pd.concat(model_metrics, ignore_index=True) if model_metrics else None
    )
    data_reductions = (
        pd.concat(data_reductions, ignore_index=True) if data_reductions else None
    )
    feature_selections = (
        pd.concat(feature_selections, ignore_index=True) if feature_selections else None
    )

    # 3. Generate charts according to your desired layouts
    # ------------------------------------------------------------------
    # Example: "accuracy_by_rmse_per_model.pdf" (single chart with all data)
    # ------------------------------------------------------------------
    if model_metrics is not None:
        plot_accuracy_by_rmse_per_model(model_metrics)

        # "actual_vs_predicted_by_model.pdf"
        # each model on its own chart, multiple lines (colors) for each operator
        plot_actual_vs_predicted_by_model(model_metrics)

    # ------------------------------------------------------------------
    # "execution_time_by_volume.pdf"
    # multiple lines with variance, one per operator
    # "execution_time_distribution_by_volume.pdf",
    # "execution_time_distribution.pdf",
    # "execution_time_vs_memory.pdf"
    # (these can be side by side or separate)
    # ------------------------------------------------------------------
    if profile_summaries is not None:
        plot_execution_time_by_volume(profile_summaries)
        plot_execution_time_distribution(profile_summaries)
        plot_execution_time_vs_memory(profile_summaries)

    # ------------------------------------------------------------------
    # "feature_impact.pdf" => multiple bars or side by side
    # ------------------------------------------------------------------
    if feature_selections is not None:
        plot_feature_impact(feature_selections)

    # ------------------------------------------------------------------
    # "inline_xline_memory_usage_progression.pdf"
    # single chart with multiple lines for each operator
    # (possible large scale differences => maybe log scale or separate subplots)
    # ------------------------------------------------------------------
    if profile_histories is not None:
        plot_memory_progression_per_operator(profile_histories)

    # ------------------------------------------------------------------
    # Memory usage plots:
    #   memory_usage_by_configuration.pdf
    #   memory_usage_distribution.pdf
    #   memory_usage_heatmap_by_time.pdf
    #   memory_usage_inlines_xlines_heatmap.pdf
    #   memory_usage_inlines_xlines_samples_heatmap.pdf
    #   memory_usage_distribution.pdf (already in the list)
    #   memory_usage_by_configuration.pdf
    #   peak_memory_by_volume.pdf
    # ------------------------------------------------------------------
    if profile_summaries is not None:
        plot_peak_memory_by_volume(profile_summaries)

    if profile_histories is not None:
        plot_memory_usage_distribution(profile_histories)
        plot_memory_usage_heatmap_by_time(profile_histories)
        plot_memory_usage_by_configuration(profile_histories)
        plot_inlines_xlines_heatmap(profile_histories)
        plot_inlines_xlines_samples_3d(profile_histories)

    # ------------------------------------------------------------------
    # "memory_vs_dimensions_pairplot.pdf" => one per operator
    # but here we can place operator name on the file
    # or create a combined pairplot with operator color/hue
    # for clarity, this code produces separate pairplots per operator
    # ------------------------------------------------------------------
    if profile_histories is not None:
        plot_memory_vs_dimensions_pairplot(profile_histories)

    # ------------------------------------------------------------------
    # "memory_vs_volume_regression.pdf"
    # multiple lines on the same chart or color by operator
    # ------------------------------------------------------------------
    if profile_summaries is not None:
        plot_memory_vs_volume_regression(profile_summaries)

    # ------------------------------------------------------------------
    # Additional model metrics
    #   metrics_evolution_by_number_of_features.pdf
    #   metrics_evolution_by_sample_size.pdf
    #   peak_memory_by_volume.pdf (handled above as "peak_memory_by_volume.pdf")
    #   performance_by_model.pdf => side by side?
    #   residual_metrics_by_number_of_features.pdf
    #   residual_metrics_by_sample_size.pdf
    #   residual_qq_plots.pdf => multiple lines
    #   residual_vs_predicted.pdf => multiple dots with different colors
    #   residuals_distribution_by_model.pdf => side by side
    #   rmse_mae_ratio_by_number_of_features.pdf,
    #   rmse_mae_ratio_by_sample_size.pdf => multiple lines
    #   score_by_model.pdf => multiple bars with multiple lines
    #   score_by_number_of_features.pdf => multiple lines
    #   score_by_sample_size.pdf => multiple lines
    # ------------------------------------------------------------------

    # These next group of charts are basically from model_metrics,
    # data_reduction, and feature_selection.

    if model_metrics is not None:
        plot_performance_by_model(model_metrics)
        plot_score_by_model(model_metrics)
        plot_residuals_distribution_by_model(model_metrics)
        plot_residual_vs_predicted(model_metrics)  # also in additional
        plot_residual_qq_plots(model_metrics)

    if data_reductions is not None:
        plot_metrics_evolution_by_sample_size(data_reductions)
        plot_score_by_sample_size(data_reductions)
        plot_rmse_mae_ratio_by_sample_size(data_reductions)
        plot_residual_metrics_by_sample_size(data_reductions)

    if feature_selections is not None:
        plot_metrics_evolution_by_number_of_features(feature_selections)
        plot_score_by_number_of_features(feature_selections)
        plot_rmse_mae_ratio_by_number_of_features(feature_selections)
        plot_residual_metrics_by_number_of_features(feature_selections)

    print("All charts generated in:", CHARTS_DIR)


def load_all_operators(operators_dir):
    """Loads CSV files for each operator directory.
    Returns a dict {operator_name: {csv_key: DataFrame, ...}, ...}.
    """
    if not os.path.isdir(operators_dir):
        print(f"Operators directory not found: {operators_dir}")
        return {}

    operators = [op for op in os.listdir(operators_dir) if not op.startswith(".")]
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
# Chart Functions
# ------------------------------------------------------------------------------


def plot_accuracy_by_rmse_per_model(df):
    """Plots a single scatter chart for all operators/models:
    x=RMSE, y=Accuracy, color/hue=operator."""
    out_path = os.path.join(CHARTS_DIR, "accuracy_by_rmse_per_model.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="rmse",
        y="accuracy",
        hue="operator",
        style="model_name",
        s=100,
        ax=ax,
    )
    ax.set_xlabel("RMSE (Lower is Better)")
    ax.set_ylabel("Accuracy (Higher is Better)")
    ax.set_title("Accuracy vs. RMSE (All Operators)")
    handles, labels = ax.get_legend_handles_labels()

    op_count = df["operator"].nunique()
    model_count = df["model_name"].nunique()

    # Operator legend
    legend1 = ax.legend(
        handles[1 : op_count + 1],
        labels[1 : op_count + 1],
        title="Operator",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax.add_artist(legend1)

    # Model legend
    ax.legend(
        handles[op_count + 2 : op_count + model_count],
        labels[op_count + 2 : op_count + model_count],
        title="Model",
        bbox_to_anchor=(1.05, 0),
        loc="lower left",
    )

    save_chart(fig, out_path)


def plot_actual_vs_predicted_by_model(df):
    """
    Draws a grid of subplots, one per model.
    Displays actual (dashed) and predicted (solid) lines, color-coded by operator.
    Only the last row shows the x-axis labels.
    Produces two legends in the top-right corner, side by side: one for operators and one for line style.
    Increases the suptitle font size.
    """

    out_path = os.path.join(CHARTS_DIR, "actual_vs_predicted_by_model.pdf")
    models = df["model_name"].unique()
    operators = df["operator"].unique()
    num_models = len(models)

    # Define colors for each operator
    color_map = dict(zip(operators, sns.color_palette(n_colors=len(operators))))

    # Figure layout
    cols = 3
    rows = int(np.ceil(num_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        subset = df[df["model_name"] == model]

        for op in subset["operator"].unique():
            sub_op = subset[subset["operator"] == op]
            for _, row in sub_op.iterrows():
                y_test = np.array(ast.literal_eval(row["y_test"]))
                y_pred = np.array(ast.literal_eval(row["y_pred"]))

                # Actual => dashed
                ax.plot(
                    y_test, linestyle="--", color=color_map[op], label=f"{op} (actual)"
                )
                # Predicted => solid
                ax.plot(
                    y_pred,
                    linestyle="-",
                    color=color_map[op],
                    label=f"{op} (predicted)",
                )

        ax.set_title(model)

        # Show x-axis only on the last row
        current_row = i // cols
        if current_row == rows - 1:
            ax.set_xlabel("Sample Index")
        else:
            ax.set_xlabel("")

        ax.set_ylabel("Value")

    # Remove any local legends
    for ax in axes:
        ax.legend_ = None

    # Build operator handles (color-based)
    operator_handles = []
    for op in operators:
        operator_handles.append(
            Line2D([], [], color=color_map[op], marker="o", linestyle="None", label=op)
        )

    # Build style handles (actual vs predicted)
    style_actual = Line2D([], [], color="black", linestyle="--", label="Actual")
    style_pred = Line2D([], [], color="black", linestyle="-", label="Predicted")

    # Place two legends in the top-right corner, side by side
    # Adjust 'bbox_to_anchor' to shift them slightly if needed
    legend_ops = fig.legend(
        handles=operator_handles,
        labels=[op for op in operators],
        title="Operators",
        loc="upper right",
        bbox_to_anchor=(0.69, 0.98),
    )
    fig.add_artist(legend_ops)

    legend_style = fig.legend(
        handles=[style_actual, style_pred],
        labels=["Actual", "Predicted"],
        title="Type of Result",
        loc="upper right",
        bbox_to_anchor=(0.79, 0.98),
    )
    fig.add_artist(legend_style)

    # Adjust subplots so legends do not cover subplots
    fig.subplots_adjust(top=0.86, right=0.78)

    # Increase the suptitle font size
    fig.suptitle(
        "Actual vs. Predicted by Model (All Operators)",
        fontsize=21,
        ha="right",
        y="0.95",
    )

    # Save with bbox_inches="tight" so legends are included
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_execution_time_by_volume(df):
    """
    Plots multiple lines for each operator, x=volume, y=execution_time_avg,
    with fill for min-max or std dev if available.
    """
    out_path = os.path.join(CHARTS_DIR, "execution_time_by_volume.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()

    for op in df["operator"].unique():
        sub = df[df["operator"] == op].dropna(subset=["volume", "execution_time_avg"])
        sub = sub.sort_values("volume")
        ax.plot(sub["volume"], sub["execution_time_avg"], marker="o", label=op)
        # Attempt a std dev fill if columns exist
        if {"execution_time_min", "execution_time_max"}.issubset(sub.columns):
            ax.fill_between(
                sub["volume"],
                sub["execution_time_min"],
                sub["execution_time_max"],
                alpha=0.2,
            )

    ax.set_xlabel("Volume")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Execution Time by Volume (All Opertors)")
    ax.legend()

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )

    save_chart(fig, out_path)


def plot_execution_time_distribution(df):
    """
    Creates a distribution (hist or box) for execution_time_avg across all operators.
    """
    out_path = os.path.join(CHARTS_DIR, "execution_time_distribution.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="operator", y="execution_time_avg", ax=ax)
    ax.set_title("Execution Time Distribution (All Operators)")
    ax.set_xlabel("Operator")
    ax.set_ylabel("Execution Time (s)")
    save_chart(fig, out_path)


def plot_execution_time_vs_memory(df):
    """
    Scatter of execution_time_avg vs. peak_memory_usage_avg for all operators.
    """
    out_path = os.path.join(CHARTS_DIR, "execution_time_vs_memory.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="peak_memory_usage_avg",
        y="execution_time_avg",
        hue="operator",
        ax=ax,
        alpha=0.7,
    )

    ax.set_xlabel("Avg Peak Memory (GB)")
    ax.set_ylabel("Avg Execution Time (s)")
    ax.set_title("Execution Time vs. Memory (All Operators)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    save_chart(fig, out_path)


def plot_feature_impact(df):
    """
    Creates a chart showing the impact of removing each feature (ΔRMSE).
    This may require single-feature removal steps.
    """
    out_path = os.path.join(CHARTS_DIR, "feature_impact.pdf")
    print("Generating:", out_path)

    # Because each operator might have different sets, do subplots or hue=operator
    # Approach: combine all, then show a bar chart with hue=operator for average ΔRMSE
    df = df.copy()
    df["selected_features"] = df["selected_features"].apply(ast.literal_eval)

    # A row with 'num_features' = n and the next row with 'num_features' = n-1
    # might differ by exactly one removed feature if it was done step-by-step
    # (This logic is naive; refine if needed.)
    records = []
    # Sort by operator and num_features descending
    df.sort_values(["operator", "num_features"], ascending=[True, False], inplace=True)

    for op in df["operator"].unique():
        sub_op = df[df["operator"] == op].reset_index(drop=True)
        for i in range(len(sub_op) - 1):
            f_current = set(sub_op.loc[i, "selected_features"])
            f_next = set(sub_op.loc[i + 1, "selected_features"])
            removed = f_current - f_next
            if len(removed) == 1:
                removed_feat = list(removed)[0]
                delta_rmse = sub_op.loc[i + 1, "rmse"] - sub_op.loc[i, "rmse"]
                records.append(
                    {
                        "operator": op,
                        "removed_feature": removed_feat,
                        "delta_rmse": delta_rmse,
                    }
                )

    if not records:
        print("No single-feature removal steps found in feature_selection data.")
        return

    rec_df = pd.DataFrame(records)
    fig, ax = plt.subplots()
    sns.barplot(data=rec_df, x="removed_feature", y="delta_rmse", hue="operator", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Removed Feature")
    ax.set_ylabel("Avg ΔRMSE")
    ax.set_title("Impact of Removing Each Feature on RMSE (All Operators)")
    ax.legend(bbox_to_anchor=(1, 1))
    save_chart(fig, out_path)


def plot_memory_progression_per_operator(df):
    """
    Generates one chart per operator, preserving the inlines/xlines facet structure.
    Each subplot shows memory usage vs. relative_time with lines colored/styled by 'samples'.
    Implements optional smoothing/downsampling for 'gst3d'.
    Moves the legend to the top, near the title.
    """

    # 1) Optional smoothing or downsampling for "gst3d"
    # Example: apply a small rolling average and reduce data points for lines.
    # Adjust rolling window or step for more/less smoothing/downsampling.
    def smooth_and_downsample(sub_df):
        sub_df = sub_df.sort_values("relative_time").copy()
        # Rolling average
        sub_df["captured_memory_usage"] = (
            sub_df["captured_memory_usage"]
            .rolling(window=5, center=True, min_periods=1)
            .mean()
        )
        # Downsampling every nth point (e.g., keep every 5th point)
        sub_df = sub_df.iloc[::5, :]
        return sub_df

    # Apply only to "gst3d" rows if needed
    gst3d_mask = df["operator"] == "gst3d"
    df_gst3d = df[gst3d_mask]
    df_not_gst3d = df[~gst3d_mask]

    if not df_gst3d.empty:
        df_gst3d = df_gst3d.groupby(
            ["xlines", "inlines", "samples"], group_keys=True
        ).apply(smooth_and_downsample)
    # Combine them back
    df = pd.concat([df_not_gst3d, df_gst3d], ignore_index=True)

    # 2) Convert these columns to strings if needed for faceting
    df["xlines"] = df["xlines"].astype(str)
    df["inlines"] = df["inlines"].astype(str)
    df["samples"] = df["samples"].astype(str)

    # Loop through each operator
    for op in df["operator"].unique():
        sub_op = df[df["operator"] == op].copy()
        if sub_op.empty:
            continue

        out_path = os.path.join(CHARTS_DIR, f"memory_progression_{op}.pdf")
        print(f"Generating: {out_path}")

        # 3) Use relplot with fewer markers. The 'markevery' param (matplotlib>=3.3)
        # can skip markers. Seaborn calls it as a style override in line_kws.
        # Here we place the legend at the top center using 'g._legend' calls.
        g = sns.relplot(
            data=sub_op,
            x="relative_time",
            y="captured_memory_usage",
            row="xlines",
            col="inlines",
            hue="samples",
            style="samples",
            kind="line",
            estimator=None,
            facet_kws={"sharex": False, "sharey": False},
            height=3,
            aspect=1.4,
        )

        g.set_axis_labels("Relative Time", "Memory Usage (GB)")
        g.set_titles(
            row_template="Xlines={row_name}", col_template="Inlines={col_name}"
        )
        g.fig.suptitle(
            f"Memory Usage Progression - Operator: {op}",
            fontsize=21,
            y="1.01",
        )

        for ax in g.axes.flat:
            rect = ax.patch
            rect.set_edgecolor("black")  # Color of the border
            rect.set_linewidth(1.5)  # Thickness of the border

        # Access the legend
        legend = g._legend  # or g.fig.legends[0] if you're using a custom placement

        # Style the legend
        legend.get_frame().set_edgecolor("black")  # Add black border
        legend.get_frame().set_linewidth(2)  # Border thickness
        legend.get_frame().set_linestyle("--")  # Dashed border
        legend.set_bbox_to_anchor((1, 0.5))  # Reposition if needed (optional)

        g.tight_layout()
        g.fig.savefig(out_path, bbox_inches="tight")
        plt.close(g.fig)


def plot_memory_usage_distribution(df):
    """
    Creates a distribution (violin or box) of memory usage across volumes/operators.
    """
    out_path = os.path.join(CHARTS_DIR, "memory_usage_distribution.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    sns.violinplot(
        data=df,
        x="operator",
        y="captured_memory_usage",
        cut=0,
        inner="quartile",
        bw="silverman",
        ax=ax,
    )
    ax.set_title("Memory Usage Distribution (All Operators)")
    ax.set_xlabel("Operator")
    ax.set_ylabel("Memory Usage (GB)")
    save_chart(fig, out_path)


def plot_memory_usage_heatmap_by_time(df):
    """
    Creates a heatmap of memory usage vs. time bins (x-axis) and operator or volume (y-axis).
    This example aggregates by operator + volume label. Adjust as needed.
    """
    out_path = os.path.join(CHARTS_DIR, "memory_usage_heatmap_by_time.pdf")
    print("Generating:", out_path)

    df = df.copy()
    df["time_bin"] = pd.cut(df["relative_time"], bins=50, labels=False)
    # Combine operator & volume as a grouping label so that each row in heatmap is an operator-volume pair
    df["op_vol"] = df["operator"] + "_" + df["volume"].astype(str)

    pivoted = (
        df.groupby(["op_vol", "time_bin"])["captured_memory_usage"].mean().unstack()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivoted, cmap="viridis", ax=ax)
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Operator-Volume")
    ax.set_title("Memory Usage Over Time (All Operators)")
    save_chart(fig, out_path)


def plot_memory_usage_by_configuration(df):
    """
    3D style plot or line plot showing how memory usage changes by time for each operator & volume.
    This approach is fairly dense, so adapt as needed.
    """
    out_path = os.path.join(CHARTS_DIR, "memory_usage_by_configuration.pdf")
    print("Generating:", out_path)

    # Example: 2D lineplot for each operator, volume, ignoring inlines/xlines
    df = df.copy()
    fig, ax = plt.subplots()
    for op in df["operator"].unique():
        sub_op = df[df["operator"] == op]
        for vol in sub_op["volume"].unique():
            sub_vol = sub_op[sub_op["volume"] == vol].sort_values("relative_time")
            ax.plot(
                sub_vol["relative_time"],
                sub_vol["captured_memory_usage"],
                alpha=0.3,
                label=f"{op}-{vol}",
            )
    ax.set_xlabel("Relative Time")
    ax.set_ylabel("Memory Usage (GB)")
    ax.set_title("Memory Usage Over Time by Config (All Operators)")
    ax.legend(bbox_to_anchor=(1, 1), ncol=1)
    save_chart(fig, out_path)


def plot_inlines_xlines_heatmap(df):
    """
    Creates a heatmap for peak memory usage across inlines,xlines for each operator.
    Might need subplots or one big pivot. This example demonstrates subplots by operator.
    """
    out_path = os.path.join(CHARTS_DIR, "memory_usage_inlines_xlines_heatmap.pdf")
    print("Generating:", out_path)

    # Use subplots with facet by operator
    operators = df["operator"].unique()
    cols = 3
    rows = int(np.ceil(len(operators) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, op in enumerate(operators):
        ax = axes[i]
        sub = df[df["operator"] == op]
        pivoted = (
            sub.groupby(["inlines", "xlines"])["captured_memory_usage"].max().unstack()
        )
        sns.heatmap(pivoted, cmap="viridis", ax=ax)
        ax.set_title(op)
        ax.set_xlabel("Xlines")
        ax.set_ylabel("Inlines")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Peak Memory Usage Heatmap by Inlines-Xlines")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_chart(fig, out_path)


def plot_inlines_xlines_samples_3d(df):
    """
    3D scatter of memory usage vs inlines,xlines,samples.
    Subplots by operator or color by operator in a single plot.
    """
    out_path = os.path.join(
        CHARTS_DIR, "memory_usage_inlines_xlines_samples_heatmap.pdf"
    )
    print("Generating:", out_path)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    grouped = df.groupby(["operator", "inlines", "xlines", "samples"], as_index=False)[
        "captured_memory_usage"
    ].max()

    # Use color map indexing by operator
    ops = grouped["operator"].unique()
    color_map = dict(zip(ops, sns.color_palette(n_colors=len(ops))))

    for op in ops:
        sub = grouped[grouped["operator"] == op]
        ax.scatter(
            sub["inlines"],
            sub["xlines"],
            sub["samples"],
            c=[color_map[op]] * len(sub),
            label=op,
            alpha=0.6,
            s=50,
        )

    ax.set_xlabel("Inlines")
    ax.set_ylabel("Xlines")
    ax.set_zlabel("Samples")
    ax.set_title("3D Peak Memory Usage (All Operators)")
    ax.legend(bbox_to_anchor=(1, 1))
    save_chart(fig, out_path)


def plot_peak_memory_by_volume(df):
    """
    Plots peak memory usage by volume with variance (std dev) for all operators on one chart.
    """
    out_path = os.path.join(CHARTS_DIR, "peak_memory_by_volume.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].dropna(
            subset=["peak_memory_usage_avg", "volume"]
        )
        sub = sub.sort_values("volume")
        ax.plot(sub["volume"], sub["peak_memory_usage_avg"], marker="o", label=op)
        if {"peak_memory_usage_std_dev"}.issubset(sub.columns):
            low = sub["peak_memory_usage_avg"] - sub["peak_memory_usage_std_dev"]
            high = sub["peak_memory_usage_avg"] + sub["peak_memory_usage_std_dev"]
            ax.fill_between(sub["volume"], low, high, alpha=0.2)

    ax.set_xlabel("Volume")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("Peak Memory by Volume (All Operators)")
    ax.legend()

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: format_volume_label(x))
    )

    save_chart(fig, out_path)


def plot_memory_vs_dimensions_pairplot(df):
    """
    Creates pairplots of inlines,xlines,samples,captured_memory_usage.
    Usually done separately by operator to avoid extremely large multi-operator plot.
    """
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].copy()
        needed = ["inlines", "xlines", "samples", "captured_memory_usage"]
        if not set(needed).issubset(sub.columns):
            continue

        out_path = os.path.join(CHARTS_DIR, f"memory_vs_dimensions_pairplot_{op}.pdf")
        print("Generating:", out_path)
        subset = sub[needed]
        g = sns.pairplot(
            subset,
            kind="reg",
            diag_kind="kde",
            plot_kws={"line_kws": {"color": "red"}},
        )
        g.fig.suptitle(f"Memory vs. Dimensions Pairplot - Operator: {op}", y=1.02)
        g.fig.tight_layout()
        g.fig.savefig(out_path)
        plt.close(g.fig)


def plot_memory_vs_volume_regression(df):
    """
    Plots memory usage vs. volume with a linear regression overlay, color-coded by operator.
    """
    out_path = os.path.join(CHARTS_DIR, "memory_vs_volume_regression.pdf")
    print("Generating:", out_path)

    if not {"volume", "peak_memory_usage_avg", "operator"}.issubset(df.columns):
        print("Data missing necessary columns for memory vs. volume regression.")
        return

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df, x="volume", y="peak_memory_usage_avg", hue="operator", ax=ax, alpha=0.7
    )
    # Add a regression line per operator
    operators = df["operator"].unique()
    for op in operators:
        sub = df[df["operator"] == op].sort_values("volume")
        if len(sub) > 1:
            x = sub["volume"].values
            y = sub["peak_memory_usage_avg"].values
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m * x + b, label=f"{op} fit")

    ax.set_xlabel("Volume")
    ax.set_ylabel("Avg Peak Memory (GB)")
    ax.set_title("Memory vs. Volume (Linear Fit by Operator)")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: format_volume_label(v))
    )

    save_chart(fig, out_path)


def plot_residual_vs_predicted(df):
    """
    Residual (y_pred - y_test) vs. predicted, grouped by model_name/operator.
    """
    out_path = os.path.join(CHARTS_DIR, "residual_vs_predicted.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    # Show multiple dots with different colors (operator) or shapes (model)
    # Because each row has a list of residuals. Flatten them for a single big scatter.
    # Include columns: model_name, operator

    all_points = []
    for idx, row in df.iterrows():
        if "residuals" not in row or "y_pred" not in row:
            continue
        residuals = np.array(ast.literal_eval(row["residuals"]))
        preds = np.array(ast.literal_eval(row["y_pred"]))
        if len(residuals) != len(preds):
            continue

        for res, p in zip(residuals, preds):
            all_points.append(
                {
                    "residual": res,
                    "predicted": p,
                    "model_name": row["model_name"],
                    "operator": row["operator"],
                }
            )

    if not all_points:
        print("No valid residual/predicted data found.")
        return

    df_points = pd.DataFrame(all_points)
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
    ax.set_title("Residual vs. Predicted (All Operators & Models)")
    ax.legend(bbox_to_anchor=(1, 1))
    save_chart(fig, out_path)


def plot_residual_qq_plots(df):
    """
    QQ-plot for residuals by operator/model. Plots subplots.
    """
    import scipy.stats as stats

    out_path = os.path.join(CHARTS_DIR, "residual_qq_plots.pdf")
    print("Generating:", out_path)

    # Collect each (operator, model_name) => array of residuals
    df = df.copy()
    df["residuals_parsed"] = df["residuals"].apply(
        lambda x: np.array(ast.literal_eval(x)) if pd.notna(x) else []
    )

    groups = df.groupby(["operator", "model_name"])
    n = len(groups)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, ((op, model), sub) in enumerate(groups):
        ax = axes[i]
        # Merge all residual arrays from sub
        combined = np.hstack(sub["residuals_parsed"].values)
        stats.probplot(combined, dist="norm", plot=ax)
        ax.set_title(f"{op}-{model}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Residual QQ-Plots (All Operators & Models)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_chart(fig, out_path)


# ------------------------------------------------------------------------------
# Additional Model/Data Reduction/Feature Selection Charts
# ------------------------------------------------------------------------------
def plot_performance_by_model(df):
    """Side-by-side bars for RMSE, MAE, R², Accuracy, grouped by model, color by operator."""
    out_path = os.path.join(CHARTS_DIR, "performance_by_model.pdf")
    print("Generating:", out_path)

    # Melt the relevant columns: [rmse, mae, r2, accuracy]
    keep_cols = ["model_name", "operator", "rmse", "mae", "r2", "accuracy"]
    sub = df[keep_cols].melt(
        id_vars=["operator", "model_name"], var_name="metric", value_name="value"
    )

    fig, ax = plt.subplots()
    sns.barplot(data=sub, x="model_name", y="value", hue="metric", ci=None, ax=ax)
    ax.set_title("Comparison of Model Performance (All Operators)")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    save_chart(fig, out_path)


def plot_score_by_model(df):
    """
    Plots model 'score' with operator as color/hue, side by side bars or grouped bars.
    """
    out_path = os.path.join(CHARTS_DIR, "score_by_model.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    sns.barplot(data=df, x="model_name", y="score", hue="operator", ax=ax)
    ax.set_title("Model Score (All Operators)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1, 1))
    save_chart(fig, out_path)


def plot_residuals_distribution_by_model(df):
    """
    Side-by-side distribution (violin or box) for residuals by model_name & operator.
    """
    out_path = os.path.join(CHARTS_DIR, "residuals_distribution_by_model.pdf")
    print("Generating:", out_path)

    # Expand residuals into a long DataFrame
    rows = []
    for idx, row in df.iterrows():
        if pd.notna(row.get("residuals")):
            arr = np.array(ast.literal_eval(row["residuals"]))
            for val in arr:
                rows.append(
                    {
                        "operator": row["operator"],
                        "model_name": row["model_name"],
                        "residual": val,
                    }
                )
    if not rows:
        print("No residual data found for distribution plot.")
        return

    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    sns.violinplot(
        data=long_df, x="model_name", y="residual", hue="operator", cut=0, ax=ax
    )
    ax.axhline(0, linestyle="--", color="black")
    ax.set_title("Residuals Distribution by Model (All Operators)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Residual")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    save_chart(fig, out_path)


def plot_metrics_evolution_by_sample_size(df):
    """
    Plots RMSE, MAE, R², Accuracy vs. number_of_samples in a single figure with multiple lines by operator.
    """
    out_path = os.path.join(CHARTS_DIR, "metrics_evolution_by_sample_size.pdf")
    print("Generating:", out_path)

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
    fig.suptitle("Metrics Evolution by Sample Size (All Operators)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_chart(fig, out_path)


def plot_score_by_sample_size(df):
    out_path = os.path.join(CHARTS_DIR, "score_by_sample_size.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].sort_values("num_samples")
        ax.plot(sub["num_samples"], sub["score"], marker="o", label=op)
    ax.set_title("Model Score vs. Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Score")
    ax.legend()
    save_chart(fig, out_path)


def plot_rmse_mae_ratio_by_sample_size(df):
    out_path = os.path.join(CHARTS_DIR, "rmse_mae_ratio_by_sample_size.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].sort_values("num_samples")
        ratio = sub["rmse"] / sub["mae"]
        ax.plot(sub["num_samples"], ratio, marker="o", label=op)
    ax.set_title("RMSE/MAE Ratio Over Data Reduction (All Operators)")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE/MAE Ratio")
    ax.legend()
    save_chart(fig, out_path)


def plot_residual_metrics_by_sample_size(df):
    """
    Plots residual-based metrics vs sample_size for each operator.
    e.g., we can put MAE, RMSE from the 'residuals' column or use precomputed columns.
    """
    out_path = os.path.join(CHARTS_DIR, "residual_metrics_by_sample_size.pdf")
    print("Generating:", out_path)

    # Use columns if they exist. Otherwise, compute from 'residuals'
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
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].sort_values("num_samples")
        ax.plot(sub["num_samples"], sub["mae_calc"], marker="o", label=f"{op} MAE")
        ax.plot(sub["num_samples"], sub["rmse_calc"], marker="s", label=f"{op} RMSE")

    ax.set_title("Error Metrics vs. Number of Samples")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Error")
    ax.legend()
    save_chart(fig, out_path)


def plot_metrics_evolution_by_number_of_features(df):
    """
    2x2 chart for RMSE, MAE, R², Accuracy vs. num_features.
    One line per operator.
    """
    out_path = os.path.join(CHARTS_DIR, "metrics_evolution_by_number_of_features.pdf")
    print("Generating:", out_path)

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
    fig.suptitle("Metrics Evolution by Number of Features (All Operators)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_chart(fig, out_path)


def plot_score_by_number_of_features(df):
    out_path = os.path.join(CHARTS_DIR, "score_by_number_of_features.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].sort_values("num_features")
        ax.plot(sub["num_features"], sub["score"], marker="o", label=op)
    ax.set_title("Score vs. Number of Features (All Operators)")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score")
    ax.legend()
    save_chart(fig, out_path)


def plot_rmse_mae_ratio_by_number_of_features(df):
    out_path = os.path.join(CHARTS_DIR, "rmse_mae_ratio_by_number_of_features.pdf")
    print("Generating:", out_path)

    fig, ax = plt.subplots()
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].sort_values("num_features")
        ratio = sub["rmse"] / sub["mae"]
        ax.plot(sub["num_features"], ratio, marker="o", label=op)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("RMSE/MAE Ratio")
    ax.set_title("RMSE/MAE Ratio by Number of Features (All Operators)")
    ax.legend()
    save_chart(fig, out_path)


def plot_residual_metrics_by_number_of_features(df):
    """
    Similar approach as sample_size residual metrics but for feature_count.
    """
    out_path = os.path.join(CHARTS_DIR, "residual_metrics_by_number_of_features.pdf")
    print("Generating:", out_path)

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
    for op in df["operator"].unique():
        sub = df[df["operator"] == op].sort_values("num_features")
        ax.plot(sub["num_features"], sub["mae_calc"], marker="o", label=f"{op} MAE")
        ax.plot(sub["num_features"], sub["rmse_calc"], marker="s", label=f"{op} RMSE")

    ax.set_title("Residual Metrics vs. Number of Features (All Operators)")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Error")
    ax.legend()
    save_chart(fig, out_path)


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


def save_chart(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
