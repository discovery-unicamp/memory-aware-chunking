"""
Script for building datasets, training multiple regression models, and evaluating them
for memory usage prediction (or any generic numeric target).

Environment Variables (with defaults):
  - OUTPUT_DIR: The base output directory (default './out')
  - PROFILES_DIR: Directory where '.prof' files (profiles) are located
                 (default: '<OUTPUT_DIR>/profiles')
  - RESULTS_DIR: Directory to store results (default: '<OUTPUT_DIR>/results')
  - OPERATORS_DIR: Directory for operator-specific results (default: '<RESULTS_DIR>/operators')
  - PROFILER: The profiler name, used to pick memory usage data keys (default 'kernel')
  - TEST_SIZE: Fraction for train/test split (default '0.2')
  - ACCURACY_THRESHOLD: Threshold to compute "accuracy" in terms of relative error (default '0.1')
  - MODELS_TO_EVALUATE: Comma-separated list of model keys (default includes many regressors)
  - OPTUNA_TRIALS: Number of trials for the optuna "best model weighting" search (default '50')
  - SCORE_ACCEPTANCE_THRESHOLD: Threshold for model score acceptance (default '0.1')
  - RANDOM_STATE: Random state seed for reproducibility (default '42')
"""

import json
import os
import pickle
import random

import numpy as np
import optuna
import pandas as pd
from common import transformers
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from traceq import load_profile
from xgboost import XGBRegressor

# ------------------------------------------------------------------------------
# Global Configuration (Environment Variables)
# ------------------------------------------------------------------------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out")
PROFILES_DIR = os.getenv("PROFILES_DIR", f"{OUTPUT_DIR}/profiles")
RESULTS_DIR = os.getenv("RESULTS_DIR", f"{OUTPUT_DIR}/results")
OPERATORS_DIR = os.getenv("OPERATOR_DIR", f"{RESULTS_DIR}/operators")
PROFILER = os.getenv("PROFILER", "kernel")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.05"))
SCORE_ACCEPTANCE_THRESHOLD = float(os.getenv("SCORE_ACCEPTANCE_THRESHOLD", "0.05"))
MODELS_TO_EVALUATE = os.getenv(
    "MODELS_TO_EVALUATE",
    "linear_regression,polynomial_regression,decision_tree,random_forest,"
    "gradient_boosting,xgboost,support_vector_regression,elastic_net",
).split(",")
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Global model constructor lookup
MODEL_CONSTRUCTORS_HASHMAP = {
    "linear_regression": lambda: LinearRegression(),
    "polynomial_regression": lambda: Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin_reg", LinearRegression()),
        ]
    ),
    "decision_tree": lambda: DecisionTreeRegressor(),
    "random_forest": lambda: RandomForestRegressor(),
    "gradient_boosting": lambda: GradientBoostingRegressor(),
    "xgboost": lambda: XGBRegressor(),
    "support_vector_regression": lambda: SVR(),
    "elastic_net": lambda: ElasticNet(),
}


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    """
    Main function that:
      1. Reads environment variables and prints config.
      2. Discovers and loads profile files.
      3. Builds a dataset/dataframe from those profiles.
      4. Extracts features for memory usage prediction.
      5. Trains/evaluates multiple models, finds best model weights (Optuna).
      6. Evaluates data reduction and feature selection using the best models.
    """
    print("Collecting results...")
    print("Using args:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  PROFILES_DIR: {PROFILES_DIR}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print(f"  OPERATORS_DIR: {OPERATORS_DIR}")
    print(f"  PROFILER: {PROFILER}")
    print(f"  TEST_SIZE: {TEST_SIZE}")
    print(f"  ACCURACY_THRESHOLD: {ACCURACY_THRESHOLD}")
    print(f"  MODELS_TO_EVALUATE: {MODELS_TO_EVALUATE}")
    print(f"  OPTUNA_TRIALS: {OPTUNA_TRIALS}")
    print(f"  RANDOM_STATE: {RANDOM_STATE}")
    print(f"  SCORE_ACCEPTANCE_THRESHOLD: {SCORE_ACCEPTANCE_THRESHOLD}")
    print()

    profile_filepaths = get_profile_filepaths()
    profiles = get_profiles(profile_filepaths)
    dataset = build_dataset(profile_filepaths, profiles)
    df = build_dataframe(dataset)
    df_features = extract_features(df)

    best_models, best_weights = find_best_models(df_features)

    data_reduction_results = evaluate_data_reduction(
        df_features,
        best_models,
        best_weights,
    )
    feature_selection_results = evaluate_feature_selection(
        df_features,
        best_models,
        best_weights,
    )

    build_final_model(
        best_models,
        df_features,
        data_reduction_results,
        feature_selection_results,
    )


# ------------------------------------------------------------------------------
# STEP 1 & 2: Getting & Loading Profiles
# ------------------------------------------------------------------------------
def get_profile_filepaths():
    """
    Finds all '.prof' filepaths in PROFILES_DIR.
    Returns a list of filenames (not full paths).
    """
    print("---------- STEP 1: Getting profile file paths")
    if not os.path.isdir(PROFILES_DIR):
        raise FileNotFoundError(f"Profiles directory not found: {PROFILES_DIR}")
    profile_files = [f for f in os.listdir(PROFILES_DIR) if f.endswith(".prof")]
    print(f"Found {len(profile_files)} profiles in {PROFILES_DIR}\n")
    return profile_files


def get_profiles(profile_filepaths):
    """
    Loads each profile file (e.g., JSON data) via `traceq.load_profile`.
    Returns a list of loaded profile objects.
    """
    print("---------- STEP 2: Getting profiles")
    profiles = [
        load_profile(os.path.join(PROFILES_DIR, filename))
        for filename in profile_filepaths
    ]
    print("Finished loading profiles. Sample profile:")
    if profiles:
        sample = random.choice(profiles)
        sample_json = json.dumps(sample, indent=4).split("\n")
        print("\n".join(sample_json[:20]))
        print("...")
    print()
    return profiles


# ------------------------------------------------------------------------------
# STEP 3: Building Dataset
# ------------------------------------------------------------------------------
def build_dataset(
    profile_filepaths, profiles, memory_usage_unit="gb", timestamp_unit="s"
):
    """
    Transforms the raw profiles into a structured list of dicts (dataset).
    Applies unit transformations if needed, and extracts inline/xline/sample info.
    """
    print("---------- STEP 3: Building dataset")
    dataset = []
    unit_transformers = {
        "kb_gb": transformers.transform_kb_to_gb,
        "ns_s": transformers.transform_ns_to_s,
    }

    for profile_filepath, profile in zip(profile_filepaths, profiles):
        # Parse name -> e.g. 'envelope-100-100-100-<session_id>.prof'
        profile_parts = profile_filepath.split("/")[-1].split(".")[0].split("-")
        inlines, xlines, samples, session_id = profile_parts[-4:]
        operator = "-".join(profile_parts[0:-4])

        # Memory usage
        profiler_data_key = f"{PROFILER}_memory_usage"
        profiler_unit = profile["metadata"][f"{profiler_data_key}_unit"]
        profiler_unit_transformer = unit_transformers[
            f"{profiler_unit}_{memory_usage_unit}"
        ]
        memory_usage_history = [
            profiler_unit_transformer(x[profiler_data_key]) for x in profile["data"]
        ]

        # Timestamps
        original_timestamp_unit = profile["metadata"]["unix_timestamp_unit"]
        timestamp_unit_transformer = unit_transformers[
            f"{original_timestamp_unit}_{timestamp_unit}"
        ]
        timestamp_history = [
            timestamp_unit_transformer(x["unix_timestamp"]) for x in profile["data"]
        ]

        dataset.append(
            {
                "session_id": session_id,
                "operator": operator,
                "inlines": int(inlines),
                "xlines": int(xlines),
                "samples": int(samples),
                "peak_memory_usage": max(memory_usage_history),
                "memory_usage_unit": memory_usage_unit,
                "memory_usage_history": memory_usage_history,
                "timestamp_history": timestamp_history,
                "timestamp_unit": timestamp_unit,
            }
        )

    print("Finished transforming the profiles into a dataset. Sample data:")
    if dataset:
        sample = random.choice(dataset)
        sample_json = json.dumps(sample, indent=4).split("\n")
        print("\n".join(sample_json[:20]))
        print("...")
    print()

    _extract_profiles_dataset_results(dataset)
    return dataset


def _extract_profiles_dataset_results(dataset):
    """
    Helper to build a "history" CSV and a "summary" CSV for each operator.
    Explodes memory usage/time sequences, saves them per-operator, etc.
    """
    print("Extracting results from profiles dataset...")

    df = pd.DataFrame(dataset)
    df["timestamped_memory_usage"] = df.apply(
        lambda row: list(zip(row["memory_usage_history"], row["timestamp_history"])),
        axis=1,
    )
    df = df.explode("timestamped_memory_usage")
    df[["captured_memory_usage", "timestamp"]] = pd.DataFrame(
        df["timestamped_memory_usage"].tolist(), index=df.index
    )
    df.drop(
        columns=[
            "timestamped_memory_usage",
            "memory_usage_history",
            "timestamp_history",
        ],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    df["volume"] = df["inlines"] * df["xlines"] * df["samples"]
    _extract_profiles_dataset_history(df)
    _extract_profiles_dataset_summary(df)


def _extract_profiles_dataset_history(df):
    """
    Extracts per-operator time-series "history" from the exploded DF
    and saves as 'profile_history.csv'.
    """
    print("Extracting history from profiles dataframe...")
    df = df.sort_values(by=["operator", "session_id", "timestamp"]).copy()
    df["relative_time"] = df.groupby("session_id").cumcount()

    for operator in df["operator"].unique():
        op_df = df[df["operator"] == operator].drop(columns=["operator"])
        sanitized = operator.replace(" ", "_").lower()
        out_path = f"{OPERATORS_DIR}/{sanitized}/results/profile_history.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        op_df.to_csv(out_path, index=False)
        print(f"Saved operator '{operator}' history to {out_path}")

    print("Finished extracting history. Sample:")
    print(df.head())


def _extract_profiles_dataset_summary(df):
    """
    Aggregates peak memory usage and execution time by volume/operator,
    computing average/min/max, etc., then saves as 'profile_summary.csv'.
    """
    print("Extracting summary from profiles dataframe...")
    df_exec = (
        df.groupby(["volume", "session_id", "operator"])["timestamp"]
        .agg(["min", "max"])
        .reset_index()
    )
    df_exec["execution_time"] = df_exec["max"] - df_exec["min"]
    df_exec["execution_time_unit"] = df["timestamp_unit"].iloc[0]

    df_memory = (
        df.groupby(["volume", "session_id", "operator"])["captured_memory_usage"]
        .max()
        .reset_index()
    )
    df_memory["captured_memory_usage_unit"] = df["memory_usage_unit"].iloc[0]

    df_grouped = pd.merge(df_memory, df_exec, on=["volume", "session_id", "operator"])
    df_summary = (
        df_grouped.groupby(["volume", "operator"])
        .agg(
            peak_memory_usage_avg=("captured_memory_usage", "mean"),
            peak_memory_usage_std_dev=("captured_memory_usage", "std"),
            peak_memory_usage_min=("captured_memory_usage", "min"),
            peak_memory_usage_max=("captured_memory_usage", "max"),
            execution_time_avg=("execution_time", "mean"),
            execution_time_std_dev=("execution_time", "std"),
            execution_time_min=("execution_time", "min"),
            execution_time_max=("execution_time", "max"),
            n_samples=("captured_memory_usage", "count"),
        )
        .reset_index()
    )

    df_summary["peak_memory_usage_cv"] = (
        df_summary["peak_memory_usage_std_dev"] / df_summary["peak_memory_usage_avg"]
    )
    df_summary["execution_time_cv"] = (
        df_summary["execution_time_std_dev"] / df_summary["execution_time_avg"]
    )

    df_summary["peak_memory_usage_unit"] = df["memory_usage_unit"].iloc[0]
    df_summary["execution_time_unit"] = df["timestamp_unit"].iloc[0]

    for operator in df_summary["operator"].unique():
        op_df = df_summary[df_summary["operator"] == operator].drop(
            columns=["operator"]
        )
        sanitized = operator.replace(" ", "_").lower()
        out_path = f"{OPERATORS_DIR}/{sanitized}/results/profile_summary.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        op_df.to_csv(out_path, index=False)
        print(f"Saved operator '{operator}' summary to {out_path}")

    print("Finished extracting summary. Sample:")
    print(df_summary.head())


# ------------------------------------------------------------------------------
# STEP 4: Building DataFrame
# ------------------------------------------------------------------------------
def build_dataframe(dataset):
    """
    Aggregates the dataset into a simpler DataFrame of (inlines, xlines, samples, operator)
    -> avg_peak_memory_usage. Saves as CSV for further usage.
    """
    print("---------- STEP 4: Building dataframe")
    df = pd.DataFrame(dataset)
    df = df.groupby(["inlines", "xlines", "samples", "operator"], as_index=False).agg(
        avg_peak_memory_usage=("peak_memory_usage", "mean")
    )

    print("Finished creating dataframe. Sample data:")
    print(df.head())

    # Save intermediate dataset
    df_output = f"{RESULTS_DIR}/data/dataset.csv"
    os.makedirs(os.path.dirname(df_output), exist_ok=True)
    df.to_csv(df_output, index=False)
    print(f"Finished saving dataframe to {df_output}")
    print()
    return df


# ------------------------------------------------------------------------------
# STEP 5: Feature Extraction
# ------------------------------------------------------------------------------
def extract_features(df):
    """
    Generates additional numeric features from (inlines, xlines, samples, operator, etc.).
    Saves the resulting feature DataFrame to CSV, then returns it.
    """
    print("---------- STEP 5: Extracting features")
    df_features = df.copy()

    # Basic combos
    df_features["volume"] = (
        df_features["inlines"] * df_features["xlines"] * df_features["samples"]
    )
    df_features["inlines_x_xlines"] = df_features["inlines"] * df_features["xlines"]
    df_features["inlines_x_samples"] = df_features["inlines"] * df_features["samples"]
    df_features["xlines_x_samples"] = df_features["xlines"] * df_features["samples"]

    # Geometry
    df_features["diagonal_length"] = np.sqrt(
        df_features["inlines"] ** 2
        + df_features["xlines"] ** 2
        + df_features["samples"] ** 2
    )
    df_features["surface_area"] = 2 * (
        df_features["inlines"] * df_features["xlines"]
        + df_features["inlines"] * df_features["samples"]
        + df_features["xlines"] * df_features["samples"]
    )

    # Log transformations
    df_features["log_inlines"] = np.log2(df_features["inlines"])
    df_features["log_xlines"] = np.log2(df_features["xlines"])
    df_features["log_samples"] = np.log2(df_features["samples"])
    df_features["log_volume"] = np.log2(df_features["volume"])

    # Ratios
    df_features["inline_to_xlines_ratio"] = (
        df_features["inlines"] / df_features["xlines"]
    )
    df_features["inlines_to_samples_ratio"] = (
        df_features["inlines"] / df_features["samples"]
    )
    df_features["xlines_to_samples_ratio"] = (
        df_features["xlines"] / df_features["samples"]
    )

    total_sum = df_features["inlines"] + df_features["xlines"] + df_features["samples"]
    df_features["inlines_to_total_ratio"] = df_features["inlines"] / total_sum
    df_features["xlines_to_total_ratio"] = df_features["xlines"] / total_sum
    df_features["samples_to_total_ratio"] = df_features["samples"] / total_sum

    # Additional aggregates
    df_features["mean_inlines_xlines"] = (
        df_features["inlines"] + df_features["xlines"]
    ) / 2
    # Could be improved (the 'np.std([...])' across columns is not always correct row-wise),
    # but let's keep as is from your code:
    df_features["std_inlines_xlines"] = np.std(
        [df_features["inlines"], df_features["xlines"]]
    )

    # Quadratic / log combos
    df_features["quadratic_interaction"] = df_features["volume"] ** 2
    df_features["log_volume_x_log_diagonal"] = df_features["log_volume"] * np.log1p(
        df_features["diagonal_length"]
    )

    print("Finished extracting features. Sample data:")
    print(df_features.head())

    # Save to CSV
    df_features_output = f"{RESULTS_DIR}/data/features.csv"
    os.makedirs(os.path.dirname(df_features_output), exist_ok=True)
    df_features.to_csv(df_features_output, index=False)
    print(f"Finished saving features to {df_features_output}\n")

    return df_features


# ------------------------------------------------------------------------------
# STEP 6: Find Best Models
# ------------------------------------------------------------------------------
def find_best_models(df_features, models_to_evaluate=MODELS_TO_EVALUATE):
    """
    Trains/evaluates each model in `models_to_evaluate` for each operator,
    storing metrics and picking the best combination of weight parameters (via Optuna).
    Returns a dict of best_models for each operator, and best_weights for each operator.
    """
    print("---------- STEP 6: Evaluating models")
    invalid_models = [
        m for m in models_to_evaluate if m not in MODEL_CONSTRUCTORS_HASHMAP
    ]
    if invalid_models:
        raise ValueError(f"Invalid models specified: {invalid_models}")

    operators = df_features["operator"].unique()
    enabled_models = [
        (mname, MODEL_CONSTRUCTORS_HASHMAP[mname]()) for mname in models_to_evaluate
    ]

    best_models = {op: None for op in operators}
    best_weights = {op: None for op in operators}

    for operator in operators:
        df_op = df_features[df_features["operator"] == operator]
        operator_output_dir = f"{OPERATORS_DIR}/{operator}"

        operator_train_results = {}
        for model_name, model_instance in enabled_models:
            print(f"Evaluating model for {operator}: {model_name}")
            model_output_dir = f"{operator_output_dir}/models/{model_name}"
            os.makedirs(model_output_dir, exist_ok=True)

            train_result = train_model(model_name, model_instance, df_op)
            operator_train_results[model_name] = train_result
            save_train_result(train_result, model_output_dir)

        chosen_model, chosen_weights = find_best_model(
            operator_train_results, operator_output_dir
        )
        best_models[operator] = chosen_model
        best_weights[operator] = chosen_weights

    print("Finished evaluating models. Best models per operator:")
    print(json.dumps(best_models, indent=4))
    print()
    return best_models, best_weights


def train_model(
    model_name,
    model,
    df_op,
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE,
):
    """
    Splits data into train/test, fits the model (iteratively reweighting underestimates),
    then computes RMSE/MAE/R2/Accuracy. Returns a dict with the trained model object,
    data subsets, and metrics.
    """
    X = df_op.drop(columns=["operator", "avg_peak_memory_usage"])
    y = df_op["avg_peak_memory_usage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    rmse, mae, r2, accuracy, residuals, y_pred = get_model_metrics(
        model,
        X_test,
        y_test,
    )

    print(
        f"Results for {model_name}: "
        f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Accuracy={accuracy:.2%}"
    )

    return {
        "model": model,
        "data": {
            "X": X,
            "X_train": X_train,
            "X_test": X_test,
            "y": y,
            "y_train": y_train,
            "y_test": y_test.to_list(),
            "y_pred": y_pred.tolist(),
            "residuals": residuals.to_list(),
        },
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "accuracy": accuracy,
        },
    }


def get_model_metrics(model, X_test, y_test, acc_threshold=ACCURACY_THRESHOLD):
    """
    Predicts on X_test and calculates RMSE, MAE, R^2, and "accuracy"
    as the fraction within `acc_threshold` of relative error.
    """
    y_pred = model.predict(X_test)
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    within_tolerance = (np.abs((y_pred - y_test) / y_test) <= acc_threshold) | (
        y_pred >= y_test
    )

    accuracy_val = np.mean(within_tolerance)
    residuals = y_test - y_pred

    return rmse_val, mae_val, r2_val, accuracy_val, residuals, y_pred


def find_best_model(train_results, operator_output_dir, n_trials=OPTUNA_TRIALS):
    """
    Uses Optuna to find the best scoring weight parameters for accuracy, rmse, mae, r2.
    Saves a 'model_metrics.csv' with each model's computed score.
    Returns (best_model_name, best_weights).
    """
    metrics_map = {mn: res["metrics"] for mn, res in train_results.items()}

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, metrics_map), n_trials=n_trials)
    best_params = study.best_params

    metric_results = []
    for model_name, mm in metrics_map.items():
        # Reconstruct train data
        data_portion = train_results[model_name]["data"]
        score_val = calculate_model_score(
            mm["accuracy"],
            mm["rmse"],
            mm["mae"],
            mm["r2"],
            best_params,
        )

        metric_results.append(
            {
                "model_name": model_name,
                "score": score_val,
                "accuracy": mm["accuracy"],
                "rmse": mm["rmse"],
                "mae": mm["mae"],
                "r2": mm["r2"],
                "residuals": data_portion["residuals"],
                "y_pred": data_portion["y_pred"],
                "y_test": data_portion["y_test"],
                **best_params,
            }
        )

    df_metrics = pd.DataFrame(metric_results)
    operator_results_dir = f"{operator_output_dir}/results"
    os.makedirs(operator_results_dir, exist_ok=True)
    df_metrics.to_csv(f"{operator_results_dir}/model_metrics.csv", index=False)

    # Pick best model by highest score
    best_row = df_metrics.loc[df_metrics["score"].idxmax()]
    print(f"Best model for {operator_output_dir}: {best_row['model_name']}")
    print(f"  Accuracy: {best_row['accuracy']:.2%}")
    print(f"  RMSE: {best_row['rmse']}")
    print(f"  MAE: {best_row['mae']}")
    print(f"  R2: {best_row['r2']}")

    return best_row["model_name"], best_params


def objective(trial, metrics_map):
    """
    Optuna objective function: param search for weighting of accuracy, RMSE, MAE, R2.
    The goal is to maximize the sum of weighted model metrics across all models.
    """
    acc_weight = trial.suggest_float("accuracy_weight", 1.0, 2.0)
    rmse_weight = trial.suggest_float("rmse_weight", 0.5, 1.5)
    mae_weight = trial.suggest_float("mae_weight", 0.5, 1.5)
    r2_weight = trial.suggest_float("r2_weight", 0.2, 1.0)

    total_score = 0
    for _, mm in metrics_map.items():
        score_val = (
            acc_weight * mm["accuracy"]
            - rmse_weight * mm["rmse"]
            - mae_weight * mm["mae"]
            + r2_weight * mm["r2"]
        )
        total_score += score_val
    return total_score


def calculate_model_score(acc, rmse, mae, r2, weights):
    """
    Utility to compute a single "score" from accuracy, RMSE, MAE, R^2
    given the weighting parameters discovered by Optuna.
    """
    return (
        weights["accuracy_weight"] * acc
        - weights["rmse_weight"] * rmse
        - weights["mae_weight"] * mae
        + weights["r2_weight"] * r2
    )


# ------------------------------------------------------------------------------
# STEP 7: Evaluate Data Reduction
# ------------------------------------------------------------------------------
def evaluate_data_reduction(df_features, best_models, best_weights, min_size=10):
    """
    Iteratively trains the best model while progressively reducing data size
    to see how performance changes. Saves 'data_reduction.csv' per operator.
    """
    print("---------- STEP 7: Evaluating data reduction")
    operators = df_features["operator"].unique()
    df_data_reduction = pd.DataFrame(
        {
            "operator": pd.Series(dtype="string"),
            "num_samples": pd.Series(dtype="int32"),
            "model_name": pd.Series(dtype="string"),
            "rmse": pd.Series(dtype="float64"),
            "mae": pd.Series(dtype="float64"),
            "r2": pd.Series(dtype="float64"),
            "accuracy": pd.Series(dtype="float64"),
            "score": pd.Series(dtype="float64"),
        }
    )

    for operator in operators:
        print(f"Evaluating data reduction for {operator}...")
        df_op = df_features[df_features["operator"] == operator].copy()
        output_dir = f"{OPERATORS_DIR}/{operator}"
        os.makedirs(output_dir, exist_ok=True)

        data_reduction_results = pd.DataFrame(
            {
                "num_samples": pd.Series(dtype="int32"),
                "model_name": pd.Series(dtype="string"),
                "rmse": pd.Series(dtype="float64"),
                "mae": pd.Series(dtype="float64"),
                "r2": pd.Series(dtype="float64"),
                "accuracy": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
            }
        )

        model_name = best_models[operator]
        build_model = MODEL_CONSTRUCTORS_HASHMAP[model_name]
        model = build_model()
        weights = best_weights[operator]

        while len(df_op) >= min_size:
            train_res = train_model(model_name, model, df_op)
            m = train_res["metrics"]
            data_portion = train_res["data"]
            score_val = calculate_model_score(
                m["accuracy"], m["rmse"], m["mae"], m["r2"], weights
            )

            row = {
                "num_samples": len(df_op),
                "model_name": model_name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "accuracy": m["accuracy"],
                "residuals": data_portion["residuals"],
                "y_pred": data_portion["y_pred"],
                "y_test": data_portion["y_test"],
                "score": score_val,
            }
            data_reduction_results = pd.concat(
                [data_reduction_results, pd.DataFrame([row])],
                ignore_index=True,
            )
            df_data_reduction = pd.concat(
                [
                    df_data_reduction,
                    pd.DataFrame(
                        [
                            {
                                "operator": operator,
                                **row,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            # Removes ~10% of samples
            n_samples = int(len(df_op) * 0.9)
            df_op = df_op.sample(n=n_samples, random_state=RANDOM_STATE)

        # Save results
        print(f"Finished data reduction for {operator}. Sample data:")
        print(data_reduction_results.head())

        out_path = f"{output_dir}/results/data_reduction.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data_reduction_results.to_csv(out_path, index=False)

    print("Finished evaluating data reduction.\n")

    return df_data_reduction


# ------------------------------------------------------------------------------
# STEP 8: Evaluate Feature Selection
# ------------------------------------------------------------------------------
def evaluate_feature_selection(df_features, best_models, best_weights, min_size=1):
    """
    Iteratively removes the "least-important" feature (via SelectKBest) to see
    how performance changes as features are removed. Saves 'feature_selection.csv'
    per operator.
    """
    print("---------- STEP 8: Evaluating feature selection")
    operators = df_features["operator"].unique()
    df_feature_selection = pd.DataFrame(
        {
            "operator": pd.Series(dtype="string"),
            "num_features": pd.Series(dtype="int32"),
            "selected_features": pd.Series(dtype="string"),
            "model_name": pd.Series(dtype="string"),
            "rmse": pd.Series(dtype="float64"),
            "mae": pd.Series(dtype="float64"),
            "r2": pd.Series(dtype="float64"),
            "accuracy": pd.Series(dtype="float64"),
            "score": pd.Series(dtype="float64"),
        }
    )

    for operator in operators:
        print(f"Evaluating feature selection for {operator}...")
        df_op = df_features[df_features["operator"] == operator].copy()
        output_dir = f"{OPERATORS_DIR}/{operator}"
        os.makedirs(output_dir, exist_ok=True)

        feature_results = pd.DataFrame(
            {
                "num_features": pd.Series(dtype="int32"),
                "selected_features": pd.Series(dtype="string"),
                "model_name": pd.Series(dtype="string"),
                "rmse": pd.Series(dtype="float64"),
                "mae": pd.Series(dtype="float64"),
                "r2": pd.Series(dtype="float64"),
                "accuracy": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
            }
        )

        model_name = best_models[operator]
        build_model = MODEL_CONSTRUCTORS_HASHMAP[model_name]
        model = build_model()
        weights = best_weights[operator]

        num_feats = len(df_op.columns)
        while num_feats >= min_size:
            train_res = train_model(model_name, model, df_op)
            m = train_res["metrics"]
            data_portion = train_res["data"]
            score_val = calculate_model_score(
                m["accuracy"], m["rmse"], m["mae"], m["r2"], weights
            )

            selected_feats = list(
                set(df_op.columns.tolist()) - {"operator", "avg_peak_memory_usage"}
            )
            row = {
                "num_features": len(df_op.columns),
                "selected_features": selected_feats,
                "model_name": model_name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "accuracy": m["accuracy"],
                "residuals": data_portion["residuals"],
                "y_pred": data_portion["y_pred"],
                "y_test": data_portion["y_test"],
                "score": score_val,
            }
            feature_results = pd.concat(
                [feature_results, pd.DataFrame([row])],
                ignore_index=True,
            )
            df_feature_selection = pd.concat(
                [
                    df_feature_selection,
                    pd.DataFrame(
                        [
                            {
                                "operator": operator,
                                **row,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            # Use SelectKBest to remove one feature at a time
            selector = SelectKBest(score_func=f_regression, k=num_feats - 1)
            X_data, y_data = data_portion["X"], data_portion["y"]
            selector.fit_transform(X_data, y_data)

            # Build new columns list
            keep_cols = ["operator", "avg_peak_memory_usage"] + X_data.columns[
                selector.get_support()
            ].tolist()
            df_op = df_op[keep_cols]
            num_feats -= 1

        print(f"Finished feature selection for {operator}. Sample data:")
        print(feature_results.head())

        out_path = f"{output_dir}/results/feature_selection.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        feature_results.to_csv(out_path, index=False)

    print("Finished evaluating feature selection.\n")

    return df_feature_selection


# ------------------------------------------------------------------------------
# STEP 9: Build Final Model
# ------------------------------------------------------------------------------
def build_final_model(
    best_models,
    df_features,
    data_reduction_results,
    feature_selection_results,
    score_acceptance_threshold=SCORE_ACCEPTANCE_THRESHOLD,
):
    print("---------- STEP 9: Building final model")
    operators = best_models.keys()

    for operator in operators:
        print(f"Evaluating feature selection and data reduction for '{operator}'...")
        output_dir = f"{OUTPUT_DIR}/best_models"
        os.makedirs(output_dir, exist_ok=True)

        # ---------------------------
        # 1) Filter the data by operator
        # ---------------------------
        df_op_features = df_features[df_features["operator"] == operator].copy()

        # ---------------------------
        # 2) Feature Selection: find row meeting threshold with smallest num_features
        # ---------------------------
        fs_op = feature_selection_results[
            feature_selection_results["operator"] == operator
        ].copy()
        if fs_op.empty:
            print(f"No feature selection results found for {operator}. Skipping...")
            continue

        max_fs_score = fs_op["score"].max()
        fs_acceptance_limit = max_fs_score * (1 - score_acceptance_threshold)

        # Filter rows that meet or exceed acceptance limit
        fs_candidates = fs_op[fs_op["score"] >= fs_acceptance_limit]
        if fs_candidates.empty:
            # If nothing meets the threshold, just pick the highest scoring row
            fs_candidates = fs_op[fs_op["score"] == max_fs_score]

        # Sort by num_features ascending
        fs_candidates = fs_candidates.sort_values(by="num_features", ascending=True)
        best_fs_row = fs_candidates.iloc[0]
        selected_features = best_fs_row["selected_features"]

        # ---------------------------
        # 3) Data Reduction: find row meeting threshold with smallest num_samples
        # ---------------------------
        dr_op = data_reduction_results[
            data_reduction_results["operator"] == operator
        ].copy()
        if dr_op.empty:
            print(f"No data reduction results found for {operator}. Skipping...")
            continue

        max_dr_score = dr_op["score"].max()
        dr_acceptance_limit = max_dr_score * (1 - score_acceptance_threshold)

        # Filter rows that meet or exceed acceptance limit
        dr_candidates = dr_op[dr_op["score"] >= dr_acceptance_limit]
        if dr_candidates.empty:
            # If nothing meets the threshold, just pick the highest scoring row
            dr_candidates = dr_op[dr_op["score"] == max_dr_score]

        # Sort by num_samples ascending
        dr_candidates = dr_candidates.sort_values(by="num_samples", ascending=True)
        best_dr_row = dr_candidates.iloc[0]
        best_num_samples = int(best_dr_row["num_samples"])

        # ---------------------------
        # 4) Prepare Final Training Data
        # ---------------------------
        # Limit the dataset to best_num_samples (if available) and select only chosen features
        if len(df_op_features) > best_num_samples:
            df_op_features = df_op_features.sample(
                n=best_num_samples, random_state=RANDOM_STATE
            )

        # Ensure the selected features exist in df_op_features
        valid_selected_features = [
            feat for feat in selected_features if feat in df_op_features.columns
        ]
        if not valid_selected_features:
            print(f"No valid selected features for {operator}. Skipping...")
            continue

        features = ["operator", "avg_peak_memory_usage", *valid_selected_features]
        df_op_features = df_op_features[features].copy()

        # ---------------------------
        # 5) Train the Best Model
        # ---------------------------
        best_model_name = best_models[operator]
        build_model = MODEL_CONSTRUCTORS_HASHMAP[best_model_name]
        best_model = build_model()
        results = train_model(best_model_name, best_model, df_op_features)

        # ---------------------------
        # 6) Save the Trained Model
        # ---------------------------
        model_path = os.path.join(output_dir, f"{operator}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(results["model"], f)
        print(f"Model saved at: {model_path}")


# ------------------------------------------------------------------------------
# Saving Train Results
# ------------------------------------------------------------------------------
def save_train_result(train_result, output_dir):
    """
    Saves the model training data and pickled model to the specified output_dir.
    """
    print(f"Saving data for model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    save_model_train_data(train_result, output_dir)
    save_model(train_result, output_dir)


def save_model_train_data(train_result, output_dir):
    """
    Saves X/X_train/X_test/y/y_train as CSV, plus residuals/predictions.
    """
    print("Saving model train data...")
    data = train_result["data"]

    data["X"].to_csv(f"{output_dir}/X.csv")
    data["X_train"].to_csv(f"{output_dir}/X_train.csv")
    data["X_test"].to_csv(f"{output_dir}/X_test.csv")
    data["y"].to_csv(f"{output_dir}/y.csv")
    data["y_train"].to_csv(f"{output_dir}/y_train.csv")

    pd.DataFrame(data["y_test"]).to_csv(f"{output_dir}/y_test.csv", index=False)
    pd.DataFrame(data["residuals"]).to_csv(f"{output_dir}/residuals.csv", index=False)
    pd.DataFrame(data["y_pred"]).to_csv(f"{output_dir}/y_pred.csv", index=False)


def save_model(train_result, output_dir):
    """
    Pickles the trained model object to 'model.pkl'.
    """
    print(f"Saving model object to {output_dir}...")
    model_obj = train_result["model"]
    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(model_obj, f)


# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
