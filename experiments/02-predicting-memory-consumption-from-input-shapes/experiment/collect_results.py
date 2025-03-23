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
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from traceq import load_profile
from xgboost import XGBRegressor

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out")
PROFILES_DIR = os.getenv("PROFILES_DIR", f"{OUTPUT_DIR}/profiles")
RESULTS_DIR = os.getenv("RESULTS_DIR", f"{OUTPUT_DIR}/results")
OPERATORS_DIR = os.getenv("OPERATOR_DIR", f"{RESULTS_DIR}/operators")
PROFILER = os.getenv("PROFILER", "kernel")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.1"))
MODELS_TO_EVALUATE = os.getenv(
    "MODELS_TO_EVALUATE",
    "linear_regression,polynomial_regression,decision_tree,random_forest,gradient_boosting,neural_network,xgboost,support_vector_regression,elastic_net",
).split(",")
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MODELS_HASHMAP = {
    "linear_regression": LinearRegression(),
    "polynomial_regression": Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin_reg", LinearRegression()),
        ]
    ),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "neural_network": Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor())]),
    "xgboost": XGBRegressor(),
    "support_vector_regression": SVR(),
    "elastic_net": ElasticNet(),
}


def main():
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
    print()

    profile_filepaths = __get_profile_filepaths()
    profiles = __get_profiles(profile_filepaths)
    dataset = __build_dataset(profile_filepaths, profiles)
    df = __build_dataframe(dataset)
    df_features = __extract_features(df)
    best_models, best_weights = __find_best_models(df_features)

    __evaluate_data_reduction(df_features, best_models, best_weights)
    __evaluate_feature_selection(df_features, best_models, best_weights)


def __get_profile_filepaths():
    print("---------- STEP 1: Getting profile file paths")
    profile_filepaths = [f for f in os.listdir(PROFILES_DIR) if f.endswith(".prof")]
    print(f"Found {len(profile_filepaths)} profiles in {PROFILES_DIR}")
    print()

    return profile_filepaths


def __get_profiles(profile_filepaths: list[str]):
    print("---------- STEP 2: Getting profiles")
    profiles = [load_profile(os.path.join(PROFILES_DIR, f)) for f in profile_filepaths]
    print("Finished loading profiles. Sample profile:")
    print(
        "\n".join(
            json.dumps(profiles[random.choice(range(len(profiles)))], indent=4).split(
                "\n"
            )[:20]
        )
    )
    print("...")
    print()

    return profiles


def __build_dataset(
    profile_filepaths: list[str],
    profiles: list,
    memory_usage_unit: str = "gb",
    timestamp_unit: str = "s",
) -> list:
    print("---------- STEP 3: Building dataset")
    dataset = []
    unit_transformers = {
        "kb_gb": transformers.transform_kb_to_gb,
        "ns_s": transformers.transform_ns_to_s,
    }

    for profile_filepath, profile in zip(profile_filepaths, profiles):
        profile_parts = profile_filepath.split("/")[-1].split(".")[0].split("-")
        inlines, xlines, samples, session_id = profile_parts[-4:]
        operator = "-".join(profile_parts[0:-4])

        profiler_data_key = f"{PROFILER}_memory_usage"
        profiler_unit = profile["metadata"][f"{profiler_data_key}_unit"]
        profiler_unit_transformer = unit_transformers[
            f"{profiler_unit}_{memory_usage_unit}"
        ]
        memory_usage_history = [
            profiler_unit_transformer(x[profiler_data_key]) for x in profile["data"]
        ]

        original_timestamp_unit = profile["metadata"][f"unix_timestamp_unit"]
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
    print(
        "\n".join(
            json.dumps(dataset[random.choice(range(len(dataset)))], indent=4).split(
                "\n"
            )[:20]
        )
    )
    print("...")
    print()

    __extract_profiles_dataset_results(dataset)

    return dataset


def __extract_profiles_dataset_results(dataset: list):
    print("Extracting results from profiles dataframe...")

    df = pd.DataFrame(dataset)
    df["timestamped_memory_usage"] = df.apply(
        lambda row: list(zip(row["memory_usage_history"], row["timestamp_history"])),
        axis=1,
    )
    df = df.explode("timestamped_memory_usage")
    df[["captured_memory_usage", "timestamp"]] = pd.DataFrame(
        df["timestamped_memory_usage"].tolist(), index=df.index
    )
    df = df.drop(
        columns=[
            "timestamped_memory_usage",
            "memory_usage_history",
            "timestamp_history",
        ]
    )
    df = df.reset_index(drop=True)
    df["volume"] = df["inlines"] * df["xlines"] * df["samples"]

    print("Finished building profiles dataframe. Sample data:")
    print(df.head())

    __extract_profies_dataset_history(df)
    __extract_profiles_dataset_summary(df)


def __extract_profies_dataset_history(df: pd.DataFrame):
    print("Extracting history from profiles dataframe...")

    df = df.copy()
    df = df.sort_values(by=["operator", "session_id", "timestamp"])
    df["relative_time"] = df.groupby("session_id").cumcount()

    for operator in df["operator"].unique():
        operator_df = df[df["operator"] == operator]
        operator_df = operator_df.drop(columns=["operator"])
        sanitized_name = operator.replace(" ", "_").lower()
        operator_path = f"{OPERATORS_DIR}/{sanitized_name}/results/profile_history.csv"
        os.makedirs(os.path.dirname(operator_path), exist_ok=True)
        operator_df.to_csv(operator_path, index=False)
        print(f"Saved operator '{operator}' history to {operator_path}")

    print("Finished extracting history from profiles dataframe. Sample data:")
    print(df.head())


def __extract_profiles_dataset_summary(df: pd.DataFrame):
    print("Extracting summary from profiles dataframe...")

    df = df.copy()

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
        operator_df = df_summary[df_summary["operator"] == operator]
        operator_df = operator_df.drop(columns=["operator"])
        sanitized_name = operator.replace(" ", "_").lower()
        operator_path = f"{OPERATORS_DIR}/{sanitized_name}/results/profile_summary.csv"
        os.makedirs(os.path.dirname(operator_path), exist_ok=True)
        operator_df.to_csv(operator_path, index=False)
        print(f"Saved operator '{operator}' summary to {operator_path}")

    print("Finished extracting summary from profiles dataframe. Sample data:")
    print(df_summary.head())


def __build_dataframe(dataset: list) -> pd.DataFrame:
    print("---------- STEP 4: Building dataframe")
    df = pd.DataFrame(dataset)
    df = df.groupby(["inlines", "xlines", "samples", "operator"], as_index=False).agg(
        avg_peak_memory_usage=("peak_memory_usage", "mean")
    )
    print("Finished creating dataframe. Sample data:")
    print(df.head())
    print("Saving dataframe...")
    df_output = f"{RESULTS_DIR}/data/dataset.csv"
    os.makedirs(os.path.dirname(df_output), exist_ok=True)
    df.to_csv(df_output, index=False)
    print(f"Finished saving dataframe to {df_output}")
    print()

    return df


def __extract_features(df: pd.DataFrame) -> pd.DataFrame:
    print("---------- STEP 5: Extracting features")
    df_features = df.copy()

    df_features["volume"] = (
        df_features["inlines"] * df_features["xlines"] * df_features["samples"]
    )
    df_features["inlines_x_xlines"] = df_features["inlines"] * df_features["xlines"]
    df_features["inlines_x_samples"] = df_features["inlines"] * df_features["samples"]
    df_features["xlines_x_samples"] = df_features["xlines"] * df_features["samples"]
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
    df_features["log_inlines"] = np.log2(df_features["inlines"])
    df_features["log_xlines"] = np.log2(df_features["xlines"])
    df_features["log_samples"] = np.log2(df_features["samples"])
    df_features["log_volume"] = np.log2(df_features["volume"])
    df_features["inline_to_xlines_ratio"] = (
        df_features["inlines"] / df_features["xlines"]
    )
    df_features["inlines_to_samples_ratio"] = (
        df_features["inlines"] / df_features["samples"]
    )
    df_features["xlines_to_samples_ratio"] = (
        df_features["xlines"] / df_features["samples"]
    )
    df_features["inlines_to_total_ratio"] = df_features["inlines"] / (
        df_features["inlines"] + df_features["xlines"] + df_features["samples"]
    )
    df_features["xlines_to_total_ratio"] = df_features["xlines"] / (
        df_features["inlines"] + df_features["xlines"] + df_features["samples"]
    )
    df_features["samples_to_total_ratio"] = df_features["samples"] / (
        df_features["inlines"] + df_features["xlines"] + df_features["samples"]
    )
    df_features["mean_inlines_xlines"] = (
        df_features["inlines"] + df_features["xlines"]
    ) / 2
    df_features["std_inlines_xlines"] = np.std(
        [df_features["inlines"], df_features["xlines"]]
    )
    df_features["quadratic_interaction"] = df_features["volume"] ** 2
    df_features["log_volume_x_log_diagonal"] = df_features["log_volume"] * np.log1p(
        df_features["diagonal_length"]
    )

    print("Finished extracting features. Sample data:")
    print(df_features.head())
    print("Saving features...")
    df_features_output = f"{RESULTS_DIR}/data/features.csv"
    df_features.to_csv(df_features_output, index=False)
    print(f"Finished saving features to {df_features_output}")
    print()

    return df_features


def __find_best_models(
    df_features: pd.DataFrame,
    models_to_evaluate=MODELS_TO_EVALUATE,
):
    print("---------- STEP 6: Evaluating models")
    invalid_models = [m for m in models_to_evaluate if m not in MODELS_HASHMAP]
    if invalid_models:
        raise ValueError(f"Invalid models: {invalid_models}")

    operators = df_features["operator"].unique()
    enabled_models = [
        (model_name, MODELS_HASHMAP[model_name]) for model_name in models_to_evaluate
    ]
    best_models = {operator: None for operator in operators}
    best_weights = {operator: None for operator in operators}

    for operator in operators:
        df_operator = df_features[df_features["operator"] == operator]
        operator_output_dir = f"{OPERATORS_DIR}/{operator}"

        operator_train_results = {}
        for model_name, model in enabled_models:
            print(f"Evaluating model for {operator}: {model_name}")
            model_output_dir = f"{operator_output_dir}/models/{model_name}"
            os.makedirs(model_output_dir, exist_ok=True)

            train_result = __train_model(
                model_name,
                model,
                df_operator,
            )
            operator_train_results[model_name] = train_result
            __save_train_result(train_result, model_output_dir)

        best_models[operator], best_weights[operator] = __find_best_model(
            operator_train_results,
            operator_output_dir,
        )

    print("Finished evaluating models")
    print("Best models:")
    print(json.dumps(best_models, indent=4))
    print()

    return best_models, best_weights


def __train_model(
    model_name,
    model,
    df_operator: pd.DataFrame,
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE,
):
    X = df_operator.drop(
        columns=[
            "operator",
            "avg_peak_memory_usage",
        ]
    )
    y = df_operator["avg_peak_memory_usage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Training {model_name}")
    model.fit(X_train, y_train)

    rmse, mae, r2, accuracy, residuals, y_pred = __get_model_metrics(
        model,
        X_test,
        y_test,
    )

    print(f"Results for {model_name} model:")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")
    print(f"  Accuracy: {accuracy * 100}%")

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


def __get_model_metrics(model, X_test, y_test, acc_threshold=ACCURACY_THRESHOLD):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    within_tolerance = np.abs((y_pred - y_test) / y_test) <= acc_threshold
    accuracy = np.mean(within_tolerance)
    residuals = y_test - y_pred

    return rmse, mae, r2, accuracy, residuals, y_pred


def __find_best_model(
    train_results: dict,
    operator_output_dir: str,
    n_trials=OPTUNA_TRIALS,
):
    metrics = {
        model_name: result["metrics"] for model_name, result in train_results.items()
    }
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: __objective(trial, metrics),
        n_trials=n_trials,
    )
    best_weights = study.best_params
    metric_results = []

    for model_name, model_metrics in metrics.items():
        score = __calculate_model_score(
            model_metrics["accuracy"],
            model_metrics["rmse"],
            model_metrics["mae"],
            model_metrics["r2"],
            best_weights,
        )
        train_result_data = train_results[model_name]["data"]

        metric_results.append(
            {
                "model_name": model_name,
                "score": score,
                "accuracy": model_metrics["accuracy"],
                "rmse": model_metrics["rmse"],
                "mae": model_metrics["mae"],
                "r2": model_metrics["r2"],
                "residuals": train_result_data["residuals"],
                "y_pred": train_result_data["y_pred"],
                "y_test": train_result_data["y_test"],
                **best_weights,
            }
        )

    df_metrics = pd.DataFrame(metric_results)
    print("Saving metrics...")
    operator_results_dir = f"{operator_output_dir}/results"
    os.makedirs(operator_results_dir, exist_ok=True)
    df_metrics.to_csv(f"{operator_results_dir}/model_metrics.csv", index=False)

    best_model = df_metrics.loc[df_metrics["score"].idxmax()]
    print(f"Best model for {operator_output_dir}: {best_model['model_name']}")
    print(f"  Accuracy: {best_model['accuracy'] * 100}%")
    print(f"  RMSE: {best_model['rmse']}")
    print(f"  MAE: {best_model['mae']}")
    print(f"  R2: {best_model['r2']}")

    return best_model["model_name"], best_weights


def __objective(trial, metrics):
    acc_weight = trial.suggest_float("accuracy_weight", 1.0, 2.0)
    rmse_weight = trial.suggest_float("rmse_weight", 0.5, 1.5)
    mae_weight = trial.suggest_float("mae_weight", 0.5, 1.5)
    r2_weight = trial.suggest_float("r2_weight", 0.2, 1.0)

    total_score = 0

    for model_name, model_metrics in metrics.items():
        score = (
            acc_weight * model_metrics["accuracy"]
            - rmse_weight * model_metrics["rmse"]
            - mae_weight * model_metrics["mae"]
            + r2_weight * model_metrics["r2"]
        )
        total_score += score

    return total_score


def __calculate_model_score(acc, rmse, mae, r2, weights):
    return (
        weights["accuracy_weight"] * acc
        - weights["rmse_weight"] * rmse
        - weights["mae_weight"] * mae
        + weights["r2_weight"] * r2
    )


def __evaluate_data_reduction(
    df_features: pd.DataFrame,
    best_models: dict,
    best_weights: dict,
    min_size: int = 10,
):
    print("---------- STEP 7: Evaluating data reduction")
    operators = df_features["operator"].unique()

    for operator in operators:
        print(f"Evaluating {operator}...")

        df_operator = df_features[df_features["operator"] == operator]
        operator_output_dir = f"{OPERATORS_DIR}/{operator}"
        os.makedirs(operator_output_dir, exist_ok=True)
        operator_data_reduction_results = pd.DataFrame(
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

        operator_model = best_models[operator]
        print(f"Using model {operator_model} for operator {operator}...")
        model = MODELS_HASHMAP[operator_model]
        model_weights = best_weights[operator]

        while len(df_operator) >= min_size:
            train_result = __train_model(
                operator_model,
                model,
                df_operator,
            )
            train_result_metrics = train_result["metrics"]
            train_result_data = train_result["data"]
            score = __calculate_model_score(
                train_result_metrics["accuracy"],
                train_result_metrics["rmse"],
                train_result_metrics["mae"],
                train_result_metrics["r2"],
                model_weights,
            )

            operator_data_reduction_results = pd.concat(
                [
                    operator_data_reduction_results,
                    pd.DataFrame(
                        {
                            "num_samples": [len(df_operator)],
                            "model_name": [operator_model],
                            "rmse": [train_result_metrics["rmse"]],
                            "mae": [train_result_metrics["mae"]],
                            "r2": [train_result_metrics["r2"]],
                            "accuracy": [train_result_metrics["accuracy"]],
                            "residuals": [train_result_data["residuals"]],
                            "y_pred": [train_result_data["y_pred"]],
                            "y_test": [train_result_data["y_test"]],
                            "score": [score],
                        }
                    ),
                ],
                ignore_index=True,
            )

            df_operator = df_operator.sort_values(
                "volume", ascending=False
            ).reset_index(drop=True)
            indices_to_remove = __get_linear_indices(df_operator)
            df_operator = df_operator.drop(index=indices_to_remove).reset_index(
                drop=True
            )

        print(
            f"Finished evaluating data reduction for operator {operator}. Sample data:"
        )
        print(operator_data_reduction_results.head())

        data_reduction_path = f"{operator_output_dir}/results/data_reduction.csv"
        os.makedirs(os.path.dirname(data_reduction_path), exist_ok=True)
        operator_data_reduction_results.to_csv(data_reduction_path, index=False)

    print("Finished evaluating data reduction")
    print()


def __evaluate_feature_selection(
    df_features: pd.DataFrame,
    best_models: dict,
    best_weights: dict,
    min_size: int = 1,
):
    print("---------- STEP 8: Evaluating feature selection")
    operators = df_features["operator"].unique()

    for operator in operators:
        print(f"Evaluating {operator}...")

        df_operator = df_features[df_features["operator"] == operator]
        operator_output_dir = f"{OPERATORS_DIR}/{operator}"
        os.makedirs(operator_output_dir, exist_ok=True)
        operator_feature_selection_results = pd.DataFrame(
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

        operator_model = best_models[operator]
        print(f"Using model {operator_model} for operator {operator}...")
        model = MODELS_HASHMAP[operator_model]
        model_weights = best_weights[operator]
        num_features = len(df_operator.columns)

        while num_features >= min_size:
            train_result = __train_model(
                operator_model,
                model,
                df_operator,
            )
            train_result_metrics = train_result["metrics"]
            train_result_data = train_result["data"]
            score = __calculate_model_score(
                train_result_metrics["accuracy"],
                train_result_metrics["rmse"],
                train_result_metrics["mae"],
                train_result_metrics["r2"],
                model_weights,
            )

            operator_feature_selection_results = pd.concat(
                [
                    operator_feature_selection_results,
                    pd.DataFrame(
                        {
                            "num_features": [len(df_operator.columns)],
                            "selected_features": [
                                list(
                                    set(df_operator.columns.tolist())
                                    - {"operator", "avg_peak_memory_usage"}
                                )
                            ],
                            "model_name": [operator_model],
                            "rmse": [train_result_metrics["rmse"]],
                            "mae": [train_result_metrics["mae"]],
                            "r2": [train_result_metrics["r2"]],
                            "accuracy": [train_result_metrics["accuracy"]],
                            "residuals": [train_result_data["residuals"]],
                            "y_pred": [train_result_data["y_pred"]],
                            "y_test": [train_result_data["y_test"]],
                            "score": [score],
                        }
                    ),
                ],
                ignore_index=True,
            )

            selector = SelectKBest(score_func=f_regression, k=num_features - 1)
            selector.fit_transform(train_result_data["X"], train_result_data["y"])
            selected_columns = [
                "operator",
                "avg_peak_memory_usage",
                *train_result_data["X"].columns[selector.get_support()],
            ]
            df_operator = df_operator[selected_columns]
            num_features = num_features - 1

        print(
            f"Finished evaluating feature selection for operator {operator}. Sample data:"
        )
        print(operator_feature_selection_results.head())

        feature_selection_path = f"{operator_output_dir}/results/feature_selection.csv"
        os.makedirs(os.path.dirname(feature_selection_path), exist_ok=True)
        operator_feature_selection_results.to_csv(feature_selection_path, index=False)

    print("Finished evaluating feature selection")
    print()


def __get_linear_indices(df: pd.DataFrame):
    num_to_remove = int(0.2 * len(df))
    return np.linspace(len(df) // 4, 3 * len(df) // 4, num_to_remove, dtype=int)


def __save_train_result(train_result: dict, output_dir: str):
    print(f"Saving data for model on {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    __save_model_train_data(train_result, output_dir)
    __save_model(train_result, output_dir)


def __save_model_train_data(train_result: dict, output_dir: str):
    print(f"Saving model train data...")

    data = train_result["data"]
    data["X"].to_csv(f"{output_dir}/X.csv")
    data["X_train"].to_csv(f"{output_dir}/X_train.csv")
    data["X_test"].to_csv(f"{output_dir}/X_test.csv")
    data["y"].to_csv(f"{output_dir}/y.csv")
    data["y_train"].to_csv(f"{output_dir}/y_train.csv")
    pd.DataFrame(data["y_test"]).to_csv(f"{output_dir}/y_test.csv")
    pd.DataFrame(data["residuals"]).to_csv(f"{output_dir}/residuals.csv")
    pd.DataFrame(data["y_pred"]).to_csv(f"{output_dir}/y_pred.csv")


def __save_model(train_result: dict, output_dir: str):
    print(f"Saving model for {output_dir}...")
    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(train_result["model"], f)


if __name__ == "__main__":
    main()
