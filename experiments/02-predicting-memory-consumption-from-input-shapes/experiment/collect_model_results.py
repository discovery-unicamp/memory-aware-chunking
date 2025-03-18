import json
import os
import pickle
import random

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
    "linear_regression,polynomial_regression,decision_tree,random_forest,gradient_boosting,neural_network,xgboost,gaussian_process,bayesian_ridge",
).split(",")
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "100"))


def main():
    print("Collecting model results...")
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
    best_models = __find_best_models(df_features)
    print(best_models)


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


def __build_dataset(profile_filepaths: list[str], profiles: list) -> list:
    print("---------- STEP 3: Building dataset")
    dataset = []
    for profile_filepath, profile in zip(profile_filepaths, profiles):
        operator, inlines, xlines, samples, session_id = (
            profile_filepath.split("/")[-1].split(".")[0].split("-")
        )
        profiler_data_key = f"{PROFILER}_memory_usage"
        profiler_unit = profile["metadata"][f"{profiler_data_key}_unit"]

        dataset.append(
            {
                "session_id": session_id,
                "operator": operator,
                "inlines": int(inlines),
                "xlines": int(xlines),
                "samples": int(samples),
                "peak_memory_usage": max(
                    profile["data"], key=lambda x: x[profiler_data_key]
                )[profiler_data_key],
                "memory_usage_unit": profiler_unit,
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

    return dataset


def __build_dataframe(dataset: list) -> pd.DataFrame:
    print("---------- STEP 4: Building dataframe")
    df = pd.DataFrame(dataset)
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
    df_features: pd.DataFrame, models_to_evaluate=MODELS_TO_EVALUATE
):
    print("---------- STEP 6: Evaluating models")
    models_hashmap = {
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
        "neural_network": Pipeline(
            [("scaler", StandardScaler()), ("mlp", MLPRegressor())]
        ),
        "xgboost": XGBRegressor(),
        "gaussian_process": GaussianProcessRegressor(),
        "bayesian_ridge": BayesianRidge(),
    }

    invalid_models = [m for m in models_to_evaluate if m not in models_hashmap]
    if invalid_models:
        raise ValueError(f"Invalid models: {invalid_models}")

    operators = df_features["operator"].unique()
    enabled_models = [
        (model_name, models_hashmap[model_name]) for model_name in models_to_evaluate
    ]
    best_models = {operator: None for operator in operators}

    for operator in operators:
        df_operator = df_features[df_features["operator"] == operator]
        operator_output_dir = f"{OPERATORS_DIR}/{operator}"
        os.makedirs(operator_output_dir, exist_ok=True)

        operator_model_metrics = {}
        for model_name, model in enabled_models:
            print(f"Evaluating model for {operator}: {model_name}")
            model_metrics = __collect_metric_for_model(
                model_name,
                model,
                df_operator,
                operator_output_dir,
            )
            operator_model_metrics[model_name] = model_metrics

        best_models[operator] = __find_best_model(
            operator_model_metrics,
            operator_output_dir,
        )

    print("Finished evaluating models")
    print()

    return best_models


def __collect_metric_for_model(
    model_name,
    model,
    df_operator: pd.DataFrame,
    operator_output_dir: str,
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE,
):
    X = df_operator.drop(
        columns=[
            "peak_memory_usage",
            "session_id",
            "memory_usage_unit",
            "operator",
        ]
    )
    y = df_operator["peak_memory_usage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Saving data for {model_name} model...")
    operator_model_output_dir = f"{operator_output_dir}/models/{model_name}"
    os.makedirs(operator_model_output_dir, exist_ok=True)
    X.to_csv(f"{operator_model_output_dir}/X.csv")
    X_train.to_csv(f"{operator_model_output_dir}/X_train.csv")
    X_test.to_csv(f"{operator_model_output_dir}/X_test.csv")
    y.to_csv(f"{operator_model_output_dir}/y.csv")
    y_train.to_csv(f"{operator_model_output_dir}/y_train.csv")
    y_test.to_csv(f"{operator_model_output_dir}/y_test.csv")

    print(f"Training {model_name}")
    model.fit(X_train, y_train)
    with open(f"{operator_model_output_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    rmse, mae, r2, accuracy, residuals, y_pred = __get_model_metrics(
        model,
        X_test,
        y_test,
    )

    print(f"Saving residuals and y_pred for {model_name} model...")
    pd.DataFrame(residuals).to_csv(f"{operator_model_output_dir}/residuals.csv")
    pd.DataFrame(y_pred).to_csv(f"{operator_model_output_dir}/y_pred.csv")

    print(f"Results for {model_name} model:")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")
    print(f"  Accuracy: {accuracy}%")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "accuracy": accuracy,
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
    metrics,
    operator_output_dir: str,
    n_trials=OPTUNA_TRIALS,
):
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

        metric_results.append(
            {
                "model_name": model_name,
                "accuracy": model_metrics["accuracy"],
                "rmse": model_metrics["rmse"],
                "mae": model_metrics["mae"],
                "r2": model_metrics["r2"],
                "score": score,
                **best_weights,
            }
        )

    df_metrics = pd.DataFrame(metric_results)
    print("Saving metrics...")
    operator_results_dir = f"{operator_output_dir}/results"
    os.makedirs(operator_results_dir, exist_ok=True)
    df_metrics.to_csv(f"{operator_results_dir}/metrics.csv")

    best_model = df_metrics.loc[df_metrics["score"].idxmax()]
    print(f"Best model for {operator_output_dir}: {best_model['model_name']}")
    print(f"  Accuracy: {best_model['accuracy']}%")
    print(f"  RMSE: {best_model['rmse']}")
    print(f"  MAE: {best_model['mae']}")
    print(f"  R2: {best_model['r2']}")

    return best_model["model_name"]


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


if __name__ == "__main__":
    main()
