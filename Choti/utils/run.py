import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
from utils.test import test_model
import mlflow
from utils.data_preprocessing import data_preprocessing
from utils.output import plot_scatter, display_result, plot_trees
from pathlib import Path
from typing import Dict, Any, List
import xgboost as xgb
from sklearn.linear_model import ElasticNet


def run_experiments(
    test_configs: List[Dict], df: pd.DataFrame, target: str, plot_dir: Path
):
    """Run all experiments"""
    models = set(test["model"] for test in test_configs)
    feature_list = set()
    for test in test_configs:
        feature_list.update(test["features"])
    feature_list = list(feature_list)

    # Best parameters obtained from tune_model() function in notebook
    # add result here to avoid running time-consuming hyperparameter tuning
    all_best_params = {
        "Ridge": {"alpha": 0.23070995965139646},
        "LinearRegression": {"fit_intercept": True, "positive": False},
        "elastic_net": {"alpha": 0.31313498869576645, "l1_ratio": 0.31771158458416315},
        "RandomForest": {
            "n_estimators": 306,
            "max_depth": 5,
            "min_samples_split": 17,
            "min_samples_leaf": 6,
            "max_features": None,
        },
        "xgb": {
            "reg_lambda": 1,
            "max_depth": 7,
            "min_child_weight": 5,
            "learning_rate": 0.1,
            "n_estimators": 400,
            "colsample_bytree": 0.3,
            "reg_alpha": 0.1,
            "subsample": 0.6,
        },
        # "xgb": {'n_estimators': 147, 'max_depth': 5, 'learning_rate': 0.1944747457273404, 'colsample_bytree': 0.5691418204270269, 'subsample': 0.9587355115786342, 'reg_alpha': 0.3921344320588479, 'reg_lambda': 1.738191941240725},
        "Lasso": {"alpha": 0.20606372000861217},
    }

    results = []
    for test in test_configs:
        result = run_single_experiment(test, df, target, plot_dir, all_best_params)
        results.append(result)

    return results


def run_single_experiment(
    test_config: Dict[Any, Any],
    df: pd.DataFrame,
    target: str,
    plot_dir: Path,
    all_best_params,
) -> dict[Any, Any]:
    """Run a single ML experiment"""
    with mlflow.start_run(run_name=test_config["desc"]):
        # Log parameters
        mlflow.log_param("model", test_config["model"])
        mlflow.log_param("features", ",".join(test_config["features"]))
        mlflow.log_param("location", test_config["location"])
        mlflow.log_param("scale", test_config["scale"])

        params = all_best_params.get(test_config["model"], {})
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        df_copy = df.copy()
        X_train, X_test, y_train, y_test = data_preprocessing(
            df_copy,
            model=test_config["model"],
            feature_list=test_config["features"],
            target=target,
            scale=test_config["scale"],
            location=test_config["location"],
        )

        result = run_train_test(
            test_config["model"],
            X_train,
            X_test,
            y_train,
            y_test,
            all_best_params,
            test_config["features"],
            plot_dir=str(plot_dir),
        )

        result["title"] = test_config["desc"]

        metrics = [
            "train_R2",
            "train_MAE",
            "train_RMSE",
            "test_R2",
            "test_MAE",
            "test_RMSE",
            "average_target",
        ]
        for metric in metrics:
            if metric in result:
                mlflow.log_metric(metric, result[metric])

        train_r2 = result.get("train_R2")
        test_r2 = result.get("test_R2")
        if train_r2 is not None and test_r2 is not None:
            r2_gap = abs(train_r2 - test_r2)
            mlflow.log_metric("r2_train_test_gap", r2_gap)

        plot_path = plot_dir / f"{test_config['desc']}.png"
        plot_scatter(y_test, result["y_pred"], str(plot_path), result["title"])
        mlflow.log_artifact(str(plot_path))
        df_result = display_result(result)
        print(df_result)

        return result


def run_train_test(
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    all_best_params: Dict[str, dict],
    features,
    plot_dir,
) -> dict:
    """Train model with pre-determined best parameters"""

    best_params = all_best_params.get(model_name, {})
    if model_name == "Ridge":
        model = Ridge(**best_params)
    elif model_name == "LinearRegression":
        valid_params = ["fit_intercept", "positive", "copy_X"]
        lr_params = {k: v for k, v in best_params.items() if k in valid_params}
        model = LinearRegression(**lr_params)
    elif model_name == "Lasso":
        model = Lasso(max_iter=5000, **best_params)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(random_state=42, **best_params)
    elif model_name == "xgb":
        model = xgb.XGBRegressor(random_state=42, **best_params)
    elif model_name == "elastic_net":
        model = ElasticNet(**best_params)
    elif model_name.startswith("voting"):
        model = create_voting_regressor(all_best_params)
    elif model_name.startswith("stacking"):
        model = create_stacking_regressor(all_best_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_result = test_model(y_train, y_train_pred)
    test_result = test_model(y_test, y_test_pred)

    result = {
        "model": model_name,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": (
            X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
        ),
        "train_R2": train_result["R2"],
        "train_MAE": train_result["MAE"],
        "train_RMSE": train_result["RMSE"],
        "test_R2": test_result["R2"],
        "test_MAE": test_result["MAE"],
        "test_RMSE": test_result["RMSE"],
        "average_target": test_result["Average_Target"],
        "y_test": y_test,
        "y_pred": y_test_pred,
    }

    return result


def create_voting_regressor(all_best_params: Dict = None):
    """Create voting regressor with base models"""

    ridge_params = all_best_params.get("Ridge", {}) if all_best_params else {}
    rf_params = all_best_params.get("RandomForest", {}) if all_best_params else {}
    xgb_params = all_best_params.get("xgb", {}) if all_best_params else {}

    base_models = [
        ("ridge", Ridge(random_state=42, **ridge_params)),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)),
        ("xgb", xgb.XGBRegressor(random_state=42, **xgb_params)),
    ]

    model = VotingRegressor(estimators=base_models, n_jobs=-1)

    return model


def create_stacking_regressor(all_best_params: Dict = None):
    """Create voting regressor with base models"""

    ridge_params = all_best_params.get("Ridge", {}) if all_best_params else {}
    rf_params = all_best_params.get("RandomForest", {}) if all_best_params else {}
    xgb_params = all_best_params.get("xgb", {}) if all_best_params else {}

    base_models = [
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)),
        ("xgb", xgb.XGBRegressor(random_state=42, **xgb_params)),
    ]
    meta_model = Ridge(random_state=42, **ridge_params)

    model = StackingRegressor(
        estimators=base_models, final_estimator=meta_model, passthrough=False
    )

    return model
