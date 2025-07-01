import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from utils.test import test_model
import mlflow
from utils.data_preprocessing import data_preprocessing
from utils.output import plot_scatter, display_result, plot_trees
from pathlib import Path
from typing import Dict, Any, List
import xgboost as xgb
from sklearn.linear_model import ElasticNet


def tune_model(model_name, X_train, y_train) -> dict:
    """Run hyperparameter tuning once and return best parameters for each model"""
    model_map = {
        "xgb": xgb.XGBRegressor(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=5000),
        "RandomForest": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
    }

    param_grids = {
        "xgb": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "colsample_bytree": [0.3, 0.7],
            "subsample": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 1.5, 2],
        },
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "RandomForest": {
    "n_estimators": [100, 250, 500],
    'max_depth': [5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 3, 5, 10],
    "max_features": ["sqrt", "log2", None],
},

        "LinearRegression": {"fit_intercept": [True, False], "positive": [True, False]},
    }

    model = model_map[model_name]
    param_grid = param_grids[model_name]

    print(f"Tuning hyperparameters for {model_name}...")
    # grid_search = GridSearchCV(
    #     model,
    #     param_grid,
    #     cv=3,
    #     scoring='r2',
    #     n_jobs=-1,
    #     verbose=1
    # )
    # grid_search.fit(X_train, y_train)

    random_search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=50, 
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best CV score for {model_name}: {random_search.best_score_}")

    return random_search.best_params_


def run_experiments(
    test_configs: List[Dict], df: pd.DataFrame, target: str, plot_dir: Path
):
    """Run all experiments"""
    models = set(test["model"] for test in test_configs)
    feature_list = set()
    for test in test_configs:
        feature_list.update(test["features"])
    feature_list = list(feature_list)

    # Best parameters obtained from tune_model() function
    # add result here to avoid running time-consuming hyperparameter tuning
    all_best_params = {
        "Ridge": {"alpha": 10.0},
        "LinearRegression": {"fit_intercept": True, "positive": False},
        # "RandomForest": {'n_estimators': 500, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': None},
        # "xgb" : {'subsample': 0.8, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.7}
    }
    for model_name in models:
        if model_name in ["xgb", "RandomForest"]:
            df_tune = df.copy()
            X_train, X_test, y_train, y_test = data_preprocessing(
                df_tune,
                model=model_name,
                feature_list=feature_list,
                target=target,
                scale=True,
                location="all",
            )
            best_param = tune_model(model_name, X_train, y_train)
            all_best_params[model_name] = best_param

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

        # Prepare data
        df_copy = df.copy()
        X_train, X_test, y_train, y_test = data_preprocessing(
            df_copy,
            model=test_config["model"],
            feature_list=test_config["features"],
            target=target,
            scale=test_config["scale"],
            location=test_config["location"],
        )

        # Train and test model
        result = run_train_test(
            test_config["model"], X_train, X_test, y_train, y_test, all_best_params, test_config["features"], plot_dir=str(plot_dir))

        # Process results
        result["title"] = test_config["desc"]

        metrics = ["train_R2", "train_MAE", "train_RMSE", "test_R2", "test_MAE", "test_RMSE", "average_target"]
        for metric in metrics:
            if metric in result:
                mlflow.log_metric(metric, result[metric])


        plot_path = plot_dir / f"{test_config['desc']}.png"
        plot_scatter(y_test, result["y_pred"], str(plot_path), result["title"])
        mlflow.log_artifact(str(plot_path))
        df_result = display_result(result)
        print(df_result)

        return result


def run_train_test(
    model_name: str, X_train, X_test, y_train, y_test, all_best_params: Dict[str, dict], features, plot_dir
) -> dict:
    """Train model with pre-determined best parameters"""

    best_params = all_best_params.get(model_name, {})
    if model_name == "Ridge":
        model = Ridge(random_state=42, **best_params)
    elif model_name == "LinearRegression":
        valid_params = ["fit_intercept", "positive", "copy_X"]
        lr_params = {k: v for k, v in best_params.items() if k in valid_params}
        model = LinearRegression(**lr_params)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(random_state=42, **best_params)
    elif model_name == "xgb":
        X_train, X_test, y_train, y_test = convert_categorical(
            X_train, X_test, y_train, y_test
        )
        model = xgb.XGBRegressor(
            random_state=42, enable_categorical=True, **best_params
        )
    elif model_name == "elastic_net":
        model = ElasticNet(alpha=0.5, l1_ratio=0.7)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_result = test_model(y_train, y_train_pred)
    test_result = test_model(y_test, y_test_pred)
    
    if model_name == "RandomForest":
        plot_trees(model, features, plot_dir)

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


def convert_categorical(X_train, X_test, y_train, y_test):
    categorical_columns = X_train.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    return X_train, X_test, y_train, y_test


