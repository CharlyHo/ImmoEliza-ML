import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import optuna
import joblib

def remove_outliers(df: pd.DataFrame, feature: str, lower_quantile=0.01, upper_quantile=0.99) -> pd.DataFrame:
    """Remove outliers outside specified quantiles for a feature."""
    lower_bound = df[feature].quantile(lower_quantile)
    upper_bound = df[feature].quantile(upper_quantile)
    filtered_df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return filtered_df

def build_preprocessing_pipeline(
    categorical_features: list[str], numerical_features: list[str]
) -> ColumnTransformer:
    """Create preprocessing pipeline for numerical and categorical data."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def log_transform(y: pd.Series) -> np.ndarray:
    """Apply log1p transform."""
    return np.log1p(y)

def inverse_log_transform(y: np.ndarray) -> np.ndarray:
    """Inverse of log1p transform."""
    return np.expm1(y)

def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray, preprocessor: ColumnTransformer):
    """Optuna objective function to minimize RMSE with cross-validation."""
    params = {
    'n_estimators': 3000,  # More trees with early stopping to allow learning longer
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),

    'num_leaves': trial.suggest_int('num_leaves', 31, 256),  # Larger can capture more complexity
    'max_depth': trial.suggest_int('max_depth', -1, 20),     # -1 means no limit

    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),  # Controls overfitting
    'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),  # Sometimes helps with outliers

    'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Random row sampling (bagging)
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Feature sampling

    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),  # L1
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),  # L2

    'random_state': 42,
    'n_jobs': -1,
  }


    model = LGBMRegressor(**params)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    scores = cross_val_score(pipeline, X, y, scoring="neg_root_mean_squared_error", cv=5)
    rmse = -np.mean(scores)
    return float(rmse)

def main():
    filename = r"C:/Users/Becode/immo-eliza-ML/immoEliza-ML/ml_ready_real_estate_data_soft_filled.csv"
    df = pd.read_csv(filename)

    # Remove outliers on 'price'
    df_clean = remove_outliers(df, "price", 0.01, 0.99)

    # Separate features and target
    X = df_clean.drop(columns=["price"])
    y = df_clean["price"]

    # Log-transform the target
    y_log = log_transform(y)

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Split train/test
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(categorical_features, numerical_features)

    # Optimize hyperparameters with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train_log, preprocessor), n_trials=50)
    print("Best hyperparameters:", study.best_params)

    # Train final model with best params
    best_params = study.best_params
    best_params.update({'n_estimators': 1000, 'random_state': 42, 'n_jobs': -1})
    model = LGBMRegressor(**best_params)

    final_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    final_pipeline.fit(X_train, y_train_log)

    # Predict on train and test
    y_train_pred_log = final_pipeline.predict(X_train)
    y_test_pred_log = final_pipeline.predict(X_test)

    # Inverse transform predictions and true targets
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)
    y_train_true = np.expm1(y_train_log)
    y_test_true = np.expm1(y_test_log)



    print("Train MAE:", mean_absolute_error(y_train_true, y_train_pred))
    print("Train RMSE:", mean_squared_error(y_train_true, y_train_pred))
    print("Train R2:", r2_score(y_train_true, y_train_pred))

    print("Test MAE:", mean_absolute_error(y_test_true, y_test_pred))
    print("Test RMSE:", mean_squared_error(y_test_true, y_test_pred))
    print("Test R2:", r2_score(y_test_true, y_test_pred))

    # Save final pipeline (including preprocessing and model)
    joblib.dump(final_pipeline, "immoeliza_lgbm_optimized_pipeline.pkl")
    print("Model pipeline saved as immoeliza_lgbm_optimized_pipeline.pkl")

if __name__ == "__main__":
    main()
