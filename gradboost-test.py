import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load data
filename = r"C:/Users/Becode/immo-eliza-ML/immoEliza-ML/ml_ready_real_estate_data_soft_filled.csv"
df = pd.read_csv(filename)

# Target variable (use log to reduce skew)
df = df[df['price'] > 0]  # remove non-positive prices
y = np.log1p(df['price'])
X = df.drop(columns=['price'])

# Identify categorical columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

# Split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Preprocess inputs
X_train_transformed = preprocessor.fit_transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)
X_test_transformed = preprocessor.transform(X_test)

# Step: Remove outliers from key numerical features
def remove_feature_outliers(df, features, multiplier=2):
    for col in features:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Choose the most relevant numeric features to clean
outlier_features = ['area', 'kitchen_surface', 'terrace_surface', 'bedroom_count', 'facade_count']

# Apply outlier removal
df = remove_feature_outliers(df, outlier_features)

# Ensure price > 0 and set target
df = df[df['price'] > 0]
y = np.log1p(df['price'])
X = df.drop(columns=['price'])


def objective(trial):
    params = {
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_valid_transformed, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100)]
    )
    preds = model.predict(X_valid_transformed)
    preds = np.asarray(preds).ravel()
    rmse = mean_squared_error(y_valid, preds)
    return rmse

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best trial:")
print(study.best_trial.params)

# Train final model with best params
best_params = study.best_trial.params
best_params['n_estimators'] = 3000
best_params['random_state'] = 42
best_params['n_jobs'] = -1

final_model = lgb.LGBMRegressor(**best_params)
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", final_model)
])

final_pipeline.fit(X_train_full, y_train_full)

# Evaluate
def inverse_log_transform(y):
    return np.expm1(y)

y_train_pred = inverse_log_transform(final_pipeline.predict(X_train_full))
y_test_pred = inverse_log_transform(final_pipeline.predict(X_test))
y_train_true = inverse_log_transform(y_train_full)
y_test_true = inverse_log_transform(y_test)

print("\nTrain MAE:", mean_absolute_error(y_train_true, y_train_pred))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
print("Train R2:", r2_score(y_train_true, y_train_pred))
print("Test MAE:", mean_absolute_error(y_test_true, y_test_pred))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
print("Test R2:", r2_score(y_test_true, y_test_pred))

# Save the pipeline
joblib.dump(final_pipeline, "immoeliza_lgbm_optimized_pipeline.pkl")
print("Model pipeline saved as immoeliza_lgbm_optimized_pipeline.pkl")

# adding feauture importance to reduce unnecessary features and reduce noise
import matplotlib.pyplot as plt
import seaborn as sns

# Extract fitted preprocessor from pipeline
fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
fitted_model = final_pipeline.named_steps["model"]

# Get feature names safely from fitted transformers
def get_feature_names_from_fitted_preprocessor(preprocessor):
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            ohe = transformer
            try:
                ohe_feature_names = ohe.get_feature_names_out(columns)
                feature_names.extend(ohe_feature_names)
            except:
                feature_names.extend(columns)  # fallback if get_feature_names_out isn't supported
    return feature_names

# Get feature names
feature_names = get_feature_names_from_fitted_preprocessor(fitted_preprocessor)

# Get feature importances
importances = fitted_model.feature_importances_

# Create DataFrame and plot
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Plot top 30
plt.figure(figsize=(10, 12))
sns.barplot(data=feature_importance_df.head(30), x='importance', y='feature')
plt.title("Top 30 Feature Importances (LightGBM)")
plt.tight_layout()
plt.show()

# Show least important features
low_importance = feature_importance_df[feature_importance_df['importance'] < 5]
print("\n⚠️ Features with very low importance (consider dropping):")
print(low_importance[['feature', 'importance']])
