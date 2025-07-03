import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


os.makedirs("ML-Clean", exist_ok=True)

# Transofrmation des colonnes à info en lettre pour des chiffres
def map_postcode_to_province_code(postcode: int) -> int:
    try:
        pc = int(postcode)
        if 1000 <= pc <= 1299:
            return 11  # Bruxelles
        elif 2000 <= pc <= 2999:
            return 1   # Anvers
        elif 3500 <= pc <= 3999:
            return 2   # Limbourg
        elif 9000 <= pc <= 9999:
            return 3   # Flandre orientale
        elif 8000 <= pc <= 8999:
            return 4   # Flandre occidentale
        elif 1500 <= pc <= 1999:
            return 5   # Brabant flamand
        elif 1300 <= pc <= 1499:
            return 6   # Brabant wallon
        elif 6000 <= pc <= 6599 or 7000 <= pc <= 7999:
            return 7   # Hainaut
        elif 4000 <= pc <= 4999:
            return 8   # Liège
        elif 6600 <= pc <= 6999:
            return 9   # Luxembourg
        elif 5000 <= pc <= 5999:
            return 10  # Namur
        else:
            return -1  # inconnu
    except:
        return -1
    
# Mapping type
def map_property_type(property_type: str) -> int:
    if isinstance(property_type, str):
        if property_type.lower() == "house":
            return 1
        elif property_type.lower() == "apartment":
            return 2
    return -1  # inconnu ou autre


# Mapping des subtypes
subtype_mapping = {
    "APARTMENT": 1,
    "HOUSE": 2,
    "FLAT_STUDIO": 3,
    "DUPLEX": 4,
    "PENTHOUSE": 5,
    "GROUND_FLOOR": 6,
    "APARTMENT_BLOCK": 7,
    "MANSION": 8,
    "EXCEPTIONAL_PROPERTY": 9,
    "MIXED_USE_BUILDING": 10,
    "TRIPLEX": 11,
    "LOFT": 12,
    "VILLA": 13,
    "TOWN_HOUSE": 14,
    "CHALET": 15,
    "MANOR_HOUSE": 16,
    "SERVICE_FLAT": 17,
    "KOT": 18,
    "FARMHOUSE": 19,
    "BUNGALOW": 20,
    "COUNTRY_COTTAGE": 21,
    "OTHER_PROPERTY": 22,
    "CASTLE": 23,
    "PAVILION": 24
}


def simplify_postcode_data(input_csv: str = "ML-IncludingMissing/data/data_cleanned.csv") -> pd.DataFrame:
    """
    Charge le CSV cleaned, remplace la colonne province par un identifiant numérique
    et sauvegarde dans un nouveau CSV sous ML-Clean.
    """
    # Chargement
    df = pd.read_csv(input_csv)

    # Remplacement de province par le code
    df["province"] = df["postCode"].apply(map_postcode_to_province_code)

    # Remplacement type de bien
    df["type"] = df["type"].apply(map_property_type)

    # Remplacer le subtype par un code numérique
    df["subtype"] = df["subtype"].map(subtype_mapping).fillna(-1).astype(int)

    # Supprimer la colonne locality si elle existe
    if "locality" in df.columns:
        df = df.drop(columns=["locality"])

    # Sauvegarde
    clean_output_csv = "ML-IncludingMissing/data/data_cleanned_ML.csv"
    df.to_csv(clean_output_csv, index=False)
    print(f"✅ CSV sauvegardé sous : {clean_output_csv}")

    return df

simplify_postcode_data()



# Nettoyage
def clean_features(X: pd.DataFrame, y: pd.Series, 
                   constant_thresh: float = 0.98, 
                   low_corr_thresh: float = 0.05, 
                   high_corr_thresh: float = 0.85) -> pd.DataFrame:
    # Colonnes quasi-constantes
    quasi_constant_features = []
    for col in X.columns:
        dominant = X[col].value_counts(normalize=True).max()
        if dominant > constant_thresh:
            quasi_constant_features.append(col)

    print(f"🧹 Quasi-constantes supprimées : {quasi_constant_features}")
    X_cleaned = X.drop(columns=quasi_constant_features)

    # Corrélation faible avec la target
    corr_matrix = X_cleaned.copy()
    corr_matrix["price"] = y
    correlations = corr_matrix.corr()["price"].abs()
    low_corr = correlations[correlations < low_corr_thresh].index.tolist()
    if "price" in low_corr:
        low_corr.remove("price")
    print(f"🧹 Corrélation trop faible avec la target supprimée : {low_corr}")
    X_cleaned = X_cleaned.drop(columns=low_corr)

    # Colonnes trop corrélées entre elles
    corr = X_cleaned.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > high_corr_thresh)]
    print(f"🧹 Colonnes multicolinéaires supprimées : {high_corr_cols}")
    X_cleaned = X_cleaned.drop(columns=high_corr_cols)


    return X_cleaned


if __name__ == "__main__":
    # Charger les données
    data = pd.read_csv("ML-IncludingMissing/data/data_cleanned_ML.csv")

    # target
    y = data["price"]
    X = data.drop(columns=["price"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # nettoyage des features
    X_train_cleaned = clean_features(X_train, y_train)
    X_test_cleaned  = X_test[X_train_cleaned.columns]

    # EDA rapide
    train_data = X_train_cleaned.join(y_train)
    train_data.hist(figsize=(16, 10))
    plt.show()

    plt.figure(figsize=(16, 10))
    sns.heatmap(train_data.corr(), annot=False, cmap="coolwarm")
    plt.show()

    # recalcul features
    num_features = X_train_cleaned.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_features = X_train_cleaned.select_dtypes(include=["object", "category"]).columns.tolist()

    # pipeline
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", "passthrough", cat_features)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("xgb", XGBRegressor(objective="reg:squarederror", random_state=42))
    ])

    param_grid = {
        "xgb__n_estimators": [100, 200, 300],
        "xgb__max_depth": [8, 12, 16],
        "xgb__learning_rate": [0.1, 0.2, 0.3],
        "xgb__subsample": [0.6, 0.8, 1],
        "xgb__colsample_bytree": [0.5, 0.8, 1],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train_cleaned, y_train)

    best_model = grid_search.best_estimator_
    print(f"✅ Best XGB params: {grid_search.best_params_}")

    y_train_pred = best_model.predict(X_train_cleaned)
    y_test_pred  = best_model.predict(X_test_cleaned)

    print(f"XGBoost R2 train: {r2_score(y_train, y_train_pred):.3f}")
    print(f"XGBoost R2 test:  {r2_score(y_test, y_test_pred):.3f}")
    print(f"XGBoost MSE test: {mean_squared_error(y_test, y_test_pred):.2f}")

    joblib.dump(best_model, "best_xgb_model.joblib")
    print("✅ Modèle sauvegardé sous best_xgb_model.joblib")