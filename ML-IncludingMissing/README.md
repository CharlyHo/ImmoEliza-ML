# 🏠 ImmoEliza Price Prediction - Machine Learning Pipeline

## 📌 Objective

Build a robust machine learning model to predict real estate prices from Belgian property listings, following a rigorous process of cleaning, preparation, and modeling.

---

## 🚀 Project Steps
## 1️⃣ Data Collection & Initial Cleaning

    Scraping of raw real estate data

    Manual cleaning (removing incomplete listings and duplicates)

    Saved to data_cleanned.csv

## 2️⃣ Encoding Categorical Variables

    Transforming province using the postal code via map_postcode_to_province_code

    Transforming type (house/apartment) into a numerical variable

    Transforming subtype into numeric codes using a dictionary

    Dropping the locality column (not relevant for modeling)

    Result saved to:
    ML-IncludingMissing/data/data_cleanned_ML.csv

## 3️⃣ Feature Cleaning

    Removing quasi-constant columns

    Removing columns with too low correlation with the target

    Removing multicollinear columns (strongly correlated with each other)

    Goal: reduce dimensionality and avoid overfitting

## 4️⃣ Train/Test Split

    80% / 20% split using train_test_split

    Visualization:

        histograms

        correlation heatmap

    Study of variable distributions

## 5️⃣ Preprocessing (Pipeline)

    Scikit-learn pipeline including:

        SimpleImputer (handling missing values)

        StandardScaler (scaling features)

    Integrated into a ColumnTransformer to handle both numeric and categorical variables

    Added an XGBoost model as the final estimator

## 6️⃣ Hyperparameter Optimization

    GridSearchCV with 5-fold cross-validation

    Grid search on:

        n_estimators

        max_depth

        learning_rate

        subsample

        colsample_bytree

    Scoring based on negative MSE

## 7️⃣ Final Training & Saving

    Training the best pipeline

    Performance evaluation:

        R2 train/test

        MSE test

    Model saved with joblib to:
    best_xgb_model.joblib

## 📊 Observed Results

 	Quasi-constantes supprimées : ['hasAirConditioning']
    Corrélation trop faible avec la target supprimée : ['province', 'postCode', 'hasAttic', 'hasBasement', 'diningRoomSurface', 'floorCount', 'streetFacadeWidth', 'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels', 'kitchenSurface', 'hasLivingRoom', 'livingRoomSurface', 'hasGarden', 'gardenSurface', 'parkingCountIndoor', 'parkingCountOutdoor', 'buildingConditionNormalize']
    Colonnes multicolinéaires supprimées : []

    XGBoost (optimized)

        R2 train: 0.968

        R2 test: 0.811

        MSE test: 45,577,460,004

    Random Forest (optimized)

        R2 train: 0.970

        R2 test: 0.778

The performance shows an excellent bias-variance trade-off, with XGBoost remaining the best.


## 📌 Installation

dl ML.py, data folder and install package from requirements
run ML.py
