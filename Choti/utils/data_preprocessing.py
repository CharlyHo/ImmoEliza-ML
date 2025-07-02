import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.feature_engineering import (
    add_lat_lon,
    add_cluster_loc,
    add_location_distances,
    drop_lat_lon,
)


def encode_categorical_features(df, feature_list):
    categorical_cols = [
        "province_encoded",
        "type_encoded",
        "subtype_encoded",
        "postCode",
        "location_cluster",
    ]

    cols_to_encode = [col for col in categorical_cols if col in feature_list]

    if cols_to_encode:
        original_cols = set(df.columns)
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        new_cols = [col for col in df.columns if col not in original_cols]
        feature_list = update_feature_list(feature_list, cols_to_encode, new_cols)
    return df, feature_list


def data_preprocessing(df, model, feature_list, target, scale, location):
    df = df.copy()

    if any(col in feature_list for col in ["lat", "lon"]):
        df = add_lat_lon(df)

    if "location_cluster" in feature_list:
        df = add_cluster_loc(df)

    if "distance_from_key_location" in feature_list:
        df = add_location_distances(df)

    if "location" in ["cluster_no_lat_lon", "distance_no_lat_lon", "both_no_lat_lon"]:
        df = drop_lat_lon(df)
        feature_list.pop("lat", None)
        feature_list.pop("lon", None)

    df, feature_list = encode_categorical_features(df, feature_list)

    X = df[feature_list]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index
        )
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test


def update_feature_list(feature_list, cols_to_encode, new_cols):
    updated_features = [col for col in feature_list if col not in cols_to_encode]
    updated_features.extend(new_cols)

    return updated_features


def convert_categorical(X_train, X_test, y_train, y_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_columns = X_train.select_dtypes(include=["object"]).columns

    for col in categorical_columns:
        train_unique = set(X_train[col].dropna().unique())
        test_unique = set(X_test[col].dropna().unique())
        all_categories = list(train_unique.union(test_unique))
        X_train[col] = pd.Categorical(X_train[col], categories=all_categories)
        X_test[col] = pd.Categorical(X_test[col], categories=all_categories)

    return X_train, X_test, y_train, y_test
