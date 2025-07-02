import os
import pandas as pd
from sklearn.cluster import KMeans
from haversine import haversine
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import List
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb


def drop_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["lat", "lon"], axis=1)
    return df

def add_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    df["postCode"] = df["postCode"].astype(str)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "georef-belgium-postal-codes.csv")
    geo_df = pd.read_csv(data_path, delimiter=";")
    geo_df[["lat", "lon"]] = geo_df["Geo Point"].str.split(",", expand=True)
    geo_df["lat"] = geo_df["lat"].astype(float)
    geo_df["lon"] = geo_df["lon"].astype(float)
    geo_df["postCode"] = geo_df["Post code"].astype(str)
    df = df.merge(geo_df[["postCode", "lat", "lon"]], on="postCode", how="left")
    return df


def add_cluster_loc(df: pd.DataFrame) -> pd.DataFrame:
    coords = df[["lat", "lon"]]
    kmeans = KMeans(n_clusters=10, random_state=42)
    df["location_cluster"] = kmeans.fit_predict(coords)
    return df


def add_location_distances(df: pd.DataFrame) -> pd.DataFrame:
    """add distance columns by using lat, lon to get distance to key location in Belgium"""
    df["lat_lon"] = list(zip(df["lat"], df["lon"]))

    # Coordinates for Brussels and Knokke-Heist
    central_brussels = (50.8465573, 4.351697)
    knokke_heist = (51.346352, 3.285860)

    brussels_distances = []
    knokke_distances = []
    for coord in df["lat_lon"]:
        brussels_distances.append(haversine(central_brussels, coord))
        knokke_distances.append(haversine(knokke_heist, coord))

    df["distance_from_brussels"] = brussels_distances
    df["distance_from_knokke"] = knokke_distances

    df["distance_from_key_location"] = df[
        ["distance_from_brussels", "distance_from_knokke"]
    ].min(axis=1)

    df = df.drop(columns="lat_lon")
    return df


def feature_selection(
    df: pd.DataFrame, n_features: int, target: str, model_name: str
) -> List:
    X = df.drop(target, axis=1)
    y = df[target]

    if model_name in ["LinearRegression", "Ridge", "Lasso"]:
        estimator = LinearRegression()
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        top_features = X.columns[selector.support_]

    elif model_name == "RandomForest":
        estimator = RandomForestRegressor(random_state=42)
        selector = SelectFromModel(estimator, threshold="mean")
        selector.fit(X, y)
        top_features = X.columns[selector.get_support()]

    elif model_name == "xgb":
        model = xgb.XGBRegressor()
        selector = SelectFromModel(model, prefit=True, threshold="mean")
        model.fit(X, y)
        top_features = X.columns[selector.get_support()]

    print(f"top {n_features} features for {model_name} :", list(top_features))
    return list(top_features)
