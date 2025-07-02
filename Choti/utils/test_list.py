import pandas as pd
from typing import Dict, Any, List
from utils.feature_engineering import feature_selection


def create_test_list(df: pd.DataFrame, target: str) -> List[Dict[Any, Any]]:
    """crate experimental list"""

    test_list = []

    base_features = df.columns.drop(target).tolist()
    base_features_except_postcode = [
        col for col in base_features if col.lower() != "postcode"
    ]

    model_list = [
        "xgb",
        "Ridge",
        "elastic_net",
        "Lasso",
        "RandomForest",
        "LinearRegression",
        "stacking_ridge_xgb_rf",
        "voting_ridge_xgb_rf",
    ]
    scale_list = [False, True]
    location_list = [
        "none",
        "only_lat_lon",
        "cluster",
        "distance",
        "all",
        "cluster_no_lat_lon",
        "distance_no_lat_lon",
        "both_no_lat_lon",
    ]

    # location_list = ["only_lat_lon"]
    # model_list = ["elastic_net"]
    # model_list = ["Ridge"]
    # model_list = ["xgb"]
    # model_list = ["Lasso"]
    # model_list = ["RandomForest"]
    # model_list = ["LinearRegression"]
    # model_list = ["stacking_ridge_xgb_rf"]
    # model_list = ["voting_ridge_xgb_rf"]
    # top_features_linear = feature_selection(df, 10, target, "LinearRegression")
    # top_features_randomforest = feature_selection(df, 10, target, "RandomForest")
    # top_features_xgb = feature_selection(df, 10, target, "xgb")

    # Best features obtained from feature_selection() function
    # add result here to avoid running time-consuming hyperparameter tuning
    top_features_linear = [
        "bedroomCount",
        "toiletCount",
        "hasArmoredDoor_encoded",
        "hasVisiophone_encoded",
        "hasOffice_encoded",
        "hasSwimmingPool_encoded",
        "hasFireplace_encoded",
        "hasDressingRoom_encoded",
        "hasHeatPump_encoded",
        "hasPhotovoltaicPanels_encoded",
    ]
    top_features_randomforest = [
        "bathroomCount",
        "habitableSurface",
        "terraceSurface",
        "postCode",
        "province_encoded",
        "epcScore_encoded",
    ]
    top_features_xgb = [
        "bathroomCount",
        "habitableSurface",
        "postCode",
        "province_encoded",
        "type_encoded",
        "epcScore_encoded",
        "hasLift_encoded",
        "hasHeatPump_encoded",
        "hasPhotovoltaicPanels_encoded",
    ]

    features_list = {
        "all_feature": base_features,
        "all_except_postcode": base_features_except_postcode,
        "top_features": [],
    }

    for model in model_list:
        for k, v in features_list.items():

            features_base = []
            if k == "top_features":
                if model in ["LinearRegression", "Ridge", "Lasso", "elastic_net"]:
                    features_base = top_features_linear
                elif model in ["RandomForest"]:
                    features_base = top_features_randomforest
                elif model in ["xgb", "stacking_ridge_xgb_rf", "voting_ridge_xgb_rf"]:
                    features_base = top_features_xgb
            else:
                features_base = v

            for scale in scale_list:
                for location in location_list:
                    features = (
                        features_base.copy()
                        if isinstance(features_base, list)
                        else list(features_base)
                    )
                    if location != "none":
                        if "lat" not in features:
                            features.append("lat")
                        if "lon" not in features:
                            features.append("lon")
                    if location in ["cluster", "both"]:
                        features.append("location_cluster")
                    if location in ["distance", "both"]:
                        features.append("distance_from_key_location")

                    features = list(set(features))

                    test_list.append(
                        {
                            "desc": f"{model}_{k}_{'scaled' if scale else 'unscaled'}_loc_{location}_",
                            "model": model,
                            "features": features,
                            "scale": scale,
                            "location": location,
                        }
                    )

    return test_list
