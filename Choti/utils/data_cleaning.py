import numpy as np
import pandas as pd
import os


def load_and_clean_data(data_path) -> pd.DataFrame:
    """Load and clean the dataset"""
    df = pd.read_csv(data_path)
    df_clean = fill_epc_by_subtype_mode(df)

    return df_clean


def fill_epc_by_subtype_mode(df: pd.DataFrame) -> pd.DataFrame:
    epc_mode = df.groupby("subtype_encoded")["epcScore_encoded"].agg(
        lambda x: x.mode().values[0]
    )
    df["epcScore_encoded"] = df["epcScore_encoded"].fillna(
        df["subtype_encoded"].map(epc_mode)
    )
    return df

