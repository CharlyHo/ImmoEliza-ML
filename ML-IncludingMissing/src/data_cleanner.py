import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

class DataCleanner:
    """
    A data cleaning utility class for loading, analyzing, cleaning, normalizing,
    and exporting real estate property datasets.

    Supports multiple file formats and provides functions for data quality analysis,
    duplicate removal, error correction, normalization of categorical features,
    and output file export.
    """

    def __init__(self, data_file_path: str) -> None:
        """
        Initialize the DataCleanner with the path to the data file.

        Args:
            data_file_path (str): Path to the input data file.
        """
        self.data_file_path = data_file_path

    def load_data_file(self) -> pd.DataFrame:
        """
        Load a data file into a pandas DataFrame.

        Supports CSV, JSON, Excel, Parquet, TXT (tab-delimited), and XML formats.
        Handles missing or empty files gracefully.

        Returns:
            pd.DataFrame: Loaded data or empty DataFrame on failure.
        """
        if not os.path.exists(self.data_file_path) or os.path.getsize(self.data_file_path) == 0:
            print(f"[WARNING] File is missing or empty: {self.data_file_path}")
            return pd.DataFrame()

        suffix = Path(self.data_file_path).suffix.lower()

        try:
            match suffix:
                case ".csv":
                    df = pd.read_csv(self.data_file_path)
                case ".json":
                    df = pd.read_json(self.data_file_path)
                case ".xls" | ".xlsx":
                    df = pd.read_excel(self.data_file_path)
                case ".parquet":
                    df = pd.read_parquet(self.data_file_path)
                case ".txt":
                    df = pd.read_csv(self.data_file_path, delimiter="\t")  # Or adjust delimiter
                case ".xml":
                    df = pd.read_xml(self.data_file_path)
                case _:
                    print(f"[ERROR] Unsupported file format: {suffix}")
                    return pd.DataFrame()

            if df.empty:
                print(f"[WARNING] File loaded but contains no data: {self.data_file_path}")
                return pd.DataFrame()

            print(f"[INFO] Loaded {self.data_file_path} ({len(df)} rows)")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to read {self.data_file_path}: {e}")
            return pd.DataFrame()
    
    def analyze_data_quality(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the data quality of a DataFrame.

        Computes and returns a summary including:
        - Data types
        - Non-null counts
        - Missing value counts and percentages
        - Number of unique values per column

        Args:
            df (pd.DataFrame): DataFrame to analyze.

        Returns:
            pd.DataFrame: Summary of data quality metrics per column.
        """
        if df.empty:
            print("[WARNING] The DataFrame is empty.")
            return pd.DataFrame()

        summary = pd.DataFrame({
            "Data type": df.dtypes,
            "Non-null count": df.notnull().sum(),
            "Missing count": df.isnull().sum(),
            "Missing %": df.isnull().mean() * 100,
            "Unique values": df.nunique()
        })

        summary = summary.sort_values(by="Missing %", ascending=False)
        return summary

    def clean_duplicates(self) -> pd.DataFrame:
        """
        Clean the dataset by removing duplicate rows and dropping irrelevant columns.

        Also prints data quality metrics before and after cleaning.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """

        # Step 1: Load the raw dataset
        df = self.load_data_file()
        if df.empty:
            print("[WARNING] Loaded DataFrame is empty.")
            return pd.DataFrame()

        print("📊 Data Quality BEFORE cleaning:")
        summary_before = self.analyze_data_quality(df)
        print(summary_before)

        # Step 2: Remove exact duplicates
        cleaned_df = df.drop_duplicates()

        # Step 3: Drop irrelevant or problematic columns (if they exist)
        columns_to_drop = [
            "monthlyCost",
            "accessibleDisabledPeople",
            "hasBalcony",
            "url",
            "Unnamed: 0",
            "id",
            "terraceOrientation",
            "gardenOrientation"
        ]
        cleaned_df.drop(columns=[col for col in columns_to_drop if col in cleaned_df.columns], inplace=True)

        # Step 4: Show data quality summary after cleaning
        print("\n📊 Data Quality AFTER cleaning:")
        summary_after = self.analyze_data_quality(cleaned_df)
        print(summary_after)

        return cleaned_df
        
    def clean_errors(self) -> pd.DataFrame:
        """
        Perform error correction and standardization on the dataset.

        - Standardizes 'locality' to uppercase and strips whitespace.
        - Normalizes locality names per postal code using the most frequent value.
        - Drops 'streetFacadeWidth' column (due to excessive missing data).
        - Drops rows missing a price value.
        - Converts specified boolean-like columns to integers.
        - Strips whitespace and fills missing values in specified string columns.

        Returns:
            pd.DataFrame: Cleaned and standardized DataFrame.
        """
        df = self.clean_duplicates()
        # Step 1: Normalize text
        if "locality" in df.columns:
            df["locality"] = df["locality"].astype(str).str.upper().str.strip()

        # Step 2:  Replace each locality with the most frequent locality for the same postal code.
        # Compute the most frequent locality for each postalCode
        most_common_locality = (
            df.groupby("postCode")["locality"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .to_dict()
        )

        # Replace all localities by the most frequent one per postalCode
        df["locality"] = df["postCode"].map(most_common_locality)

        print("[INFO] Localities standardized based on most frequent value per postal code.")

        # drop streetFacadeWidth why? >80% are empty and there's no logical value that we can put in
        df.drop("streetFacadeWidth", axis=1)

        # drop rows where price is not mentioned : 2.737629 % of proprities without price
        df = df.dropna(subset=["price"])

        # Convert column types safely, replacing invalid or NaN entries where necessary.
        int_cols = [
            "hasAirConditioning", "hasSwimmingPool", "hasDressingRoom", "hasFireplace",
            "hasThermicPanels", "hasArmoredDoor", "hasHeatPump", "hasPhotovoltaicPanels",
            "hasOffice", "hasAttic", "hasDiningRoom", "hasVisiophone", "hasGarden",
            "gardenSurface", "parkingCountOutdoor", "hasLift", "roomCount", "parkingCountIndoor",
            "hasBasement", "floorCount", "hasLivingRoom", "hasTerrace", "buildingConstructionYear",
            "facedeCount", "toiletCount", "bathroomCount", "bedroomCount", "postCode","diningRoomSurface",
            "kitchenSurface","terraceSurface","livingRoomSurface","landSurface","habitableSurface","streetFacadeWidth"
        ]

        str_cols = [
            "kitchenType", "floodZoneType", "heatingType", 
            "buildingCondition", "epcScore", "subtype", "province",
            "locality", "type"
        ]

        # Convert integer columns safely
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)

        # Convert string columns safely
        for col in str_cols:
            if col in df.columns:
                df[col]=df[col].fillna("missing value")
                df[col] = df[col].astype(str).str.strip()
               
        print("[INFO] All specified column types converted safely.")

        return df

    def normalization(self) -> pd.DataFrame:
        """
        Normalize categorical columns by mapping textual categories to numerical codes.

        Normalizes:
        - Building condition
        - EPC score
        - Heating type
        - Flood zone type
        - Kitchen type

        Returns:
            pd.DataFrame: DataFrame with additional normalized categorical columns.
        """

        # Get cleaned DataFrame
        df = self.clean_errors()

        # Normalization mappings
        building_conditions = {
            "missing value": -1,
            "GOOD": 1,
            "AS_NEW": 2,
            "TO_RENOVATE": 3, 
            "TO_BE_DONE_UP": 4,
            "JUST_RENOVATED": 5,
            "TO_RESTORE": 6
        }
        
        epc_scores = {
            "missing value": -1,
            'A++': 1,
            'A+': 2,
            'A': 3,
            'B': 4,
            'C': 5,
            'D': 6,
            'E': 7,
            'F': 8,
            'G': 9,
            'G_C': 9, # for ranges, we take the lowest score ()
            'F_D': 8,
            'C_A': 5,
            'F_C': 8,
            'E_C': 7,
            'C_B': 5,
            'E_D': 7,
            'G_F': 9,
            'D_C': 6,
            'G_E': 9,
            'X': 0
        }
        
        heating_types = {
            "missing value": -1,
            'GAS': 1,
            'FUELOIL': 2,
            'ELECTRIC': 3,
            'PELLET': 4,
            'WOOD': 5,
            'SOLAR': 6,
            'CARBON': 7
        }

        flood_zone_types = {
            "missing value": -1,
            'NON_FLOOD_ZONE': 1,
            'POSSIBLE_FLOOD_ZONE': 2,
            'RECOGNIZED_FLOOD_ZONE': 3,
            'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 4,
            'CIRCUMSCRIBED_WATERSIDE_ZONE': 5,
            'CIRCUMSCRIBED_FLOOD_ZONE': 6,
            'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 7,
            'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 8,
            'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 9
        }

        kitchen_types = {
            "missing value": -1,
            'NOT_INSTALLED': 0,
            'SEMI_EQUIPPED': 1,
            'INSTALLED': 2,
            'HYPER_EQUIPPED': 3,
            'USA_UNINSTALLED': 0,
            'USA_SEMI_EQUIPPED': 1,
            'USA_INSTALLED': 2,
            'USA_HYPER_EQUIPPED': 3
        }

        # Apply mappings to normalize categorical columns
        df["buildingConditionNormalize"] = df["buildingCondition"].replace(building_conditions)
        df["epcScoreNormalize"] = df["epcScore"].replace(epc_scores)
        df["heatingTypeNormalize"] = df["heatingType"].replace(heating_types)
        df["floodZoneTypeNormalize"] = df["floodZoneType"].replace(flood_zone_types)
        df["kitchenTypeNormalize"] = df["kitchenType"].replace(kitchen_types)

        df = df.drop(columns=["buildingCondition", "epcScore", "heatingType", "floodZoneType", "kitchenType"])

        return df
    
    def to_real_values(self) -> pd.DataFrame:
        """
        Convert normalized placeholder values (-1) to NaN to represent missing data.

        Returns:
            pd.DataFrame: Normalized DataFrame with missing values as NaN.
        """
        df = self.normalization()
        df = df.replace(-1, np.nan)
        return df

    def send_output_file(self, output_file: str):
        """
        Export the cleaned, deduplicated, and normalized DataFrame to a CSV file.

        Creates the output directory if it does not exist.

        Args:
            output_file (str): File path where to save the CSV output.

        Returns:
            None
        """
        cleaned_df = self.normalization()
        if not cleaned_df.empty:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cleaned_df.to_csv(output_file, index=False)
            print(f"[SUCCESS] Exported {len(cleaned_df)} merged records → {output_file}")
        else:
            print("[WARNING] No data exported due to empty or invalid input.")        