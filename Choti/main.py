import os
import pandas as pd
from typing import Any, Dict, List, Optional
from utils.data_cleaning import load_and_clean_data
from utils.test_list import create_test_list
from utils.data_preprocessing import data_preprocessing
import mlflow
from pathlib import Path
from utils.output import (
    process_and_save_results,
    display_summary_results,
    generate_top_results_plots,
)
from utils.run import run_experiments


def main():
    """Main entry point"""
    data_path = "./ml_ready_real_estate_data_soft_filled.csv"
    target = "price"
    output_dir = Path("Choti/output")
    plot_dir = Path("Choti/plot")

    df = load_and_clean_data(data_path)
    print(f"Data loaded successfully.")

    test_list = create_test_list(df, target)
    print(f"Created {len(test_list)} test configurations")

    # mlflow.autolog()
    results = run_experiments(test_list, df, target, plot_dir)
    results_df = process_and_save_results(results, output_dir)
    display_summary_results(results_df)
    generate_top_results_plots(results_df, output_dir)


if __name__ == "__main__":
    main()
