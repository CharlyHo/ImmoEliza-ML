import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def display_result(result: dict) -> pd.DataFrame:
    """Display result of each experiment"""
    return pd.DataFrame(
        {
            "Train R2": [result["train_R2"]],
            "Train MAE": [result["train_MAE"]],
            "Train RMSE": [result["train_RMSE"]],
            "Test R2": [result["test_R2"]],
            "Test MAE": [result["test_MAE"]],
            "Test RMSE": [result["test_RMSE"]],
            "Average Target": [result["average_target"]],
        },
        index=[result["title"]],
    )


def plot_scatter(
    y_test: pd.Series,
    y_pred: pd.Series,
    save_path: str,
    title: str,
) -> None:
    """ " Generate plots for each experiment"""

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    errors = y_pred - y_test
    plt.scatter(y_test, y_pred, c=errors, cmap="coolwarm", alpha=0.5)
    plt.colorbar(label="Prediction Error (Predicted - Actual)")
    plt.xlabel("Actual Price", fontsize=9)
    plt.ylabel("Predicted Price", fontsize=9)
    plt.title(f"Actual vs Predicted {title}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals_top(top_results: pd.DataFrame, output_dir) -> None:
    """Generate plots for top performing models"""

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "plot_residual.png")

    y_test = top_results["y_test"].tolist()
    y_pred = top_results["y_pred"].tolist()
    titles = top_results["title"].tolist()

    num_plots = len(top_results)
    fig, axis = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axis = [axis]

    for i in range(num_plots):
        residuals = y_test[i] - y_pred[i]
        axis[i].scatter(y_pred[i], residuals, c=residuals, cmap="coolwarm", alpha=0.7)
        axis[i].set_xlabel("Predicted Price", fontsize=8)
        axis[i].set_ylabel("Residuals", fontsize=8)
        axis[i].set_title(f"Residual Plot {titles[i]}")
        axis[i].axhline(y=0, color="r", linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_top_results_plots(results_df: pd.DataFrame, output_dir) -> None:
    """Generate top performing models"""
    top_results = results_df.head(3)
    plot_residuals_top(top_results, output_dir)


def display_summary_results(results_df: pd.DataFrame) -> None:
    """Display results summary"""

    cols_to_print = [
        "title",
        "train_R2",
        "train_MAE",
        "train_RMSE",
        "test_R2",
        "test_MAE",
        "test_RMSE",
        "average_target",
    ]
    available_cols = [col for col in cols_to_print if col in results_df.columns]

    print("🧪 EXPERIMENT RESULTS SUMMARY")
    print(results_df[available_cols])


def process_and_save_results(
    results: List[Dict[str, Any]], output_dir: Path
) -> pd.DataFrame:
    """save result to CSV"""

    results_df = pd.DataFrame(results)
    cols = ["title"] + [col for col in results_df.columns if col != "title"]
    results_df = results_df[cols]
    results_df = results_df.sort_values(by="test_R2", ascending=False).reset_index(
        drop=True
    )
    save_path = output_dir / "feature_engineering_results.csv"

    results_df.to_csv(save_path, index=False)
    return results_df
