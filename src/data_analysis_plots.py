import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.patches as mpatches

matplotlib.use('TkAgg')

def data_analysis_charts(df: pd.DataFrame, show_plot: bool):
    """
    Generate a series of exploratory data analysis charts on the dataset.

    This function calls individual plot functions to visualize:
    - Missing values percentages per feature
    - Correlation of features with price
    - Correlation among count-based features
    - Detection of outliers in numeric features

    Args:
        df (pd.DataFrame): The input dataset for analysis.
        show_plot (bool): Whether to display the plots interactively.

    Returns:
        None
    """

    # Plot missing values percentages
    plot_missing_values_percentage(df, "plots/01_missing_values_percentage.png", show_plot)

    # Plot correlations to price
    plot_correlations_to_price(df, "plots/02_correlation_with_variable_price.png", show_plot)

    # Plot other correlations
    # Correlation between count-based features (Pearson Correlation)
    plot_count_features_correlations(df, "plots/03_count_features_correlations.png", show_plot)

    # Plot the outliers
    plot_outliers(df, "plots/04_outliers.png", show_plot)

def plot_missing_values_percentage(df: pd.DataFrame, plot_file_path: str, show_plot: bool) -> None:
    """
    Plot a horizontal bar chart showing the percentage of missing values per feature.

    Args:
        df (pd.DataFrame): Dataset containing features to analyze for missing data.
        plot_file_path (str): Path to save the plot image file. If empty or None, plot is not saved.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        None
    """
    try:
        # Calculate missing data information
        missing_data = df.isna().sum()
        missing_pct = (missing_data / len(df)) * 100

        # Filter to features with missing values
        missing_df = missing_pct[missing_pct > 0].sort_values(ascending=False)

        # Set up the plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(missing_df.index, missing_df.values, color=plt.cm.tab20.colors[:len(missing_df)])

        # Axis Labels and Title (with units)
        plt.xlabel("Missing Percentage (%)", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title("Missing Values Percentage per Feature", fontsize=14, fontweight="bold", pad=15)

        # Add text annotations to the end of each bar (no overlapping if sorted properly)
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1,                    # small horizontal offset to the right
                    bar.get_y() + bar.get_height()/2,
                    f"{width:.1f}%", 
                    va='center', fontsize=10, fontweight="bold")

        # Invert y-axis so that features with highest missing % appear at the top
        plt.gca().invert_yaxis()

        plt.tight_layout()

        if plot_file_path:
            # Save outliers plot to file
            plt.savefig(plot_file_path, dpi=300)
            print(f"Missing Values Percentage plot saved to file: {plot_file_path}")

        if show_plot:
            # Show outliers plot on screen
            print(f"Showing plot for Missing Values Percentage...")
            plt.show()

    except Exception as e:
        print(f"[ERRO] Failed to plot Missing Values Percentage => {e}")

def plot_correlations_to_price(df: pd.DataFrame, plot_file_path: str, show_plot: bool) -> None:
    """
    Plot the correlation coefficients of numeric features with the 'price' column.

    Only numeric columns are considered, and the correlation excludes 'price' itself.

    Args:
        df (pd.DataFrame): Dataset with numeric features and a 'price' column.
        plot_file_path (str): File path to save the plot image. If None or empty, plot is not saved.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        None
    """
    try:
        # Selecting only numeric columns
        numeric_df = df.select_dtypes(include=["int64", "float64"])

        # Calculation of the correlation matrix
        corr_matrix = numeric_df.corr()

        # Extracting correlations with 'price'
        price_corr = corr_matrix["price"].drop("price").sort_values()

        # Plotting the correlations
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=price_corr.values,
            y=price_corr.index,
            hue=price_corr.index,
            palette="coolwarm",
            dodge=False,
            legend=False
        )
        plt.title("Correlation with the variable 'price'")
        plt.xlabel("Correlation coefficient")
        plt.ylabel("Features")
        plt.tight_layout()

        if plot_file_path:
            # Save correlations plot to file
            plt.savefig(plot_file_path, dpi=300)
            print(f"Correlation with the variable 'price' plot saved to file: {plot_file_path}")

        if show_plot:
            # Show correlations plot on screen
            print(f"Showing plot for correlation with the variable 'price'...")
            plt.show()

    except Exception as e:
        print(f"[ERRO] Failed to plot correlations with the the variable 'price' => {e}")

def plot_outliers(df: pd.DataFrame, plot_file_path: str, show_plot: bool) -> None:
    """
    Detect and visualize outliers in numeric features using boxplots.

    Uses the Interquartile Range (IQR) method to count outliers per feature,
    annotates counts on the plot, and colors extreme outlier counts differently.

    Args:
        df (pd.DataFrame): Dataset containing numeric features.
        plot_file_path (str): Path to save the boxplot image. If empty or None, plot is not saved.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        None
    """
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=["int64", "float64"])

        # Detect columns with outliers using the IQR method
        def outlier_count(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return ((series < lower) | (series > upper)).sum()

        # Create a dictionary with outlier counts
        outlier_counts = {col: outlier_count(numeric_df[col]) for col in numeric_df.columns}
        print(outlier_count)
        columns_with_outliers = {col: count for col, count in outlier_counts.items() if count > 0}
        print(columns_with_outliers)

        # Filter DataFrame
        filtered_df = numeric_df[list(columns_with_outliers.keys())]

        # Melt the dataframe into long format
        melted = filtered_df.melt(var_name="Feature", value_name="Value")

        # Sort features by number of outliers for better readability
        sorted_features = sorted(columns_with_outliers.items(), key=lambda x: x[1], reverse=True)
        feature_order = [feat for feat, _ in sorted_features]

        # Plot
        plt.figure(figsize=(16, 10))
        ax = sns.boxplot(
            x="Value",
            y="Feature",
            data=melted,
            order=feature_order,
            palette="coolwarm",
            showfliers=True
        )

        # Annotate outlier counts clearly to the right
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        offset = x_range * 0.06  # 6% to the right

        for i, feature in enumerate(feature_order):
            count = columns_with_outliers[feature]
            x_pos = melted[melted["Feature"] == feature]["Value"].quantile(0.97)
            
            color = "lightcoral" if count > 1000 else "green"
            
            ax.text(
                x_pos + offset, i, f"{count}",
                va='center', ha='left', fontsize=9, fontweight='bold', color=color
            )

        # Titles and labels
        plt.title("Outliers in Numeric Features", fontsize=14, fontweight='bold')
        plt.xlabel("Feature Value (various units)", fontsize=12)
        plt.ylabel("Feature Name", fontsize=12)

        # Legend
        normal_patch = mpatches.Patch(color='green', label='Normal outlier count (<= 1000)')
        extreme_patch = mpatches.Patch(color='lightcoral', label='Extreme outlier count (> 1000)')
        plt.legend(handles=[normal_patch, extreme_patch], title='Legend', loc='lower right')
    
        plt.tight_layout()

        if plot_file_path:
            # Save outliers plot to file
            plt.savefig(plot_file_path, dpi=150)
            print(f"Outliers plot saved to file: {plot_file_path}")

        if show_plot:
            # Show outliers plot on screen
            print(f"Showing plot for outliers...")
            plt.show()

    except Exception as e:
        print(f"[ERRO] Failed to plot outliers => {e}")

def plot_count_features_correlations(df: pd.DataFrame, plot_file_path: str, show_plot: bool) -> None:
    """
    Plot a heatmap of Pearson correlation coefficients among count-based features.

    The selected features represent counts of rooms, bathrooms, parking, etc.

    Args:
        df (pd.DataFrame): Dataset containing count-based features.
        plot_file_path (str): Path to save the heatmap image. If empty or None, plot is not saved.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        None
    """
    try:
        # Define count-related columns
        count_columns = [
            "bathroomCount",
            "bedroomCount",
            "facedeCount",
            "floorCount",
            "parkingCountIndoor",
            "parkingCountOutdoor",
            "roomCount",
            "toiletCount"
        ]

        # Subset the DataFrame
        count_df = df[count_columns]

        # Compute the correlation matrix
        corr_matrix = count_df.corr()

        # Set up the plot
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar_kws={"label": "Pearson Correlation Coefficient"}
        )

        # Title and labels
        plt.title("Correlation Between Count-Based Features", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Features (unit: count)", fontsize=12)
        plt.ylabel("Features (unit: count)", fontsize=12)

        # Improve layout and prevent text overlapping
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if plot_file_path:
            # Save outliers plot to file
            plt.savefig(plot_file_path, dpi=300)
            print(f"Count Features Correlations plot saved to file: {plot_file_path}")

        if show_plot:
            # Show outliers plot on screen
            print(f"Showing plot for Count Features Correlations...")
            plt.show()

    except Exception as e:
        print(f"[ERRO] Failed to plot Count Features Correlations => {e}")