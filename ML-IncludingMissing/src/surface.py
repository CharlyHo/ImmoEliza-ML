import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.patches as mpatches

matplotlib.use('TkAgg')

def generate_surface_charts(df: pd.DataFrame, show_plot: bool):
    """
    Generate and optionally display/save surface-related charts for the given property dataset.

    This function creates two plots:
    1. A histogram showing the distribution of habitable surface areas up to 1000 m².
    2. A boxplot for properties with large surface areas (greater than 1000 m²) showing price distribution.

    Args:
        df (pd.DataFrame): DataFrame containing property data, expected to include columns
                           'habitableSurface' and 'price'.
        show_plot (bool): Whether to display the generated plots interactively.

    Returns:
        None
    """
    plot_surface_histogram(df, "ML-IncludingMissing/plots/05_histogram_surface.png", show_plot)

    # big value for surface
    plot_big_surface_boxplot(df, min_surface=1000, plot_file_path="ML-IncludingMissing/plots/06_big_surface_boxplot.png", show_plot=show_plot)

def plot_surface_histogram(df: pd.DataFrame, plot_file_path: str, show_plot: bool) -> None:
    """
    Create and optionally save/show a histogram of property counts by habitable surface area.

    Filters out non-positive and excessively large values (>1000 m²) before plotting.

    Args:
        df (pd.DataFrame): DataFrame containing property data with a 'habitableSurface' column.
        plot_file_path (str): File path to save the plot image. If empty or None, the plot is not saved.
        show_plot (bool): Whether to display the plot interactively.

    Returns:
        None

    Raises:
        ValueError: If 'habitableSurface' column is missing in the DataFrame.
    """
    try:
        # Vérification de la présence de la colonne 'surface'
        if "habitableSurface" not in df.columns:
            raise ValueError("Column 'habitableSurface' not found in DataFrame.")

        # Suppression des valeurs manquantes ou aberrantes
        surface_data = df["habitableSurface"].dropna()
        surface_data = surface_data[(surface_data > 0) & (surface_data <= 1000)]  # filtre max 2000 m²

        # Création de l'histogramme
        plt.figure(figsize=(10, 6))
        sns.histplot(surface_data, bins=30, kde=False, color="skyblue", edgecolor="black")
        plt.title("Distribution du nombre de propriétés selon la surface")
        plt.xlabel("habitableSurface (m²)")
        plt.ylabel("Nombre de propriétés")
        plt.tight_layout()

        if plot_file_path:
            # Sauvegarde de l'histogramme
            plt.savefig(plot_file_path, dpi=300)

        if show_plot:
            # Affichage de l'histogramme
            plt.show()
            
    except Exception as e:
        print(f"[ERRO] Failed to plot surface histogram => {e}")

def plot_big_surface_boxplot(df: pd.DataFrame, surface_col="habitableSurface", price_col="price",
                             min_surface: int = 1000, plot_file_path: str = None, show_plot: bool = True) -> None:
    """
    Plot a boxplot of property prices for properties with a large surface area exceeding a minimum threshold.

    Filters properties by surface area and non-missing price, then plots price distribution.

    Args:
        df (pd.DataFrame): DataFrame containing property data.
        surface_col (str, optional): Name of the column containing surface area data. Defaults to "habitableSurface".
        price_col (str, optional): Name of the column containing price data. Defaults to "price".
        min_surface (int, optional): Minimum surface area threshold to filter properties. Defaults to 1000.
        plot_file_path (str, optional): File path to save the plot image. If None, the plot is not saved.
        show_plot (bool, optional): Whether to display the plot interactively. Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If either surface_col or price_col is missing in the DataFrame.
    """
    try:
        # Vérification des colonnes
        if surface_col not in df.columns or price_col not in df.columns:
            raise ValueError(f"Columns '{surface_col}' or '{price_col}' not found in DataFrame.")

        # Filtrage des données
        filtered_df = df[(df[surface_col] > min_surface) & (df[price_col].notna())]

        if filtered_df.empty:
            print(f"[INFO] No properties found with surface > {min_surface} m².")
            return

        # Création du boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=filtered_df[price_col], color="tomato")
        plt.title(f"Distribution des prix pour les propriétés avec surface > {min_surface} m²")
        plt.ylabel("Prix (€)")
        plt.tight_layout()

        # Enregistrement du graphique
        if plot_file_path:
            plt.savefig(plot_file_path, dpi=300)

        if show_plot:
            plt.show()

    except Exception as e:
        print(f"[ERROR] Failed to plot big surface boxplot => {e}")