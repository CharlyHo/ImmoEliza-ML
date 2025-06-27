import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.data_cleanner import DataCleanner

# -------------------- Préparation des données --------------------

def get_expensive_municipality_data():
    """
    Load, clean, and aggregate property price data to identify expensive municipalities in Belgium.

    - Loads data from a cleaned CSV file.
    - Filters properties with price > 10,000€ and habitable surface > 10 m².
    - Calculates price per square meter.
    - Maps postal codes to regions (Brussels, Wallonia, Flanders, Unknown).
    - Aggregates data by region, province, and locality to compute average price, median price,
      average price per m², and count of properties.
    - Adds an aggregate row for the whole country ("Belgium") by duplicating regional data.

    Returns:
        pd.DataFrame: Aggregated DataFrame with columns:
            ['region', 'province', 'locality', 'avg_price', 'med_price', 'price_m2', 'count']
    """
    # Chargement et nettoyage des données
    data = DataCleanner("Charly's model/data/data_cleanned.csv")
    df = data.load_data_file()

    # Filtrage des valeurs aberrantes
    df = df[df["price"] > 10000]
    df = df[df["habitableSurface"] > 10]
    df["price_per_m2"] = df["price"] / df["habitableSurface"]

    # Mapping des codes postaux vers les régions
    df["region"] = df["postCode"].apply(map_postcode_to_region)

    # Agrégation des statistiques
    summary_df = df[["locality", "province", "region", "price", "price_per_m2"]].copy()
    agg_df = summary_df.groupby(["region", "province", "locality"]).agg(
        avg_price=("price", "mean"),
        med_price=("price", "median"),
        price_m2=("price_per_m2", "mean"),
        count=("price", "count")
    ).reset_index()

    # Ajouter une ligne "Belgium"
    agg_df_belgium = agg_df.copy()
    agg_df_belgium["region"] = "Belgium"

    full_df = pd.concat([agg_df, agg_df_belgium], ignore_index=True)
    return full_df

def map_postcode_to_region(postcode):
    """
    Map a Belgian postal code to its corresponding region.

    Regions:
        - Brussels: 1000–1299
        - Wallonia: 1300–1499 and 4000–7999
        - Flanders: 1500–3999
        - Unknown: Any other value or invalid input

    Args:
        postcode (str or int): Postal code to map.

    Returns:
        str: Region name ("Brussels", "Wallonia", "Flanders", or "Unknown").
    """
    try:
        pc = int(postcode)
        if 1000 <= pc <= 1299:
            return "Brussels"
        elif (1300 <= pc <= 1499) or (4000 <= pc <= 7999):
            return "Wallonia"
        elif 1500 <= pc <= 3999:
            return "Flanders"
        else:
            return "Unknown"
    except:
        return "Unknown"

# -------------------- Plotting --------------------

def plot_top_expensive(df_region, title_prefix, save_path=None):
    """
    Plot bar charts for the top 10 most expensive municipalities in a given region.

    Generates three side-by-side barplots showing:
    1. Average price (€)
    2. Median price (€)
    3. Price per square meter (€)

    Each bar is annotated with its respective value.

    Args:
        df_region (pd.DataFrame): DataFrame filtered by a specific region with columns
                                  ['locality', 'avg_price', 'med_price', 'price_m2'].
        title_prefix (str): Title prefix to display on the plot.
        save_path (str, optional): File path to save the plot image. If None, the plot is not saved.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{title_prefix} - Top 10 Most Expensive Municipalities", fontsize=16)

    # Moyenne
    top_avg = df_region.nlargest(10, "avg_price")
    sns.barplot(data=top_avg, x="avg_price", y="locality", palette="Blues", ax=axes[0])
    axes[0].set_title("Average Price (€)")
    for i, row in top_avg.iterrows():
        axes[0].text(row["avg_price"], i, f"{int(row['avg_price']):,}€", va='center', ha='left', fontsize=9)

    # Médiane
    top_med = df_region.nlargest(10, "med_price")
    sns.barplot(data=top_med, x="med_price", y="locality", palette="Greens", ax=axes[1])
    axes[1].set_title("Median Price (€)")
    for i, row in top_med.iterrows():
        axes[1].text(row["med_price"], i, f"{int(row['med_price']):,}€", va='center', ha='left', fontsize=9)

    # Prix/m²
    top_m2 = df_region.nlargest(10, "price_m2")
    sns.barplot(data=top_m2, x="price_m2", y="locality", palette="Oranges", ax=axes[2])
    axes[2].set_title("Price per m² (€)")
    for i, row in top_m2.iterrows():
        axes[2].text(row["price_m2"], i, f"{int(row['price_m2']):,}€/m²", va='center', ha='left', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.show()

    plt.close()

# -------------------- Entrée principale --------------------

def generate_all_expensive_municipality_charts():
    """
    Generate and save top expensive municipality charts for Belgium and its regions.

    - Ensures the "plots" directory exists.
    - Loads and processes property data to get aggregated expensive municipality info.
    - Generates and saves bar charts for Belgium, Wallonia, Flanders, and Brussels.
    - Prints a warning if no data is available for a region.

    Returns:
        None
    """
    os.makedirs("plots", exist_ok=True)

    df = get_expensive_municipality_data()

    region_files = {
        "Belgium": "plots/07_top_expensive_belgium.png",
        "Wallonia": "plots/08_top_expensive_wallonia.png",
        "Flanders": "plots/09_top_expensive_flander.png",
        "Brussels": "plots/10_top_expensive_bruxelles.png"
    }

    for region, path in region_files.items():
        region_df = df[df["region"] == region]
        if not region_df.empty:
            plot_top_expensive(region_df, region, save_path=path)
        else:
            print(f"⚠️ Aucune donnée pour la région : {region}")
