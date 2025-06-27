import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.data_cleanner import DataCleanner

# -------------------- Préparation des données --------------------

def get_least_expensive_data():
    data = DataCleanner("data/data_cleanned.csv")
    df = data.load_data_file()

    df = df[df["price"] > 10000]
    df = df[df["habitableSurface"] > 10]
    df["price_per_m2"] = df["price"] / df["habitableSurface"]
    df["region"] = df["postCode"].apply(map_postcode_to_region)

    summary_df = df[["locality", "province", "region", "price", "price_per_m2"]].copy()
    agg_df = summary_df.groupby(["region", "province", "locality"]).agg(
        avg_price=("price", "mean"),
        med_price=("price", "median"),
        price_m2=("price_per_m2", "mean"),
        count=("price", "count")
    ).reset_index()

    agg_df_belgium = agg_df.copy()
    agg_df_belgium["region"] = "Belgium"

    return pd.concat([agg_df, agg_df_belgium], ignore_index=True)

def map_postcode_to_region(postcode):
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

def plot_least_expensive(df_region, title_prefix, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{title_prefix} - Top 10 Least Expensive Municipalities", fontsize=16)

    # Moyenne
    bottom_avg = df_region.nsmallest(10, "avg_price")
    sns.barplot(data=bottom_avg, x="avg_price", y="locality", palette="Blues", ax=axes[0])
    axes[0].set_title("Average Price (€)")
    for i, row in bottom_avg.iterrows():
        axes[0].text(row["avg_price"], i, f"{int(row['avg_price']):,}€", va='center', ha='left', fontsize=9)

    # Médiane
    bottom_med = df_region.nsmallest(10, "med_price")
    sns.barplot(data=bottom_med, x="med_price", y="locality", palette="Greens", ax=axes[1])
    axes[1].set_title("Median Price (€)")
    for i, row in bottom_med.iterrows():
        axes[1].text(row["med_price"], i, f"{int(row['med_price']):,}€", va='center', ha='left', fontsize=9)

    # Prix/m²
    bottom_m2 = df_region.nsmallest(10, "price_m2")
    sns.barplot(data=bottom_m2, x="price_m2", y="locality", palette="Oranges", ax=axes[2])
    axes[2].set_title("Price per m² (€)")
    for i, row in bottom_m2.iterrows():
        axes[2].text(row["price_m2"], i, f"{int(row['price_m2']):,}€/m²", va='center', ha='left', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.show()

    plt.close()

# -------------------- Entrée principale --------------------

def generate_all_least_expensive_charts():
    os.makedirs("plots", exist_ok=True)

    df = get_least_expensive_data()

    region_files = {
        "Belgium": "plots/11_least_expensive_belgium.png",
        "Wallonia": "plots/12_least_expensive_wallonia.png",
        "Flanders": "plots/13_least_expensive_flander.png",
        "Brussels": "plots/14_least_expensive_bruxelles.png"
    }

    for region, path in region_files.items():
        region_df = df[df["region"] == region]
        if not region_df.empty:
            plot_least_expensive(region_df, region, save_path=path)
        else:
            print(f"⚠️ Aucune donnée pour la région : {region}")
