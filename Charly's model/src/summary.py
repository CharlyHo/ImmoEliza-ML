import pandas as pd
import os
import plotly.express as px
from data_cleanner import DataCleanner

###############################################################################
# 1. VALIDATION AUTOMATISÉE
###############################################################################

def validate_dataset(df):
    required_cols = {"price", "habitableSurface", "postCode", "locality", "province", "subtype", "type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    print("✅ Dataset validation passed.")

###############################################################################
# 2. CHARGEMENT ET FILTRAGE DES DONNÉES
###############################################################################

data = DataCleanner("Charly's model/data/data_cleanned.csv")
df = data.load_data_file()
validate_dataset(df)

# Filtres qualité
df = df[
    (df["habitableSurface"].notna()) &
    (df["habitableSurface"] > 10) &
    (df["habitableSurface"] < 25000) &
    (df["price"] > 10000) &
    (df["price"] < 800000)
]

# Calcul prix/m²
df["price_per_m2"] = df["price"] / df["habitableSurface"]

# Mapping code postal → région
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

df["region"] = df["postCode"].apply(map_postcode_to_region)

# Détection type principal
df["type_main"] = df["type"].apply(lambda x: "Apartment" if "apart" in str(x).lower() else "House")

###############################################################################
# 3. AGRÉGATION PAR LOCALITÉ / REGION / SUBTYPE
###############################################################################

summary_df = df[["locality", "province", "region", "subtype", "price", "price_per_m2"]]
agg_df = summary_df.groupby(["region", "province", "locality", "subtype"]).agg(
    avg_price=("price", "mean"),
    med_price=("price", "median"),
    price_m2=("price_per_m2", "mean"),
    count=("price", "count")
).reset_index()

###############################################################################
# 4. TABLEAU PIVOTÉ : HOUSE / APPARTEMENT
###############################################################################

pivot_df = df.groupby(["region", "locality", "type_main"]).agg(
    avg_price=("price", "mean")
).reset_index()

pivot_table = pivot_df.pivot(index=["region", "locality"], columns="type_main", values="avg_price").reset_index()
pivot_table.columns.name = None  # Clean MultiIndex

###############################################################################
# 5. PIECHART PAR LOCALITÉ + REGION (SUBTYPE DISTRIBUTION)
###############################################################################

# Ensure output directory exists
os.makedirs("output/piecharts", exist_ok=True)

# Aggregate property count and average price per subtype, region, and locality
subtype_dist = df.groupby(["region", "locality", "subtype"]).agg(
    count=("price", "count"),
    avg_price=("price", "mean")
).reset_index()

# Generate one piechart per region-locality group
for (region, locality), group in subtype_dist.groupby(["region", "locality"]):
    if len(group) < 2:
        continue  # Skip small or non-diverse groups

    # Create pie chart using Plotly
    fig = px.pie(
        group,
        values="count",              # Pie slice size = number of properties
        names="subtype",             # Label each slice by property subtype
        title=f"{locality}, {region} – Subtype Distribution",
        hole=0.4,                    # Donut style
        custom_data=["avg_price"]    # Attach avg_price for hover display
    )

    # Customize hover tooltip to show average price
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Avg. Price: %{customdata[0]:,.0f} €<extra></extra>"
    )

    # Export as HTML file (interactive)
    filename = f"output/piecharts/{region}_{locality}.html".replace(" ", "_")
    fig.write_html(filename)

###############################################################################
# 6. EXPORTS
###############################################################################

os.makedirs("output", exist_ok=True)
agg_df.to_csv("output/summary_by_locality.csv", index=False)
pivot_table.to_csv("output/price_matrix_apartment_house.csv", index=False)

# Type le plus vendu
most_popular = df["type"].value_counts().reset_index()
most_popular.columns = ["type", "count"]
most_popular.to_csv("output/most_popular_type.csv", index=False)

with open("output/most_popular_type.txt", "w") as f:
    f.write(f"🏆 Most sold type: {most_popular.iloc[0]['type']} ({most_popular.iloc[0]['count']} ventes)")

print("✅ Tous les fichiers ont été générés dans le dossier 'output/'.")
