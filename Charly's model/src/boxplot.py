# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_cleanner import DataCleanner  # Custom data loading/cleaning class
import matplotlib.ticker as mtick


# Load and clean the dataset using your custom cleaner
data = DataCleanner("Charly's model/data/data_cleanned.csv")
df = data.load_data_file()

# Filter out unrealistic price and surface entries
df = df[df["price"] > 10000]
df = df[df["habitableSurface"] > 10]  # Remove invalid or extremely small surfaces

# Compute price per square meter
df["price_per_m2"] = df["price"] / df["habitableSurface"]

####################################################################################
# Add a new column 'region' based on Belgian postcode ranges
####################################################################################

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

# Apply postcode-to-region mapping
df["region"] = df["postCode"].apply(map_postcode_to_region)

# Prepare a summarized DataFrame with key geographic and pricing info
summary_df = df[["locality", "province", "region", "price", "price_per_m2"]].copy()

# Aggregate stats at locality level
agg_df = summary_df.groupby(["region", "province", "locality"]).agg(
    avg_price=("price", "mean"),
    med_price=("price", "median"),
    price_m2=("price_per_m2", "mean"),
    count=("price", "count")
).reset_index()

#####################################################################################################
# Visualize Habitable Surface by Property Subtype                                                   #
#####################################################################################################

# Keep only valid surface entries and main property types
df = df[df['habitableSurface'].notna()]
df = df[df['type'].isin(['APARTMENT', 'HOUSE'])]
df = df[df["subtype"].isin([
    'APARTMENT', 'HOUSE', 'FLAT_STUDIO', 'DUPLEX', 'PENTHOUSE', 'GROUND_FLOOR',
    'APARTMENT_BLOCK', 'MANSION', 'EXCEPTIONAL_PROPERTY', 'MIXED_USE_BUILDING',
    'TRIPLEX', 'LOFT', 'VILLA', 'TOWN_HOUSE', 'CHALET', 'MANOR_HOUSE',
    'SERVICE_FLAT', 'KOT', 'FARMHOUSE', 'BUNGALOW', 'COUNTRY_COTTAGE',
    'OTHER_PROPERTY', 'CASTLE', 'PAVILION'
])]

# Boxplot with stripplot overlay for habitable surface per subtype
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='subtype', x='habitableSurface', palette='Set2')
sns.stripplot(data=df, y='subtype', x='habitableSurface', color='gray', size=3, jitter=True, alpha=0.4)
plt.title('Habitable Surface Comparison by Property Subtype')
plt.xlabel('Habitable Surface (m²)')
plt.ylabel('Subtype')
plt.grid(True)
plt.tight_layout()
plt.show()
#####################################################################################################
# Visualize Price by Property Subtype                                                   #
#####################################################################################################

# Keep only valid surface entries and main property types
df = df[df['habitableSurface'].notna()]
df = df[df['type'].isin(['APARTMENT', 'HOUSE'])]
df = df[df["subtype"].isin([
    'APARTMENT', 'HOUSE', 'FLAT_STUDIO', 'DUPLEX', 'PENTHOUSE', 'GROUND_FLOOR',
    'APARTMENT_BLOCK', 'MANSION', 'EXCEPTIONAL_PROPERTY', 'MIXED_USE_BUILDING',
    'TRIPLEX', 'LOFT', 'VILLA', 'TOWN_HOUSE', 'CHALET', 'MANOR_HOUSE',
    'SERVICE_FLAT', 'KOT', 'FARMHOUSE', 'BUNGALOW', 'COUNTRY_COTTAGE',
    'OTHER_PROPERTY', 'CASTLE', 'PAVILION'
])]

# Boxplot with stripplot overlay for habitable surface per subtype
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='subtype', x='price', palette='Set2')
sns.stripplot(data=df, y='subtype', x='price', color='gray', size=3, jitter=True, alpha=0.4)
plt.title('Price Comparison by Property Subtype')
plt.xlabel('Price')
plt.ylabel('Subtype')
plt.grid(True)
plt.tight_layout()
plt.show()
#####################################################################################################
#   Define a modular function to compare prices by any categorical variable                         #
#####################################################################################################

def plot_price_comparaison(df_region, title, var):
    """
    Visualizes average price, median price, and price per m²
    grouped by a categorical variable (e.g., subtype, region, etc.).
    """

    # Ensure required metrics exist
    if "avg_price" not in df_region.columns:
        # Clean and compute if needed
        if "price_per_m2" not in df_region.columns and "habitableSurface" in df_region.columns:
            df_region = df_region[
                (df_region['habitableSurface'].notna()) &
                (df_region['habitableSurface'] > 10) &
                (df_region['habitableSurface'] < 25000) &
                (df_region['price'] > 10000) &
                (df_region['price'] < 800000)
            ]
            df_region["price_per_m2"] = df_region["price"] / df_region["habitableSurface"]

        df_region = (
            df_region
            .groupby(var, as_index=False)
            .agg(
                avg_price=('price', 'mean'),
                med_price=('price', 'median'),
                price_m2=('price_per_m2', 'mean')
            )
        )

    # Create 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Price Comparison by {title}", fontsize=16)
    formatter = mtick.FormatStrFormatter('%.2f')  # 2 decimal places

    # --- Average Price ---
    avg_bar = sns.barplot(
        data=df_region.nsmallest(15, "avg_price"),
        x="avg_price", y=var,
        hue=var, palette="Blues_r", ax=axes[0], dodge=False
    )
    axes[0].set_title("Average Price (€)")
    axes[0].xaxis.set_major_formatter(formatter) # Format x-axis to 2 decimals
    axes[0].tick_params(axis='x', rotation=45)  # rotate tick labels
    axes[0].bar_label(avg_bar.containers[0], fmt="€ %.2f", padding=3)
    # Remove redundant legend only if it exists
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()

    # --- Median Price ---
    med_bar = sns.barplot(
        data=df_region.nsmallest(15, "med_price"),
        x="med_price", y=var,
        hue=var, palette="Greens_r", ax=axes[1], dodge=False
    )
    axes[1].set_title("Median Price (€)")
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].bar_label(med_bar.containers[0], fmt="€ %.2f", padding=3)
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    # --- Price per m² ---
    m2_bar = sns.barplot(
        data=df_region.nsmallest(15, "price_m2"),
        x="price_m2", y=var,
        hue=var, palette="Oranges_r", ax=axes[2], dodge=False
    )
    axes[2].set_title("Price per m² (€)")
    axes[2].xaxis.set_major_formatter(formatter) 
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].bar_label(m2_bar.containers[0], fmt="€ %.2f", padding=3)
    if axes[2].get_legend() is not None:
        axes[2].get_legend().remove()

    # Layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

####################################################################################
# Add a new column 'region' based on Belgian postcode ranges
####################################################################################

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

# Apply postcode-to-region mapping
df["region"] = df["postCode"].apply(map_postcode_to_region)

# Prepare a summarized DataFrame with key geographic and pricing info
summary_df = df[["locality", "province", "region", "price", "price_per_m2"]].copy()

# Aggregate stats at locality level
agg_df = summary_df.groupby(["region", "province", "locality"]).agg(
    avg_price=("price", "mean"),
    med_price=("price", "median"),
    price_m2=("price_per_m2", "mean"),
    count=("price", "count")
).reset_index()


###########################################################################################################
#           Fonction :Price comparaison by median number of bedrooms/bathrooms                            #
###########################################################################################################
# Function to plot price metrics by subtype, with avg bedroom as label
def plot_price_by_rooms(df,room):
    # Step 1: Clean and filter data
    df = df[
        (df["subtype"].notna()) &
        (df[room].notna()) & (df[room] >= 0) &
        (df["price"].notna()) & (df["price"] > 10000) & (df["price"] < 1_000_000) &
        (df["habitableSurface"].notna()) & (df["habitableSurface"] > 10)
    ].copy()

    # Step 2: Compute price per square meter
    df["price_per_m2"] = df["price"] / df["habitableSurface"]

    # Step 3: Group by property subtype
    grouped = df.groupby("subtype", as_index=False).agg({
        "price": ["mean", "median"],
        "price_per_m2": "mean",
        room: "median"
    })

    # Step 4: Rename flattened column names
    grouped.columns = ["subtype", "avg_price", "med_price", "price_m2", "median_room"]

    # Step 5: Round results to 2 decimals
    grouped = grouped.round({
        "avg_price": 2,
        "med_price": 2,
        "price_m2": 2,
        "median_room": 1
    })

    # Step 6: Sort by average bedroom count
    grouped = grouped.sort_values("median_room", ascending=True)

    # Step 7: Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Price Metrics by Property Subtype (Labels = Median {room} Count)", fontsize=16)

    # Price metrics and palettes
    metrics = ["avg_price", "med_price", "price_m2"]
    titles = ["Average Price (€)", "Median Price (€)", "Price per m² (€)"]
    palettes = ["Blues_r", "Greens_r", "Oranges_r"]

    for idx, (col, title, palette) in enumerate(zip(metrics, titles, palettes)):
        ax = axes[idx]
        plot = sns.barplot(
            data=grouped, x=col, y="subtype",
            palette=palette, ax=ax
        )
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("€{x:,.0f}"))
        ax.tick_params(axis='x', rotation=45)

        # Add bedroom labels in front of each bar
        for container, label in zip(plot.containers, grouped["median_room"]):
            for bar in container:
                if bar.get_height() > 0:
                    ax.text(
                        bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2,
                        f"{label:.1f} br", va="center", ha="left", fontsize=9, color="black"
                    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



###########################################################################################################
#                       Price comparison by Property Subtype                                             #
###########################################################################################################

# Ensure price is available
df = df[df['price'].notna()]
# Plot price metrics per property subtype
plot_price_comparaison(df, "Property Subtype", "subtype")

###########################################################################################################
#                       Price comparison by Building Condition                                            #
###########################################################################################################

# Filter valid building conditions
df = df[df['buildingCondition'].notna()]
# Plot price metrics per building condition
plot_price_comparaison(df, "Building Condition", "buildingCondition")
###########################################################################################################
#                       Price comparison by Kitchen Type                                                  #
###########################################################################################################

# Filter valid kitchen Type
##TO REDO : fusionner les types de Kitchen et les trier
df = df[df['kitchenType'].notna()]
# Plot price metrics per building condition
plot_price_comparaison(df, "Kitchen Type", "kitchenType")
###########################################################################################################
#                       Price comparison by Heating Type                                                  #
###########################################################################################################

# Filter valid Heating Type
df = df[df['heatingType'].notna()]
# Plot price metrics per building condition
plot_price_comparaison(df, "Heating Type", "heatingType")
###########################################################################################################
#                       Price comparison by Flood Zone Type                                                 #
###########################################################################################################

# Filter valid Heating Type
df = df[df['floodZoneType'].notna()]
# Plot price metrics per building condition
plot_price_comparaison(df, "Flood Zone Type", "floodZoneType")


###########################################################################################################
#                      Fonction :Price comparaison by median_bedrooms /median_bathrooms                                       #
###########################################################################################################
plot_price_by_rooms(df,'bedroomCount')

plot_price_by_rooms(df,'bathroomCount')
