import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from data_cleanner import DataCleanner

# Chargement et préparation des données
data = DataCleanner("data/data_cleanned.csv")
df = data.load_data_file()
df = df[
        (df["price"].notna()) & (df["price"] > 10000) & (df["price"] < 1_000_000) &
        (df["habitableSurface"].notna()) & (df["habitableSurface"] > 10)
    ].copy()

df["price_per_m2"] = df["price"] / df["habitableSurface"]

####################################################################################
# Official Region Mapping by Postcode (Belgium)                                    #
#| Region   | Postcode Range        |                                              #
#| -------- | --------------------- |                                              #
#| Brussels | 1000–1299             |                                              #
#| Flanders | 1500–3999             |                                              #
#| Wallonia | 1300–1499 & 4000–7999 |                                              #
#                                                                                  #
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
    except:
        return "Unknown"

df["region"] = df["postCode"].apply(map_postcode_to_region)

# Agrégation par région
region_avg = df.groupby("region").agg(avg_price=("price", "mean")).reset_index()
fig_region = px.choropleth(
    region_avg,
    geojson="https://raw.githubusercontent.com/napoleon03/be-geojson/main/belgium_regions.geojson",
    featureidkey="properties.name",
    locations="region",
    color="avg_price",
    color_continuous_scale="Blues",
    title="💶 Average Property Price by Region"
)
fig_region.update_geos(fitbounds="locations", visible=False)

# Initialisation app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Belgium Real Estate"

# Layout responsive
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H2("🏠 Belgium Real Estate - Interactive Dashboard"), width=12)
    ], className="my-3"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Choropleth Map"),
                dbc.CardBody([
                    dcc.Graph(id="map", figure=fig_region, config={"displayModeBar": False})
                ])
            ])
        ], md=6, lg=4),  # Carte prend 1/3
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Province Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="province-bar", config={"displayModeBar": False})
                ])
            ])
        ], md=6, lg=8),  # Graph prend 2/3
    ])
], style={"padding": "20px"})


@app.callback(
    Output("province-bar", "figure"),
    Input("map", "clickData")
)
def update_province_chart(clickData):
    if not clickData:
        return px.bar(title="Click on a region to explore provinces")

    region = clickData["points"][0]["location"]
    province_avg = df[df["region"] == region].groupby("province").agg(
        avg_price=("price", "mean")).reset_index()

    fig = px.bar(
        province_avg,
        x="avg_price",
        y="province",
        orientation="h",
        title=f"Average Property Price by Province in {region}",
        color="avg_price",
        color_continuous_scale="Viridis"
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)