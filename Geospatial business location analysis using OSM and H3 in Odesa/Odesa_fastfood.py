# --------------------------- 🔹 GIS Project: Fast Food Location Analysis in Odesa ------------------------------------
"""
Author: Ivan Tatarchuk

GIS-based analytical project focused on optimizing the location strategy for fast food businesses
in the city of Odesa, Ukraine.

🎯 Objective:
To identify optimal locations for opening new fast food outlets by analyzing the spatial distribution
of existing HoReCa venues (restaurants, cafes, fast food, pubs, etc.), Points of Interest (POIs),
and population density. The project includes a fully automated backend pipeline with data collection,
processing, visualization, and decision modeling.

🗺️ Key components:
- Loading and visualizing city boundaries and administrative districts (OpenStreetMap)
- Collecting and analyzing food-related POIs using `osmnx` and `geopandas`
- Processing real-world population density data via H3 hexagons (Kontur Population)
- Generating a feature matrix for multicriteria decision-making
- Implementing a modified Voronin model with penalty logic and average-centered scoring
- Producing a final scoring map with business recommendations

Required packages:
----------------------------
pip                          25.0.1
fiona                        1.8.22
osmnx                        1.9.4
geopandas                    0.13.2
matplotlib                   3.7.5
pandas                       2.0.3
seaborn                      0.13.2
shapely                      2.0.7
numpy                        1.24.3
h3                           3.7.6
"""

# -------------------- 🔹 Library Imports --------------------

import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shapely import wkt, Polygon
import numpy as np
import h3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.stdout = open("console_output.log", "w", encoding="utf-8")

# -------------------- 🔹 Function: Load City Boundary from OpenStreetMap --------------------

def load_city_boundary(city_name: str = "Odesa, Ukraine") -> gpd.GeoDataFrame:
    """
    Loads the geographic boundary of a given city from OpenStreetMap using osmnx.

    Args:
        city_name (str): Name of the city to query (default: "Odesa, Ukraine").

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the polygon geometry of the city's boundary.
    """
    boundary = ox.geocode_to_gdf(city_name)
    return boundary

# -------------------- 🔹 Visualization: City Boundary --------------------

def plot_city_boundary(gdf_boundary: gpd.GeoDataFrame, title: str = "City Boundary of Odesa") -> None:
    """
    Plots the city boundary on a map using matplotlib.

    Args:
        gdf_boundary (gpd.GeoDataFrame): GeoDataFrame containing the polygon of the city boundary.
        title (str): Title for the plot (default: "City Boundary of Odesa").
    """
    ax = gdf_boundary.plot(color='lightblue', edgecolor='black', figsize=(10, 8))
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.show()

# -------------------- 🔹 Load Administrative Districts from OpenStreetMap --------------------

def load_city_districts(city_polygon: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Loads administrative districts (admin_level=10) located within the given city boundary from OpenStreetMap.

    Args:
        city_polygon (gpd.GeoDataFrame): GeoDataFrame containing the boundary polygon of the city.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with cleaned and filtered district boundaries.
    """
    tags = {
        "admin_level": "10",
        "boundary": "administrative"
    }

    geometry = city_polygon.loc[0, 'geometry']
    districts = ox.features_from_polygon(geometry, tags)

    # Keep only named polygon geometries
    districts = districts[
        (districts.geometry.type.isin(["Polygon", "MultiPolygon"])) &
        (districts['name'].notnull())
    ].copy()

    # Filter: only districts strictly within the city polygon and not labeled as "hromada"
    districts = districts[districts.geometry.within(geometry)]
    districts = districts[~districts['name'].str.lower().str.contains("громада")]

    # Final cleanup
    districts = districts[["name", "geometry"]].reset_index(drop=True)
    districts.rename(columns={"name": "District"}, inplace=True)

    return districts

# -------------------- 🔹 Load Food Establishment Locations from OpenStreetMap --------------------

def load_food_locations(city_polygon: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Loads food-related locations (fast food, restaurants, cafes, bars, etc.) within the city polygon from OpenStreetMap.

    The function retrieves all POIs tagged with 'amenity' and ensures that a 'cuisine' column is present, even if missing.

    Args:
        city_polygon (gpd.GeoDataFrame): Polygon representing the city boundary.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing food-related points of interest (POIs).
    """
    # Reference:
    # https://taginfo.openstreetmap.org/keys/amenity#values
    # https://taginfo.openstreetmap.org/keys/cuisine#values

    # Define relevant types of food establishments
    tags = {
        "amenity": ["fast_food", "restaurant", "cafe", "bar", "food_court", "pub", "canteen"]
    }

    # Retrieve features from OpenStreetMap
    gdf = ox.features_from_polygon(city_polygon.unary_union, tags)
    gdf = gdf[gdf.geometry.type == "Point"].copy()

    # Ensure 'cuisine' column exists
    if "cuisine" not in gdf.columns:
        gdf["cuisine"] = "no_info"
    else:
        gdf["cuisine"] = gdf["cuisine"].fillna("no_info")

    # Logging for debug
    print("🔍 Total points after amenity filtering:", len(gdf))
    print("📎 Sample unique cuisine values:", gdf["cuisine"].unique()[:10])

    return gdf

# -------------------- 🔹 Analyze Food Establishments by Administrative District --------------------

def analyze_food_by_district(gdf_food: gpd.GeoDataFrame, city_districts: gpd.GeoDataFrame) -> None:
    """
    Analyzes the number of food-related establishments (fast food, restaurants, cafes, etc.)
    within each administrative district and prints the results.

    Args:
        gdf_food (gpd.GeoDataFrame): GeoDataFrame containing food-related points (with 'amenity' tags).
        city_districts (gpd.GeoDataFrame): GeoDataFrame containing the district polygons.
    """
    # Spatial join to determine which district each food point belongs to
    joined = gpd.sjoin(gdf_food, city_districts, how="left", predicate="within")

    # Count number of establishments per district
    if 'District' in joined.columns:
        grouped = joined['District'].value_counts().reset_index()
        grouped.columns = ['District', 'Food_Count']

        print("\n📊 Number of food establishments per district:")

        # Show only valid district names
        for _, row in grouped.iterrows():
            name = row['District']
            if isinstance(name, str) and 'район' in name.lower() and name != "Одеський район":
                print(f"{name:25} → {row['Food_Count']} venues")

    else:
        print("⚠️ Column 'District' not found in the joined GeoDataFrame.")

    print(f"\n🔢 Total number of food establishments: {len(gdf_food)}")

# -------------------- 🔹 Plot Food Establishments Within City Boundary --------------------

def plot_food_locations(city_polygon: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame):
    """
    Visualizes food establishments (fast food, cafes, restaurants, bars/pubs) as point markers within the city boundary.

    Args:
        city_polygon (gpd.GeoDataFrame): GeoDataFrame containing the city boundary.
        gdf (gpd.GeoDataFrame): GeoDataFrame with food-related points of interest.
    """
    gdf = gdf.copy()
    gdf["category_group"] = gdf["amenity"].map({
        "fast_food": "Fast food",
        "food_court": "Fast food",
        "canteen": "Fast food",
        "restaurant": "Restaurant",
        "cafe": "Cafe",
        "bar": "Bar/Pub",
        "pub": "Bar/Pub"
    })
    gdf = gdf[gdf["category_group"].notna()]

    fig, ax = plt.subplots(figsize=(10, 10))
    city_polygon.boundary.plot(ax=ax, color='black', linewidth=1)

    sns.scatterplot(data=gdf, x=gdf.geometry.x, y=gdf.geometry.y, hue="category_group",
                    palette="Set1", s=30, ax=ax, edgecolor="black", linewidth=0.2)

    ax.set_title("Food Establishments in Odesa", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend(title="Category", loc="best")
    plt.tight_layout()
    plt.show()

# -------------------- 🔹 Plot Food Establishments by District --------------------

def plot_districts_and_food(city_polygon: gpd.GeoDataFrame,
                             city_districts: gpd.GeoDataFrame,
                             gdf: gpd.GeoDataFrame):
    """
    Visualizes food establishments overlaid on administrative districts of Odesa.

    Args:
        city_polygon (gpd.GeoDataFrame): GeoDataFrame with city boundary polygon.
        city_districts (gpd.GeoDataFrame): GeoDataFrame with district polygons.
        gdf (gpd.GeoDataFrame): GeoDataFrame with food-related points of interest.
    """
    gdf = gdf.copy()
    gdf["category_group"] = gdf["amenity"].map({
        "fast_food": "Fast food",
        "food_court": "Fast food",
        "canteen": "Fast food",
        "restaurant": "Restaurant",
        "cafe": "Cafe",
        "bar": "Bar/Pub",
        "pub": "Bar/Pub"
    })
    gdf = gdf[gdf["category_group"].notna()]
    joined = gpd.sjoin(gdf, city_polygon, how="inner", predicate="within")

    fig, ax = plt.subplots(figsize=(10, 10))
    city_polygon.boundary.plot(ax=ax, color='black', linewidth=1)
    city_districts.boundary.plot(ax=ax, color='blue', linewidth=0.8, linestyle="--")

    sns.scatterplot(data=joined, x=joined.geometry.x, y=joined.geometry.y, hue="category_group",
                    palette="Set1", s=25, ax=ax, edgecolor="black", linewidth=0.2)

    # Add district labels at polygon centroids
    for _, row in city_districts.iterrows():
        if row["geometry"].geom_type in ["Polygon", "MultiPolygon"]:
            centroid = row["geometry"].centroid
            ax.text(centroid.x, centroid.y, row["District"],
                    fontsize=9, ha='center', va='center', color='navy', alpha=0.7)

    ax.set_title("Food Establishments by District in Odesa", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend(title="Category", loc="best")
    plt.tight_layout()
    plt.show()

# -------------------- 🔹 Load and Categorize POIs from OpenStreetMap --------------------

def load_and_expand_poi_locations(city_polygon: gpd.GeoDataFrame,
                                  output_csv: str = "Expanded_POI_locations_Odesa.csv") -> pd.DataFrame:
    """
    Loads Points of Interest (POIs) within the city boundary from OpenStreetMap using specified tags,
    filters relevant objects, and expands them into individual rows for each matching category.

    Args:
        city_polygon (gpd.GeoDataFrame): GeoDataFrame with the city boundary polygon.
        output_csv (str): Output path to save the filtered and categorized POI dataset.

    Returns:
        pd.DataFrame: A DataFrame containing categorized POIs with geometry and attributes.
    """
    tags = {
        "shop": ["mall", "supermarket"],
        "office": True,
        "railway": ["station"],
        "amenity": ["university", "college", "marketplace", "cinema", "theatre", "nightclub", "sports_centre",
                    "stadium", "bus_station", "exhibition_centre"],
        "tourism": ["beach", "attraction", "theme_park"],
        "leisure": ["sports_centre", "fitness_centre", "stadium", "beach_resort", "water_park",
                    "ice_rink", "bowling_alley"]
    }

    # 1. Load POIs from OSM
    gdf = ox.features_from_polygon(city_polygon.unary_union, tags)
    gdf = gdf[gdf.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])].copy()

    # 2. Convert geometries to centroids (in projected CRS) and back to WGS84
    gdf = gdf.to_crs(epsg=3857)
    gdf["geometry"] = gdf["geometry"].centroid
    gdf = gdf.to_crs(epsg=4326)

    # Save intermediate cleaned data
    cols_to_keep = ['name:en', 'addr:street', 'addr:housenumber',
                    'shop', 'office', 'railway', 'amenity', 'tourism', 'leisure']
    gdf_clean = gdf[[col for col in cols_to_keep if col in gdf.columns]].copy()
    gdf_clean['geometry'] = gdf['geometry']
    gdf_clean.to_csv("Points_of_Interest_locations_Odesa.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Intermediate file saved: Points_of_Interest_locations_Odesa.csv ({len(gdf_clean)} rows)")

    # 3. Filter and categorize POIs
    filtered = []
    for _, row in gdf.iterrows():
        categories = []

        if row.get("shop") in tags["shop"]:
            categories.append("Mall / Supermarket")
        if tags["office"] is True and pd.notnull(row.get("office")):
            categories.append("Office / Business Center")
        if row.get("railway") in tags["railway"]:
            categories.append("Transport Hub")
        if row.get("amenity") in tags["amenity"]:
            if row["amenity"] in ["university", "college"]:
                categories.append("University")
            elif row["amenity"] == "marketplace":
                categories.append("Market")
            elif row["amenity"] in ["cinema", "theatre", "nightclub", "exhibition_centre"]:
                categories.append("Entertainment")
            elif row["amenity"] in ["sports_centre", "stadium"]:
                categories.append("Sport Facility")
            elif row["amenity"] == "bus_station":
                categories.append("Transport Hub")
        if row.get("tourism") in tags["tourism"]:
            categories.append("Tourism / Beach")
        if row.get("leisure") in tags["leisure"]:
            if row["leisure"] in ["sports_centre", "fitness_centre", "stadium"]:
                categories.append("Sport Facility")
            elif row["leisure"] in ["beach_resort", "water_park", "ice_rink", "bowling_alley"]:
                categories.append("Entertainment")

        for cat in set(categories):
            row_copy = row.copy()
            row_copy["category"] = cat
            filtered.append(row_copy)

    if not filtered:
        print("❌ No relevant POIs found after filtering.")
        return pd.DataFrame()

    final_df = gpd.GeoDataFrame(filtered)[['name:en', 'addr:street', 'addr:housenumber', 'category', 'geometry']]
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"✅ Final file saved: {output_csv} ({len(final_df)} rows)")
    print("📊 Category counts:", final_df['category'].value_counts().to_dict())

    return final_df

# -------------------- 🔹 Visualize Categorized POIs on City Map --------------------

def plot_poi_locations(expanded_df: gpd.GeoDataFrame, city_boundary: gpd.GeoDataFrame):
    """
    Visualizes categorized POIs on a map of Odesa, with each category represented by a distinct color.

    Args:
        expanded_df (gpd.GeoDataFrame): GeoDataFrame containing categorized POIs and geometries.
        city_boundary (gpd.GeoDataFrame): GeoDataFrame with the city boundary polygon.
    """
    # Filter only point geometries
    expanded_df = expanded_df[expanded_df.geometry.type == "Point"]

    # Initialize figure and plot
    fig, ax = plt.subplots(figsize=(12, 12))
    city_boundary.boundary.plot(ax=ax, color='black', linewidth=1)

    sns.scatterplot(
        data=expanded_df,
        x=expanded_df.geometry.x,
        y=expanded_df.geometry.y,
        hue="category",
        ax=ax,
        s=20,
        alpha=0.8,
        palette="Dark2",
        edgecolor="black",
        linewidth=0.2
    )

    ax.set_title("Categorized POIs in Odesa", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='best', title="Category", fontsize='small', title_fontsize='medium')
    plt.tight_layout()
    plt.show()

# -------------------- 🔹 Load Population Density from Kontur Dataset --------------------

def load_population_density_from_kontur(city_polygon: gpd.GeoDataFrame,
                                        kontur_path: str = "kontur_population_UA_20231101.gpkg") -> gpd.GeoDataFrame:
    """
    Loads population density data from the Kontur Population dataset (GPKG format),
    clips the data to the boundaries of Odesa, and returns a GeoDataFrame of hexagons.
    Source - https://data.humdata.org/dataset/kontur-population-ukraine

    Args:
        city_polygon (gpd.GeoDataFrame): GeoDataFrame representing the city boundary.
        kontur_path (str): File path to the Kontur Population H3 hexagon dataset (GPKG format).

    Returns:
        gpd.GeoDataFrame: Filtered hexagons that intersect the city boundary.
    """
    print("📥 Loading Kontur Population data (may take 20–30 seconds)...")
    gdf_all = gpd.read_file(kontur_path)
    print(f"✅ Total number of hexagons in file: {len(gdf_all)}")

    # Convert city geometry to single polygon
    city_geom = city_polygon.unary_union

    # Clip population data to city boundaries
    gdf_all = gdf_all.to_crs(city_boundary.crs)
    gdf_filtered = gpd.clip(gdf_all, city_boundary)

    print(f"✅ Hexagons within Odesa boundary: {len(gdf_filtered)}")

    return gdf_filtered

# -------------------- 🔹 Visualize Population Density by H3 Hexagons --------------------

def plot_population_hexagons(city_boundary: gpd.GeoDataFrame, gdf_hex: gpd.GeoDataFrame):
    """
    Plots population density using H3 hexagons overlaid on the map of Odesa.

    Args:
        city_boundary (gpd.GeoDataFrame): GeoDataFrame containing the city boundary.
        gdf_hex (gpd.GeoDataFrame): GeoDataFrame of H3 hexagons with a 'population' column.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    city_boundary.boundary.plot(ax=ax, color='black', linewidth=1)
    gdf_hex.plot(column='population', ax=ax, cmap='YlOrRd', legend=True,
                 edgecolor='black', linewidth=0.2, alpha=0.8)

    plt.title("Population Density in Odesa (Kontur H3 Hexagons)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -------------------- 🔹 Food Coverage Analysis by H3 Hexagons --------------------

def analyze_food_coverage_by_population(food_gdf: gpd.GeoDataFrame, hex_gdf: gpd.GeoDataFrame) -> None:
    """
    Analyzes the coverage of food-related points (restaurants, cafes, fast-food, etc.)
    in relation to population density per H3 hexagon.

    For each hexagon:
    - Counts the number of food points within it;
    - Normalizes this value per 1,000 residents;
    - Visualizes the resulting spatial distribution.

    Parameters:
    -----------
    food_gdf : GeoDataFrame
        GeoDataFrame with food-related POIs (e.g., filtered_food_odesa.csv), in CRS EPSG:4326.
    hex_gdf : GeoDataFrame
        GeoDataFrame with H3 hexagons and population data (e.g., from Kontur Population), in CRS EPSG:4326.
    """
    print("📊 Calculating food coverage...")

    # Ensure both GeoDataFrames are in the correct CRS
    food_gdf = food_gdf.to_crs(epsg=4326)
    hex_gdf = hex_gdf.to_crs(epsg=4326)

    # Count number of food points within each hexagon
    joined = gpd.sjoin(food_gdf, hex_gdf, how='inner', predicate='within')
    food_count = joined.groupby('index_right').size()

    # Assign food counts to hexagons
    hex_gdf['food_points'] = hex_gdf.index.map(food_count).fillna(0).astype(int)

    # Avoid division by zero
    hex_gdf['population'] = hex_gdf['population'].replace(0, np.nan)

    # Compute food points per 1000 residents
    hex_gdf['points_per_1000'] = (hex_gdf['food_points'] / hex_gdf['population']) * 1000

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    hex_gdf.plot(column='points_per_1000',
                 cmap='Greens',
                 edgecolor='gray',
                 linewidth=0.3,
                 legend=True,
                 ax=ax,
                 vmin=0, vmax=15,
                 legend_kwds={'label': "Food places per 1,000 residents"})

    ax.set_title("Food Coverage Density (per 1,000 residents)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("✅ Coverage analysis completed.")

def count_food_points_per_hex(gdf_food: gpd.GeoDataFrame, gdf_hex: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Counts the number of food POIs (e.g., fast food, cafes, restaurants) within each H3 hexagon.

    Parameters:
    -----------
    gdf_food : GeoDataFrame
        GeoDataFrame with food POI geometries.
    gdf_hex : GeoDataFrame
        GeoDataFrame with H3 hexagons containing population density data.

    Returns:
    --------
    GeoDataFrame
        The input gdf_hex enriched with a new column 'food_count'.
    """
    # Spatial join: count how many food POIs fall within each hexagon
    joined = gpd.sjoin(gdf_food, gdf_hex, how="inner", predicate="within")
    food_counts = joined.groupby("index_right").size()

    # Assign the count to the hex DataFrame
    gdf_hex["food_count"] = gdf_hex.index.map(food_counts).fillna(0).astype(int)

    return gdf_hex

# -------------------- 🔹 For Multicriteria Model: Base Matrix Generation per Hexagon --------------------

def generate_voronin_dataframe_basic(gdf_hex: gpd.GeoDataFrame,
                                     gdf_food: gpd.GeoDataFrame,
                                     gdf_poi: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generates the base matrix for the Voronin multi-criteria model.
    Each row corresponds to one hexagon, with columns for: population, food POIs (4 categories), and general POIs (7+ categories).

    Parameters:
    ----------
    gdf_hex : GeoDataFrame
        Hexagons with population.
    gdf_food : GeoDataFrame
        Food POIs (with 'amenity' column).
    gdf_poi : GeoDataFrame
        Points of interest (with 'category' column).

    Returns:
    -------
    DataFrame
        A table with hex-level aggregated values by criteria.
    """
    # Set coordinate systems
    gdf_hex = gdf_hex.to_crs(epsg=4326)
    gdf_food = gdf_food.to_crs(epsg=4326)
    gdf_poi = gdf_poi.to_crs(epsg=4326)

    df = gdf_hex[['geometry', 'population', 'h3_index']].copy()

    food_map = {
        'restaurant': 'count_restaurant',
        'cafe': 'count_cafe',
        'bar': 'count_bar_pub',
        'pub': 'count_bar_pub',
        'fast_food': 'count_fastfood',
        'food_court': 'count_fastfood',
        'canteen': 'count_fastfood'
    }

    poi_map = {
        'University': 'count_university',
        'Market': 'count_market',
        'Transport Hub': 'count_transport',
        'Office / Business Center': 'count_office',
        'Entertainment': 'count_entertainment',
        'Tourism / Beach': 'count_tourism',
        'Sport Facility': 'count_sport',
        'Mall / Supermarket': 'count_mall'
    }

    for col in set(food_map.values()).union(set(poi_map.values())):
        df[col] = 0

    # Aggregate food POIs
    joined_food = gpd.sjoin(gdf_food, gdf_hex, how='inner', predicate='within')
    for amenity_type, col_name in food_map.items():
        mask = joined_food['amenity'] == amenity_type
        counts = joined_food[mask].groupby('index_right').size()
        df.loc[df.index.isin(counts.index), col_name] += counts

    # Aggregate general POIs
    joined_poi = gpd.sjoin(gdf_poi, gdf_hex, how='inner', predicate='within')
    for cat_type, col_name in poi_map.items():
        mask = joined_poi['category'] == cat_type
        counts = joined_poi[mask].groupby('index_right').size()
        df.loc[df.index.isin(counts.index), col_name] += counts

    df["h3_index"] = gdf_hex["h3_index"].values

    return df.reset_index(drop=True)

# -------------------- 🔹 For Multicriteria Model: Extended Matrix with Neighbor Aggregation --------------------

def generate_voronin_dataframe_advanced(gdf_hex: gpd.GeoDataFrame,
                                        gdf_food: gpd.GeoDataFrame,
                                        gdf_poi: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Constructs an extended Voronin matrix that incorporates information from neighboring hexagons.

    Steps:
    1. Uses generate_voronin_dataframe_basic() to create the initial table.
    2. For each hex, computes the sum of each criterion across its neighbors.
    3. Returns a DataFrame with additional *_neigh columns.

    Parameters:
    - gdf_hex: GeoDataFrame with hexagons and population (from Kontur grid).
    - gdf_food: GeoDataFrame with food-related POIs.
    - gdf_poi: GeoDataFrame with POIs and categories.

    Returns:
    - df_voronin_advanced: DataFrame with extended neighbor-aware features.
    """

    # 1. Generate base table
    df_voronin = generate_voronin_dataframe_basic(gdf_hex, gdf_food, gdf_poi)

    # 2. Prepare for neighbor aggregation
    hex_map = df_voronin.set_index("h3_index").to_dict(orient="index")
    criteria_cols = [col for col in df_voronin.columns
        if col != "h3_index" and pd.api.types.is_numeric_dtype(df_voronin[col])]

    # Create structure for aggregated neighbor data
    aggregated_data = {f"{col}_neigh": [] for col in criteria_cols}

    # 3. Loop through each hex
    for h3_index in df_voronin["h3_index"]:
        # Get neighbors excluding the center
        neighbors = h3.k_ring(h3_index, 1) - {h3_index}
        # Filter only existing neighbors
        neighbor_values = [hex_map[n] for n in neighbors if n in hex_map]

        for col in criteria_cols:
            if neighbor_values:
                total_value = np.sum([n[col] for n in neighbor_values])
            else:
                total_value = 0
            # Store aggregated value
            aggregated_data[f"{col}_neigh"].append(total_value)

    # 4. Add aggregated columns to the main DataFrame
    for col in aggregated_data:
        df_voronin[col] = aggregated_data[col]

    return df_voronin

# -------------------- 🔹 Debug Plot: Visualizing Hexagon and Neighbors with Metric Values --------------------

def debug_plot_hex_with_values(df_voronin: pd.DataFrame,
                                target_h3: str,
                                column: str = "count_fastfood") -> None:
    """
    Visualizes the selected hexagon and its neighbors, annotating each with the specified metric.
    The central hex is also annotated with the total sum of that metric across its neighbors.

    Parameters:
    ----------
    df_voronin : pd.DataFrame
        Table containing h3_index and numeric indicator columns.
    target_h3 : str
        The H3 index of the central hexagon to visualize.
    column : str
        Column name to visualize (e.g., 'count_fastfood').
    """

    def h3_to_polygon(h3_index: str) -> Polygon:
        """Inner helper — builds a shapely Polygon from an H3 index"""
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        return Polygon(boundary)

    # 1. Get neighbors including the center (6+1 hexagons)
    neighbors = h3.k_ring(target_h3, 1)
    hex_map = df_voronin.set_index("h3_index").to_dict(orient="index")

    # 2. Build polygons and retrieve metric values
    polygons = [h3_to_polygon(h) for h in neighbors]
    values = [hex_map[h][column] if h in hex_map and column in hex_map[h] else 0 for h in neighbors]

    # 3. Plot visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    for poly, val in zip(polygons, values):
        gpd.GeoSeries([poly]).plot(ax=ax, edgecolor="black", facecolor="lightblue")
        center = poly.centroid
        ax.text(center.x, center.y, str(int(val)), ha='center', va='center', fontsize=9)

    # 4. Annotate the center with the total sum of neighbors
    neighbors_wo_center = [n for n in neighbors if n != target_h3]
    neigh_sum = sum(hex_map[n][column] for n in neighbors_wo_center if n in hex_map)
    center_poly = h3_to_polygon(target_h3)
    center = center_poly.centroid
    ax.text(center.x, center.y, f"∑={neigh_sum}", ha='center', va='center',
            fontsize=11, fontweight="bold", color="red")

    ax.set_title("Check for neighbors hex calculation", fontsize=12)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# -------------------- 🔹 Multicriteria Scoring Model with Penalty (Voronin Adaptation) --------------------

def apply_multicriteria_model_with_penalty(gdf, weights_file_path):
    """
    Applies a modified multicriteria scoring model to a GeoDataFrame of hexagons.

    Unlike the classic Voronin model, this version incorporates a penalty for below-average values,
    centering all metrics around the actual mean value of each criterion.

    Logic:
    ------
    - After normalization, all values are in the [0, 1] range.
    - For each criterion, compute the average → used as a "neutral point".
    - Values > mean → contribute positively.
    - Values < mean → penalized.
    - Formula: **score = (value - mean) * 2 * weight**

    Parameters:
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame of hexagons with numeric criteria to score.
    weights_file_path : str
        Path to Excel file with Min/Max strategies and weights.

    Returns:
    --------
    GeoDataFrame with an added 'VoroninScor' column.
    """

    # 1. Load strategy and weight definitions
    df_weights = pd.read_excel(weights_file_path)

    # 2. Filter only those fields that exist in the GeoDataFrame
    df_weights = df_weights[df_weights['Field_in_data'].isin(gdf.columns)]
    weights = df_weights.set_index('Field_in_data')['Weight'].to_dict()

    # 3. Normalize criteria using min/max logic → scale values to [0, 1]
    # For 'max' strategy: higher values are better → (val - min) / (max - min)
    # For 'min' strategy: lower values are better → (max - val) / (max - min)
    # After normalization, compute the mean of each column — used as the neutral point for scoring
    means = {}
    for _, row in df_weights.iterrows():
        col = row['Field_in_data']
        strategy = str(row['Minimax']).strip().lower()
        min_val = gdf[col].min()
        max_val = gdf[col].max()

        if max_val != min_val:
            if strategy == 'max':
                gdf[col] = (gdf[col] - min_val) / (max_val - min_val)
            elif strategy == 'min':
                gdf[col] = (max_val - gdf[col]) / (max_val - min_val)

        means[col] = gdf[col].mean()  # Store mean for each criterion (used for centered scoring)

    # 4. Compute final VoroninScor:
    # - If value > mean → reward (positive contribution)
    # - If value < mean → penalty (negative contribution)
    # - If value == mean → neutral (no contribution)
    gdf['VoroninScor'] = gdf.apply(
        lambda row: sum(((row[col] - means[col]) * 2) * weights[col] for col in weights.keys()),
        axis=1
    )

    """
    Interpretation of scoring with respect to the actual mean:

    | row[col] value  | Centered value            | Explanation                              |
    |------------------|----------------------------|-------------------------------------------|
    |       1.0        |     (1 - mean) * 2        | Very good, max reward                     |
    |     > mean       |     (val - mean) * 2      | Above average — positive contribution     |
    |       mean       |     (mean - mean) = 0     | Neutral value — no contribution           |
    |     < mean       |     (val - mean) * 2      | Below average — penalty                   |
    |       0.0        |     (0 - mean) * 2        | Worst case relative to the average        |
    """

    # 5. Save results
    gdf.to_csv("hex_voronin_scored.csv", index=False)
    print("✅ Results saved to file: hex_voronin_scored.csv")

    # 6. Show Top-5 hexagons
    top5 = gdf.nlargest(5, 'VoroninScor')
    print("\n🏆 Top-5 locations with highest score (penalized model):")
    for i, row in top5.iterrows():
        print(f"{i + 1}. Hex {row['h3_index']} — Score = {row['VoroninScor']:.3f}")

    # 7. Plot map with the best-scored hex highlighted
    best_hex = gdf.loc[gdf['VoroninScor'].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(column='VoroninScor', ax=ax, cmap='YlOrRd', edgecolor='grey', legend=True)

    # Highlight the best hexagon
    gpd.GeoSeries([best_hex['geometry']]).plot(
        ax=ax, facecolor='none', edgecolor='black', linewidth=2.5, linestyle='--'
    )

    # Add label at the centroid of the best hex
    center = best_hex['geometry'].centroid
    ax.text(center.x, center.y, "+", ha='center', va='center', fontsize=11, fontweight='bold', color='black')

    plt.title("Voronin Score Map with Penalization (best hex highlighted)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("map_voronin_score.png", dpi=300)
    plt.show()
    print("🗺️ Map saved as: map_voronin_score.png")

    # 8. Analyze the contribution of each criterion for the top hex
    print("\n📊 Contribution of each criterion to the best hex score:")
    contribs = {
        col: ((best_hex[col] - 0.5) * 2) * weights[col]
        for col in weights.keys()
    }
    sorted_contribs = dict(sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True))
    for col, val in sorted_contribs.items():
        print(f"{col:30} → contribution: {val:.4f}")

    return gdf

# -------------------- 🔹 Main --------------------

if __name__ == "__main__":
    # 0.1 Load city boundary
    city_boundary = load_city_boundary("Odesa, Ukraine")

    # 0.2 Visualize city boundary
    plot_city_boundary(city_boundary)

    # 0.3 Load city districts
    city_districts = load_city_districts(city_boundary)

    # 1.1 Load food-related locations (fast food, cafes, restaurants, etc.)
    gdf_food = load_food_locations(city_boundary)

    # 1.2 Preprocess geometries (filter out nulls)
    gdf_food = gdf_food[gdf_food.geometry.notnull()].copy()

    # 1.3.1 Visualize fast food points
    plot_food_locations(city_boundary, gdf_food)

    # 1.3.2 Visualize districts and food points together
    plot_districts_and_food(city_boundary, city_districts, gdf_food)

    # 1.4 Analyze number of food points by district
    analyze_food_by_district(gdf_food, city_districts)

    # 1.5 Save filtered food points for visual inspection
    gdf_food[['name', 'addr:street', 'addr:housenumber', 'amenity', 'cuisine', 'geometry']]\
        .sort_values(by=['name']).to_csv("filtered_food_odesa.csv", index=False, encoding='utf-8-sig')
    print("\n📥 Filtered food points saved to: filtered_food_odesa.csv")

    # 2.1 Load and expand POI dataset within city boundary
    city_boundary = ox.geocode_to_gdf("Odessa, Ukraine")
    df_expanded = load_and_expand_poi_locations(city_boundary)

    # 2.2 Visualize POI locations
    plot_poi_locations(df_expanded, city_boundary)

    # 3.1 Load population density from Kontur data
    gdf_hex_real = load_population_density_from_kontur(city_boundary, "kontur_population_UA_20231101.gpkg")

    # 3.2 Visualize population density by hexagons
    plot_population_hexagons(city_boundary, gdf_hex_real)

    # 4.1 Count food points within each hexagon
    gdf_hex_counted = count_food_points_per_hex(gdf_food, gdf_hex_real)

    # 4.2 Visualize population vs food coverage by hex
    analyze_food_coverage_by_population(gdf_food, gdf_hex_real)

    # 5.1 Create base dataset for Voronin multicriteria model
    gdf_hex_real["h3_index"] = gdf_hex_real["h3"]
    df_expanded = pd.read_csv("Expanded_POI_locations_Odesa.csv")
    df_expanded['geometry'] = df_expanded['geometry'].apply(wkt.loads)
    gdf_poi = gpd.GeoDataFrame(df_expanded, geometry='geometry', crs="EPSG:4326")
    df_voronin = generate_voronin_dataframe_basic(gdf_hex_real, gdf_food, gdf_poi)
    df_voronin.to_csv("hex_voronin_matrix.csv", index=False, encoding="utf-8-sig")
    print("✅ Voronin base matrix saved.")

    # 5.2 Create extended dataset including neighbor hexes
    df_voronin_with_neigh = generate_voronin_dataframe_advanced(gdf_hex_real, gdf_food, gdf_poi)
    df_voronin_with_neigh.to_csv("hex_voronin_matrix_with_neigh.csv", index=False, encoding="utf-8-sig")
    print("✅ Voronin extended matrix (with neighbors) saved.")

    # 5.3 Visual debug check: plot food total for the most populated hex
    max_pop_index = df_voronin_with_neigh["population"].idxmax()
    test_hex = df_voronin_with_neigh.loc[max_pop_index, "h3_index"]
    df_voronin_with_neigh["total_food"] = (
        df_voronin_with_neigh["count_fastfood"] +
        df_voronin_with_neigh["count_cafe"] +
        df_voronin_with_neigh["count_restaurant"] +
        df_voronin_with_neigh["count_bar_pub"]
    )
    debug_plot_hex_with_values(df_voronin_with_neigh, test_hex, column="total_food")

    # 6. Apply Voronin multicriteria model with penalty
    print("\n🔍 Applying Voronin multicriteria model...")

    df_matrix = pd.read_csv("hex_voronin_matrix_with_neigh.csv")
    df_matrix['geometry'] = df_matrix['geometry'].apply(wkt.loads)
    gdf_matrix = gpd.GeoDataFrame(df_matrix, geometry='geometry', crs='EPSG:4326')

    # Run the model
    gdf_result = apply_multicriteria_model_with_penalty(
        gdf_matrix,
        "minmax and weights for voronin.xlsx"
    )

    sys.stdout.close()

"""
Analysis of Results – Model Verification and Business Interpretation
--------------------------------------------------------------------

🔹 1. Step-by-step implementation:
- Initially, we collected and filtered geographic data for Odesa, including fast_food, cafes, restaurants, bars,
  and other HoReCa points of interest. Additional POI and population density data were also loaded.
- Using OSM and H3, we generated a spatial grid of hexagons covering the city and aggregated relevant features per hex.
- We developed a mechanism for spatial aggregation of neighboring hexes — enabling the creation of features such as
  "number of venues in adjacent areas", which helps model local density more accurately.
- A multi-criteria decision model (Voronin method) was built, incorporating weights, min/max strategy per criterion,
  and a tunable importance scale.
- The model was enhanced with penalties for poor values and a dynamic centering point based on the real average
  value (not a fixed 0.5).
- A visual scoring map was generated: a hexagon heatmap with top-5 recommendations and a detailed breakdown table
  explaining the contribution of each criterion — this is our final **business recommendation** for opening a fast-food
  location in Odesa.

🔹 2. What we learned / technologies used:
- Acquired practical experience with geospatial data and libraries (`geopandas`, `shapely`, `h3`, `matplotlib`).
- Parsed OSM-tagged data and implemented logic to classify food-related POIs relevant to business needs,
  including in-depth research on tag meanings directly from OpenStreetMap documentation.
- Applied min/max normalization strategies and studied centering logic (0.5 vs real mean), exploring trade-offs.
- Learned to compute spatial aggregates using neighbor-aware feature engineering to improve model robustness.
- Built a fully adapted multi-criteria decision support model (Voronin), tailored to real business decision-making.

🔹 3. R&D-driven approach and business analytics:
- Starting from a realistic business challenge (finding optimal locations for fast-food), we moved beyond
  simplistic fast_food density.
- We extended the model to over 25 spatial criteria, including: population, restaurants, cafes, bars, offices,
  universities, entertainment, malls, markets, transport — within each hex and its neighbors.
- Instead of naïvely centering at 0.5, we introduced a dynamic logic centered around real mean values to reduce
  bias and improve anomaly tolerance in urban patterns.
- We designed the model to help investors see not just “where people are”, but rather “where the best balance
  of factors exists” — low competition, high foot traffic, proximity to key POIs.
- The framework is scalable and adaptable: new criteria or weights can be added seamlessly.

✅ As a result, we developed a full-fledged geospatial CRM sub-system that supports strategic decisions for placing
HoReCa venues (not just fast food) in large cities. The logic is modular, transparent, and extendable to other cities
with minimal adjustments.

"""