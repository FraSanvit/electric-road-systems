"""Utilis."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon
# from scipy.spatial import Voronoi, cKDTree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config as cnf

import json
import pandas as pd


def plot_map(gdf, title=None):
    """
    Plot GeoDataFrame and save PNG next to the parquet result.
    """

    png_path = cnf.result_shape_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    color_map = {"maritime": "lightblue", "land": "steelblue"}

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(
        ax=ax, color=gdf["shape_class"].map(color_map), edgecolor="white", linewidth=0.5
    )

    if title is not None:
        ax.set_title(title, fontsize=14)

    # Axis labels
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)

    # Axis limits
    # ax.set_xlim(-35, 40)
    # ax.set_ylim(30, 75)

    # Tick label size
    ax.tick_params(axis="both", labelsize=14)

    # Title
    if title is not None:
        ax.set_title(title, fontsize=18)

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return png_path


def windfarms_json_to_df(json_path):
    """
    Read a GeoJSON wind farm file and return a pandas DataFrame
    with all properties plus lon and lat columns.

    Parameters
    ----------
    json_path : str or Path
        Path to GeoJSON file

    Returns
    -------
    pd.DataFrame
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for feature in data["features"]:
        props = feature["properties"].copy()

        # Extract coordinates (lon, lat)
        lon, lat = feature["geometry"]["coordinates"]

        props["lon"] = lon
        props["lat"] = lat

        records.append(props)

    df = pd.DataFrame(records)

    return df

def aggregate_wind_capacity(
    wind_df,
    shapes_gdf,
    projection_year,
    lifetime,
):
    """
    Aggregate wind capacity per shape_id with:
    - Status grouping
    - Year reassignment
    - Proper filtering rules
    """

    # Create GeoDataFrame
    wind_gdf = gpd.GeoDataFrame(
        wind_df,
        geometry=gpd.points_from_xy(wind_df["lon"], wind_df["lat"]),
        crs="EPSG:4326",
    )

    if shapes_gdf.crs != wind_gdf.crs:
        wind_gdf = wind_gdf.to_crs(shapes_gdf.crs)

    # Spatial join
    joined = gpd.sjoin(
        wind_gdf,
        shapes_gdf[["shape_id", "country_id", "shape_class", "geometry"]],
        how="left",
        predicate="within",
    )

    # Remove outside points
    joined = joined.dropna(subset=["shape_id"])

    # Clean columns
    joined.columns = joined.columns.str.strip()

    # Convert year to numeric
    joined["year"] = pd.to_numeric(joined["year"], errors="coerce")

    # Assign artificial years
    joined.loc[joined["status"].isin(["Construction", "Approved"]), "year"] = 2030
    joined.loc[joined["status"] == "Planned", "year"] = 2035

    # Remove dismantled
    joined = joined[joined["status"] != "Dismantled"]

    # Create grouped status column
    joined["status_group"] = joined["status"].map(
        {
            "Production": "prod_con_app",
            "Construction": "prod_con_app",
            "Approved": "prod_con_app",
            "Planned": "planned",
        }
    )

    # Apply filtering ONLY to prod_con_app
    joined_filtered = joined[
        (
            (joined["status_group"] == "planned")
            |
            (
                (joined["status_group"] == "prod_con_app") &
                (joined["year"] >= (projection_year - lifetime))
            )
        )
    ]

    # Reassign land → maritime
    mask_land = joined_filtered["shape_id"].astype(str).str.endswith("_land")
    joined_filtered.loc[mask_land, "shape_id"] = (
        joined_filtered.loc[mask_land, "shape_id"]
        .str.replace("_land", "_maritime", regex=False)
    )
    joined_filtered.loc[mask_land, "shape_class"] = "maritime"

    # Aggregate
    agg = (
        joined_filtered
        .groupby(
            ["shape_id", "country_id", "shape_class", "status_group"]
        )["power_mw"]
        .sum()
        .reset_index()
        .rename(columns={"power_mw": "output_capacity_mw"})
    )

    # Technology
    agg["technology"] = agg["shape_class"].map(
        {"land": "onshore", "maritime": "offshore"}
    )

    # Category
    agg["category"] = "wind"

    result = agg[
        [
            "shape_id",
            "output_capacity_mw",
            "country_id",
            "technology",
            "category",
            "status_group",
        ]
    ]

    return result


def plot_wind_farms_emodnet(wind_df, shapes_gdf):
    """
    Plot wind farm points over polygon shapes.
    
    Parameters
    ----------
    wind_df : pd.DataFrame
        Must contain lon, lat
    
    shapes_gdf : gpd.GeoDataFrame
        Must contain geometry and shape_class
    """

    # Convert wind dataframe to GeoDataFrame
    wind_gdf = gpd.GeoDataFrame(
        wind_df,
        geometry=gpd.points_from_xy(wind_df["lon"], wind_df["lat"]),
        crs="EPSG:4326",
    )

    # Reproject if necessary
    if shapes_gdf.crs != wind_gdf.crs:
        wind_gdf = wind_gdf.to_crs(shapes_gdf.crs)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color polygons by class
    color_map = {"land": "lightgrey", "maritime": "lightblue"}

    shapes_gdf.plot(
        ax=ax,
        color=shapes_gdf["shape_class"].map(color_map),
        edgecolor="white",
        linewidth=0.5,
    )

    color_map_wind = {'Production':"red", 'Dismantled': "black", 'Planned': "green", 'Approved': "gold", 'Construction': "orange"}

    # Plot wind farm points
    wind_gdf.plot(
        ax=ax,
        color=wind_gdf["status"].map(color_map_wind),
        markersize=10,
        alpha=0.7,
    )

    # ---- Create Legend ----
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=status,
            markerfacecolor=color,
            markersize=8,
        )
        for status, color in color_map_wind.items()
    ]

    ax.legend(
        handles=legend_elements,
        title="Wind Farm Status",
        fontsize=12,
        title_fontsize=13,
        loc="upper right",
    )

    ax.set_title("EMODnet Offshore Wind", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()


def extract_wind_farms_from_har(har_path):
    """
    Extract wind farm JSON data from a HAR file and return a pandas DataFrame.
    """
    with open(har_path, "r", encoding="utf-8") as f:
        har_data = json.load(f)

    entries = har_data.get("log", {}).get("entries", [])

    for entry in entries:
        request_url = entry["request"]["url"]
        if "wind_farms.json" in request_url:
            response_content = entry["response"]["content"]

            if "text" in response_content:
                text = response_content["text"]
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    print(f"Invalid JSON in {har_path}")
                    return None

                print(f"{har_path}: Found {len(data)} wind farm entries")
                return pd.DataFrame(data)

    print(f"{har_path}: No wind_farms.json response found")
    return None


def load_wind_farms_from_har_folder(folder_path):
    """
    Iterate over all HAR files in a folder, merge them into a single DataFrame,
    and remove fully duplicate rows, handling unhashable types.
    """
    all_dfs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            df = extract_wind_farms_from_har(file_path)
            if df is not None:
                all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Convert unhashable columns (lists/dicts) to strings
        for col in merged_df.columns:
            if merged_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                merged_df[col] = merged_df[col].apply(lambda x: str(x))

        # Remove fully duplicate rows
        merged_df = merged_df.drop_duplicates(ignore_index=True)

        print(f"Merged {len(all_dfs)} files into {len(merged_df)} total unique entries")
        return merged_df
    else:
        print("No valid HAR files found.")
        return pd.DataFrame()


def allocate_wind_to_shapes(df_wind, gdf_shapes, projection_year, lifetime):
    """
    Allocate offshore wind farm capacity to polygons (shapes).

    Parameters
    ----------
    df_wind : pd.DataFrame
        Wind farms with columns: latitude, longitude, totalCapacity (in kW)
    gdf_shapes : gpd.GeoDataFrame
        Polygons with columns: 'shape_id', 'country_id', 'geometry'

    Returns
    -------
    pd.DataFrame
        Aggregated wind capacities per shape:
        shape_id, output_capacity_mw, country_id, technology, category
    """
    # Convert wind farms to GeoDataFrame
    gdf_wind = gpd.GeoDataFrame(
        df_wind.copy(),
        geometry=gpd.points_from_xy(df_wind.longitude, df_wind.latitude),
        crs=gdf_shapes.crs
    )

    # Spatial join: assign each wind farm to a shape
    gdf_joined = gpd.sjoin(
        gdf_wind,
        gdf_shapes[['shape_id', 'country_id', 'geometry']],
        how='inner',
        predicate='within'
    )

    # Reassign land → maritime
    mask_land = gdf_joined["shape_id"].astype(str).str.endswith("_land")
    gdf_joined.loc[mask_land, "shape_id"] = (
        gdf_joined.loc[mask_land, "shape_id"]
        .str.replace("_land", "_maritime", regex=False)
    )
    gdf_joined.loc[mask_land, "shape_class"] = "maritime"

    # Convert kW → MW
    gdf_joined['output_capacity_mw'] = gdf_joined['totalCapacity'] / 1_000_000

    # Assign technology and category
    gdf_joined['technology'] = 'offshore'
    gdf_joined['category'] = 'wind'

    gdf_joined_filtered = gdf_joined[(gdf_joined["commissioningYear"]<=projection_year) & (gdf_joined["commissioningYear"]>=(projection_year-lifetime))]

    # Aggregate capacities per shape
    agg_cols = ['shape_id', 'country_id', 'technology', 'category']
    df_agg = gdf_joined_filtered.groupby(agg_cols, as_index=False)['output_capacity_mw'].sum()

    df_agg["shape_id"] = (
            df_agg["shape_id"]
            .str.split("_", n=2)
            .str[:2]
            .str.join("_")
        )

    return df_agg


def plot_wind_farms_windeurope(df_wind, gdf_shapes, shape_id_col="shape_id"):
    """
    Plot wind farm points on top of polygon shapes.
    
    Parameters
    ----------
    df_wind : pd.DataFrame
        DataFrame containing wind farm points with 'latitude' and 'longitude' columns.
    gdf_shapes : gpd.GeoDataFrame
        GeoDataFrame with polygons representing the shapes.
    shape_id_col : str
        Name of the column in gdf_shapes to use as shape identifier.
    """
    
    # Drop rows with missing lat/lon
    df_wind_clean = df_wind.dropna(subset=['latitude', 'longitude']).copy()
    
    # Convert wind farms DataFrame to GeoDataFrame
    gdf_wind = gpd.GeoDataFrame(
        df_wind_clean,
        geometry=[Point(xy) for xy in zip(df_wind_clean['longitude'], df_wind_clean['latitude'])],
        crs="EPSG:4326"  # WGS84 lat/lon
    )
    
    # Ensure both are in the same CRS
    if gdf_shapes.crs != gdf_wind.crs:
        gdf_shapes = gdf_shapes.to_crs(gdf_wind.crs)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    color_map = {"land": "lightgrey", "maritime": "lightblue"}

    gdf_shapes.plot(
        ax=ax,
        color=gdf_shapes["shape_class"].map(color_map),
        edgecolor="white",
        alpha=0.8
    )

    gdf_wind.plot(ax=ax, color="red", markersize=10, alpha=0.7, label="Wind Farms")
    
    # # Optional: add shape IDs as labels
    # for idx, row in gdf_shapes.iterrows():
    #     x, y = row.geometry.centroid.x, row.geometry.centroid.y
    #     ax.text(x, y, str(row[shape_id_col]), fontsize=8, ha="center", va="center")
    
    plt.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("WindEurope Offshore Wind", fontsize=16)
    plt.show()



# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_two_dfs_side_by_side_hatched_vertical_national(df1, df2, label1="DF1", label2="DF2"):
    """
    Vertical grouped bar plot for two DataFrames with DF1 hatched.
    Each shape_id gets two bars side by side.
    If status_group exists, bars are stacked.
    """
    all_shapes = sorted(set(df1['country_id']).union(df2['country_id']))
    bar_width = 0.4
    x_pos = np.arange(len(all_shapes))

    fig, ax = plt.subplots(figsize=(9, 5))

    def plot_df(df, x_pos, offset, label_prefix, hatch=None):
        if "status_group" in df.columns:
            agg = df.groupby(["country_id", "status_group"])["output_capacity_mw"].sum().unstack(fill_value=0)
            color_map = {"prod_con_app": "steelblue", "planned": "orange"}

            for i, status in enumerate(agg.columns):
                values = agg[status].reindex(all_shapes, fill_value=0)
                bottom = np.zeros(len(values))
                if i > 0:
                    bottom = agg.iloc[:, :i].reindex(all_shapes, fill_value=0).sum(axis=1)
                ax.bar(
                    x_pos + offset,
                    values,
                    bottom=bottom,
                    width=bar_width,
                    label=f"{label_prefix}: {status}" if offset == -bar_width/2 else None,
                    color=color_map.get(status, "grey"),
                    hatch=hatch
                )
        else:
            values = df.groupby("country_id")["output_capacity_mw"].sum().reindex(all_shapes, fill_value=0)
            ax.bar(
                x_pos + offset,
                values,
                width=bar_width,
                label=label_prefix,
                color="steelblue" if offset == -bar_width/2 else "orange",
                hatch=hatch
            )

    # Plot DF1 with hatch
    plot_df(df1, x_pos, offset=-bar_width/2, label_prefix=label1, hatch="//")
    # Plot DF2 without hatch
    plot_df(df2, x_pos, offset=bar_width/2, label_prefix=label2, hatch=None)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_shapes, rotation=45, ha='right')
    ax.set_ylabel("Output capacity (MW)")
    ax.set_xlabel("Shape ID")
    ax.set_title("Wind capacity comparison by shape", fontsize=16)

    # Reduce horizontal space at edges
    ax.set_xlim(-0.5, len(all_shapes)-0.5)

    ax.legend()
    plt.tight_layout()
    plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_two_dfs_side_by_side_hatched_vertical(df1, df2, label1="DF1", label2="DF2"):
    """
    Vertical grouped bar plot for two DataFrames with DF1 hatched.
    Each shape_id gets two bars side by side.
    If status_group exists, bars are stacked.
    """
    all_shapes = sorted(set(df1['shape_id']).union(df2['shape_id']))
    bar_width = 0.4
    x_pos = np.arange(len(all_shapes))

    fig, ax = plt.subplots(figsize=(18, 5))

    def plot_df(df, x_pos, offset, label_prefix, hatch=None):
        if "status_group" in df.columns:
            agg = df.groupby(["shape_id", "status_group"])["output_capacity_mw"].sum().unstack(fill_value=0)
            color_map = {"prod_con_app": "steelblue", "planned": "orange"}

            for i, status in enumerate(agg.columns):
                values = agg[status].reindex(all_shapes, fill_value=0)
                bottom = np.zeros(len(values))
                if i > 0:
                    bottom = agg.iloc[:, :i].reindex(all_shapes, fill_value=0).sum(axis=1)
                ax.bar(
                    x_pos + offset,
                    values,
                    bottom=bottom,
                    width=bar_width,
                    label=f"{label_prefix}: {status}" if offset == -bar_width/2 else None,
                    color=color_map.get(status, "grey"),
                    hatch=hatch
                )
        else:
            values = df.groupby("shape_id")["output_capacity_mw"].sum().reindex(all_shapes, fill_value=0)
            ax.bar(
                x_pos + offset,
                values,
                width=bar_width,
                label=label_prefix,
                color="steelblue" if offset == -bar_width/2 else "orange",
                hatch=hatch
            )

    # Plot DF1 with hatch
    plot_df(df1, x_pos, offset=-bar_width/2, label_prefix=label1, hatch="//")
    # Plot DF2 without hatch
    plot_df(df2, x_pos, offset=bar_width/2, label_prefix=label2, hatch=None)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_shapes, rotation=45, ha='right')
    ax.set_ylabel("Output capacity (MW)")
    ax.set_xlabel("Shape ID")
    ax.set_title("Wind capacity comparison by shape", fontsize=16)

    # Reduce horizontal space at edges
    ax.set_xlim(-0.5, len(all_shapes)-0.5)

    ax.legend()
    plt.tight_layout()
    plt.show()