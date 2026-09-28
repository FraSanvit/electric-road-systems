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
import pycountry
import pytz
from shapely import wkt
import re
import yaml
from collections import defaultdict
from ruamel.yaml import YAML


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
            | (
                (joined["status_group"] == "prod_con_app")
                & (joined["year"] >= (projection_year - lifetime))
            )
        )
    ]

    # Reassign land → maritime
    mask_land = joined_filtered["shape_id"].astype(str).str.endswith("_land")
    joined_filtered.loc[mask_land, "shape_id"] = joined_filtered.loc[
        mask_land, "shape_id"
    ].str.replace("_land", "_maritime", regex=False)
    joined_filtered.loc[mask_land, "shape_class"] = "maritime"

    # Aggregate
    agg = (
        joined_filtered.groupby(
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

    color_map_wind = {
        "Production": "red",
        "Dismantled": "black",
        "Planned": "green",
        "Approved": "gold",
        "Construction": "orange",
    }

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
        crs=gdf_shapes.crs,
    )

    # Spatial join: assign each wind farm to a shape
    gdf_joined = gpd.sjoin(
        gdf_wind,
        gdf_shapes[["shape_id", "country_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Reassign land → maritime
    mask_land = gdf_joined["shape_id"].astype(str).str.endswith("_land")
    gdf_joined.loc[mask_land, "shape_id"] = gdf_joined.loc[
        mask_land, "shape_id"
    ].str.replace("_land", "_maritime", regex=False)
    gdf_joined.loc[mask_land, "shape_class"] = "maritime"

    # Convert kW → MW
    gdf_joined["output_capacity_mw"] = gdf_joined["totalCapacity"] / 1_000_000

    # Assign technology and category
    gdf_joined["technology"] = "offshore"
    gdf_joined["category"] = "wind"

    gdf_joined_filtered = gdf_joined[
        (gdf_joined["commissioningYear"] <= projection_year)
        & (gdf_joined["commissioningYear"] >= (projection_year - lifetime))
    ]

    # Aggregate capacities per shape
    agg_cols = ["shape_id", "country_id", "technology", "category"]
    df_agg = gdf_joined_filtered.groupby(agg_cols, as_index=False)[
        "output_capacity_mw"
    ].sum()

    df_agg["shape_id"] = df_agg["shape_id"].str.split("_", n=2).str[:2].str.join("_")

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
    df_wind_clean = df_wind.dropna(subset=["latitude", "longitude"]).copy()

    # Convert wind farms DataFrame to GeoDataFrame
    gdf_wind = gpd.GeoDataFrame(
        df_wind_clean,
        geometry=[
            Point(xy)
            for xy in zip(df_wind_clean["longitude"], df_wind_clean["latitude"])
        ],
        crs="EPSG:4326",  # WGS84 lat/lon
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
        alpha=0.8,
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


def plot_two_dfs_side_by_side_hatched_vertical_national(
    df1, df2, label1="DF1", label2="DF2"
):
    """
    Vertical grouped bar plot for two DataFrames with DF1 hatched.
    Each shape_id gets two bars side by side.
    If status_group exists, bars are stacked.
    """
    all_shapes = sorted(set(df1["country_id"]).union(df2["country_id"]))
    bar_width = 0.4
    x_pos = np.arange(len(all_shapes))

    fig, ax = plt.subplots(figsize=(9, 5))

    def plot_df(df, x_pos, offset, label_prefix, hatch=None):
        if "status_group" in df.columns:
            agg = (
                df.groupby(["country_id", "status_group"])["output_capacity_mw"]
                .sum()
                .unstack(fill_value=0)
            )
            color_map = {"prod_con_app": "steelblue", "planned": "orange"}

            for i, status in enumerate(agg.columns):
                values = agg[status].reindex(all_shapes, fill_value=0)
                bottom = np.zeros(len(values))
                if i > 0:
                    bottom = (
                        agg.iloc[:, :i].reindex(all_shapes, fill_value=0).sum(axis=1)
                    )
                ax.bar(
                    x_pos + offset,
                    values,
                    bottom=bottom,
                    width=bar_width,
                    label=(
                        f"{label_prefix}: {status}"
                        if offset == -bar_width / 2
                        else None
                    ),
                    color=color_map.get(status, "grey"),
                    hatch=hatch,
                )
        else:
            values = (
                df.groupby("country_id")["output_capacity_mw"]
                .sum()
                .reindex(all_shapes, fill_value=0)
            )
            ax.bar(
                x_pos + offset,
                values,
                width=bar_width,
                label=label_prefix,
                color="steelblue" if offset == -bar_width / 2 else "orange",
                hatch=hatch,
            )

    # Plot DF1 with hatch
    plot_df(df1, x_pos, offset=-bar_width / 2, label_prefix=label1, hatch="//")
    # Plot DF2 without hatch
    plot_df(df2, x_pos, offset=bar_width / 2, label_prefix=label2, hatch=None)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_shapes, rotation=45, ha="right")
    ax.set_ylabel("Output capacity (MW)")
    ax.set_xlabel("Shape ID")
    ax.set_title("Wind capacity comparison by shape", fontsize=16)

    # Reduce horizontal space at edges
    ax.set_xlim(-0.5, len(all_shapes) - 0.5)

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
    all_shapes = sorted(set(df1["shape_id"]).union(df2["shape_id"]))
    bar_width = 0.4
    x_pos = np.arange(len(all_shapes))

    fig, ax = plt.subplots(figsize=(18, 5))

    def plot_df(df, x_pos, offset, label_prefix, hatch=None):
        if "status_group" in df.columns:
            agg = (
                df.groupby(["shape_id", "status_group"])["output_capacity_mw"]
                .sum()
                .unstack(fill_value=0)
            )
            color_map = {"prod_con_app": "steelblue", "planned": "orange"}

            for i, status in enumerate(agg.columns):
                values = agg[status].reindex(all_shapes, fill_value=0)
                bottom = np.zeros(len(values))
                if i > 0:
                    bottom = (
                        agg.iloc[:, :i].reindex(all_shapes, fill_value=0).sum(axis=1)
                    )
                ax.bar(
                    x_pos + offset,
                    values,
                    bottom=bottom,
                    width=bar_width,
                    label=(
                        f"{label_prefix}: {status}"
                        if offset == -bar_width / 2
                        else None
                    ),
                    color=color_map.get(status, "grey"),
                    hatch=hatch,
                )
        else:
            values = (
                df.groupby("shape_id")["output_capacity_mw"]
                .sum()
                .reindex(all_shapes, fill_value=0)
            )
            ax.bar(
                x_pos + offset,
                values,
                width=bar_width,
                label=label_prefix,
                color="steelblue" if offset == -bar_width / 2 else "orange",
                hatch=hatch,
            )

    # Plot DF1 with hatch
    plot_df(df1, x_pos, offset=-bar_width / 2, label_prefix=label1, hatch="//")
    # Plot DF2 without hatch
    plot_df(df2, x_pos, offset=bar_width / 2, label_prefix=label2, hatch=None)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_shapes, rotation=45, ha="right")
    ax.set_ylabel("Output capacity (MW)")
    ax.set_xlabel("Shape ID")
    ax.set_title("Wind capacity comparison by shape", fontsize=16)

    # Reduce horizontal space at edges
    ax.set_xlim(-0.5, len(all_shapes) - 0.5)

    ax.legend()
    plt.tight_layout()
    plt.show()


def read_transmission(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")  # read header

        for line in f:
            # extract geometry
            geom_match = re.search(r"LINESTRING\([^\)]*\)", line)
            geom = geom_match.group(0) if geom_match else None

            # remove geometry from the line
            line_clean = re.sub(r"'?LINESTRING\([^\)]*\)'?", "", line)

            # split remaining columns
            parts = line_clean.strip().split(",")

            rows.append(parts + [geom])

    df = pd.DataFrame(rows)
    df["geometry"] = df.iloc[:, -1]

    # fix formatting of coordinates
    df["geometry"] = df["geometry"].str.replace(
        r"(\d+\.\d+\s\d+\.\d+)\s(\d+\.\d+\s\d+\.\d+)", r"\1, \2", regex=True
    )

    def extract_geometry(row):
        for val in row:
            if isinstance(val, str) and val.strip().startswith("LINESTRING"):
                return val.strip()
        return None

    df["geometry"] = df.apply(extract_geometry, axis=1)
    df["geometry"] = df["geometry"].str.replace("'", "", regex=False)
    df["geometry"] = df["geometry"].str.replace(
        r"(\d+\.\d+\s\d+\.\d+)\s(\d+\.\d+\s\d+\.\d+)", r"\1, \2", regex=True
    )

    # drop rows without geometry
    df = df[df["geometry"].notna()]

    # --- Extract voltage symbol ---
    def extract_symbol(row):
        for val in row:
            if isinstance(val, str) and '"symbol"' in val:
                match = re.search(r'"symbol"=>"([^"]+)', val)
                if match:
                    return match.group(1).strip()
        return None

    df["voltage_symbol"] = df.apply(extract_symbol, axis=1)
    # -------------------------------

    # convert to geometry
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf


def expand_yaml_locs(file_name, subregion_list, output_path=None):
    """
    Expand country-level locs in YAML to subregions while preserving formatting.
    Handles:
    - locs: [AUT] -> expanded to [AUT_1, AUT_2, ...]
    - comma-separated location keys: "ALB,AUT" -> separate keys
    - dotted keys: "ALB.techs" -> expand "ALB" to ALB_1, ALB_2, etc.
    """

    input_path = cnf.PROJECT_ROOT / f"resources/yaml-files/{file_name}.yaml"

    if output_path is None:
        output_path = cnf.PROJECT_ROOT / f"results/yaml-files/{file_name}.yaml"
    else:
        output_path = Path(output_path)

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=4, sequence=4, offset=2)

    with open(input_path) as f:
        data = yaml.load(f)

    # Build mapping: AUT -> [AUT_1, AUT_2, ...]
    mapping = defaultdict(list)
    for loc in subregion_list:
        country = loc.split("_")[0]
        mapping[country].append(loc)

    def expand(obj):
        if isinstance(obj, dict):
            keys_to_replace = []
            for k, v in obj.items():
                # Case 1: locs list
                if k == "locs" and isinstance(v, list) and v:
                    country = v[0]
                    if country in mapping:
                        v.clear()
                        v.extend(mapping[country])
                        if hasattr(v, "fa"):
                            v.fa.set_flow_style()
                # Case 2: comma-separated keys
                elif isinstance(k, str) and "," in k and isinstance(v, dict):
                    new_entries = {}
                    for c in k.split(","):
                        c = c.strip()
                        if c in mapping:
                            for sub in mapping[c]:
                                new_entries[sub] = deepcopy(v)
                        else:
                            new_entries[c] = deepcopy(v)
                    keys_to_replace.append((k, new_entries))
                # Case 3: dotted keys like ALB.techs
                elif isinstance(k, str) and "." in k and isinstance(v, dict):
                    country, rest = k.split(".", 1)
                    if country in mapping:
                        new_entries = {}
                        for sub in mapping[country]:
                            new_entries[f"{sub}.{rest}"] = deepcopy(v)
                        keys_to_replace.append((k, new_entries))
                else:
                    expand(v)
            # Apply key replacements
            for old_k, new_dict in keys_to_replace:
                obj.pop(old_k)
                obj.update(new_dict)
        elif isinstance(obj, list):
            for item in obj:
                expand(item)

    expand(data)

    with open(output_path, "w") as f:
        yaml.dump(data, f)

    print(f"Saved expanded YAML to: {output_path}")


def transport_shares_to_df(file_name):
    """
    Reads a YAML file with annual transport distances and computes
    the share (%) per location for each tech (heavy/light transport).

    Returns a DataFrame with columns: locs, techs, unit, value
    """

    yaml = YAML()
    yaml.preserve_quotes = True

    file_path = cnf.PROJECT_ROOT / f"resources/yaml-files/{file_name}.yaml"

    with open(file_path) as f:
        data = yaml.load(f)

    # Navigate to group_constraints
    gc = data["overrides"]["annual_transport_distance"]["group_constraints"]

    rows = []
    # Collect values by country and tech
    country_tech_values = defaultdict(lambda: defaultdict(list))

    for key, entry in gc.items():
        loc = entry["locs"][0]  # ALB_1, AUT_2, etc.
        tech = entry["techs"][0]  # demand_heavy_transport / demand_light_transport

        # carrier_con_equals only has one entry
        value = list(entry["carrier_con_equals"].values())[0]

        # Store absolute value for share calculation
        country = loc.split("_")[0]  # ALB, AUT
        country_tech_values[country][tech].append((loc, abs(value)))

    # Compute shares
    for country, tech_dict in country_tech_values.items():
        for tech, loc_values in tech_dict.items():
            total = sum(v for _, v in loc_values)
            for loc, v in loc_values:
                share = v / total if total != 0 else 0
                rows.append(
                    {"locs": loc, "techs": tech, "unit": "perc", "value": share}
                )

    df = pd.DataFrame(rows)
    return df


def disaggregate_transport(df, df_share):
    """
    df: original dataframe (vehicle_type, country_code, year, value)
    df_shares: output from previous function (locs, techs, unit, value)

    Returns: DataFrame with columns (techs, locs, year, unit, value)
    """

    df = df.copy()

    # 1. Map vehicle types → techs
    df["techs"] = df["vehicle_type"].map(cnf.DEMAND_DICT)

    # 2. Map each tech to share category
    heavy_group = {"bus_transport", "heavy_duty_transport"}
    light_group = {
        "light_duty_transport",
        "motorcycle_transport",
        "passenger_car_transport",
    }

    def map_share_group(tech):
        if tech in heavy_group:
            return "demand_heavy_transport"
        elif tech in light_group:
            return "demand_light_transport"
        else:
            return None

    df["share_tech"] = df["techs"].apply(map_share_group)

    # 3. Prepare shares
    shares = df_share.copy()
    shares = shares.rename(columns={"value": "share"})
    shares["country"] = shares["locs"].str.split("_").str[0]

    # 4. Merge
    df = df.merge(
        shares,
        left_on=["country_code", "share_tech"],
        right_on=["country", "techs"],
        how="left",
        suffixes=("", "_share"),
    )

    # 5. Apply shares
    df["value"] = df["value"] * df["share"]

    # 6. Final formatting
    df_out = df[["techs", "locs", "year", "value"]].copy()
    df_out["unit"] = "Mvkm"

    # reorder columns
    df_out = df_out[["techs", "locs", "year", "unit", "value"]]

    return df_out


def save_hourly_profiles_by_tech(df_yearly, target_year):
    """
    df_yearly: DataFrame with (techs, locs, year, unit, value)
    target_year: int (e.g. 2018)

    Output:
        One CSV per tech → <output_dir>/<tech>.csv
    """

    output_dir = cnf.PROJECT_ROOT / "results/timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 👉 filter yearly data
    df_yearly = df_yearly[df_yearly["year"] == target_year].copy()

    for tech in df_yearly["techs"].unique():

        df_tech = df_yearly[df_yearly["techs"] == tech].copy()

        # build filename
        filename = tech.replace("_", "-") + "-demand.csv"
        filepath = cnf.PROJECT_ROOT / f"resources/transport-data/{filename}"

        # read profile
        prof = pd.read_csv(filepath, parse_dates=["datetime"])
        prof = prof.rename(columns={"datetime": "timesteps"})

        # 👉 filter profile to selected year
        prof = prof[prof["timesteps"].dt.year == target_year].copy()

        # melt
        prof_long = prof.melt(
            id_vars="timesteps", var_name="country", value_name="profile_value"
        )

        # compute shares
        prof_long["abs_val"] = prof_long["profile_value"].abs()
        prof_long["share"] = prof_long.groupby("country")["abs_val"].transform(
            lambda x: x / x.sum() if x.sum() != 0 else 0
        )

        # map country
        df_tech["country"] = df_tech["locs"].str.split("_").str[0]

        # merge
        df_tech = df_tech.merge(
            prof_long[["timesteps", "country", "share"]], on="country", how="left"
        )

        # apply shares
        df_tech["value"] = df_tech["value"] * df_tech["share"] / 100
        df_tech["unit"] = "100 Mvkm"

        # demand sign
        df_tech["value"] = -df_tech["value"].abs()

        # pivot → final structure
        df_out = df_tech.pivot_table(
            index="timesteps", columns="locs", values="value", aggfunc="sum"
        )

        # save
        out_file = output_dir / f"{filename}"
        df_out.to_csv(out_file)

        print(f"Saved: {out_file}")


def expand_profiles_to_subregions(file_list, subregion_list, target_year):
    """
    Expand country-level hourly profiles to subregions.

    Parameters
    ----------
    file_list : list
        List of CSV filenames (e.g. ["passenger-car-transport-demand.csv"])
    subregion_list : list
        List like ["AUT_1", "AUT_2", ...]
    target_year : int
        Year to filter (e.g. 2018)

    Output
    ------
    Saves expanded CSVs in same folder (or adjust path below)
    """

    input_dir = cnf.PROJECT_ROOT / "resources/transport-data"
    output_dir = cnf.PROJECT_ROOT / "results/timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)

    # build mapping: AUT → [AUT_1, AUT_2, ...]
    mapping = defaultdict(list)
    for loc in subregion_list:
        country = loc.split("_")[0]
        mapping[country].append(loc)

    for file_name in file_list:

        filepath = input_dir / file_name

        # read CSV
        df = pd.read_csv(filepath, parse_dates=["datetime"])

        # filter year
        df = df[df["datetime"].dt.year == target_year].copy()

        # start new dataframe with datetime
        df_out = pd.DataFrame()
        df_out["datetime"] = df["datetime"]

        # expand each country column
        for col in df.columns:
            if col == "datetime":
                continue

            if col in mapping:
                # replicate to subregions
                for sub in mapping[col]:
                    df_out[sub] = df[col].values
            else:
                # keep as-is if no mapping exists
                df_out[col] = df[col].values

        # save
        out_file = output_dir / file_name
        df_out.to_csv(out_file, index=False)

        print(f"Saved expanded file: {out_file}")


def load_brownfield_parquets(add_source=False):
    """
    Read and merge all parquet files in resources/brownfield/

    Parameters
    ----------
    add_source : bool
        If True, adds a column with the filename origin

    Returns
    -------
    pd.DataFrame
    """

    folder = cnf.PROJECT_ROOT / "resources/brownfield"

    parquet_files = list(folder.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    dfs = []

    for file in parquet_files:
        df = pd.read_parquet(file)

        if add_source:
            df["source_file"] = file.name

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["cat_tech"] = (
        df_all["category"] + "_" + df_all["technology"].str.replace(" ", "_")
    )
    df_filter = df_all[df_all["output_capacity_mw"] > 0]
    return df_filter


TECH_DICT = {
    "bioenergy_bioenergy_turbine": "chp_biofuel_extraction",
    "fossil_coal_turbine": "coal_power_plant",
    "fossil_ccgt": "ccgt",
    "fossil_ocgt": "ccgt",
    "fossil_steam_turbine": "ccgt",
    "fossil_reciprocating_engine": "ccgt",
    "hydropower_pumped_storage": "",
    "hydropower_reservoir": "",
    "hydropower_run_of_river": "",
    "nuclear_nuclear_reactor": "nuclear",
    "solar_rooftop_pv": "roof_mounted_pv",
    "solar_concentrating_solar_power": "",
    "solar_utility_pv": "open_field_pv",
    "wind_offshore": "wind_offshore",
    "wind_onshore": "wind_onshore",
}


def map_techs(df, tech_dict):
    """
    Map technologies based on TECH_DICT and CHP rule.

    Rules
    -----
    1. Use `cat_tech` to map via TECH_DICT
    2. If mapping == "" → drop row
    3. If chp == True → force tech = "chp_methane_extraction"

    Returns
    -------
    pd.DataFrame with mapped 'techs' column
    """

    df = df.copy()

    # 1. Map using dict
    df["techs"] = df["cat_tech"].map(tech_dict)

    # 2. Drop rows with empty or missing mapping
    df = df[df["techs"].notna() & (df["techs"] != "")]

    # 3. Apply CHP override
    df.loc[df["chp"] == True, "techs"] = "chp_methane_extraction"

    df_grouped = (
        df.groupby(["shape_id", "country_id", "techs"])["output_capacity_mw"]
        .sum()
        .reset_index()
    )

    return df_grouped


def write_brownfield_override(df_mapped, output_file):
    """
    Write a Calliope brownfield_capacity override YAML from df_mapped.

    Parameters
    ----------
    df_mapped : pd.DataFrame
        Must contain columns: ['shape_id', 'techs', 'output_capacity_mw']
    output_file : str or Path
        Path to save the YAML
    """
    from pathlib import Path

    output_file = Path(cnf.PROJECT_ROOT / f"results/{output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("overrides:\n")
        f.write("    brownfield_capacity:\n")
        f.write("        group_constraints:\n")

        for _, row in df_mapped.iterrows():
            loc = row["shape_id"]
            tech = row["techs"]
            cap_min = row["output_capacity_mw"] / 100_000  # scale

            key = f"{tech}_{loc}"

            # Write YAML with 4-space indentation
            f.write(f"            {key}:\n")
            f.write(f"                techs: [{tech}]\n")
            f.write(f"                locs: [{loc}]\n")  # only country code
            f.write(
                f"                energy_cap_min: {format(cap_min, 'f')} # (100'000 MW)\n\n"
            )

    print(f"Brownfield override YAML saved to: {output_file}")


from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from copy import deepcopy


def save_demand_share_yaml_literal(
    list_subregions, DEMAND_SHARE_TRANSPORT_DICT, file_name
):
    """
    Create YAML overrides with:
    - techs and locs as [ ... ] literal lists
    - demand_share_per_timestep_decision.<tech>: .inf (unquoted)
    - 4-space indentation
    - no anchors/references
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=4, offset=0)  # <-- 4 spaces everywhere
    yaml.width = 4096

    overrides_dict = {"overrides": {}}

    for demand_tech, transport_tech_list in DEMAND_SHARE_TRANSPORT_DICT.items():
        key_top = f"demand_share_{demand_tech}"
        overrides_dict["overrides"][key_top] = {"group_constraints": {}}

        for loc in list_subregions:
            subkey = f"{loc}_demand_share_{demand_tech}"
            # force [ ... ] flow style for lists
            techs_list = CommentedSeq(deepcopy(transport_tech_list))
            techs_list.fa.set_flow_style()
            locs_list = CommentedSeq([loc])
            locs_list.fa.set_flow_style()

            overrides_dict["overrides"][key_top]["group_constraints"][subkey] = {
                "techs": techs_list,
                "locs": locs_list,
                f"demand_share_per_timestep_decision.{demand_tech}": float(
                    "inf"
                ),  # <-- unquoted .inf
            }

    output_file = Path(cnf.PROJECT_ROOT / f"results/yaml-files/{file_name}.yaml")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(overrides_dict, f)

    print(f"Saved YAML to: {file_name}.yaml")


# %%
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import pandas as pd
from pathlib import Path


def distribute_ev_battery_caps(subregions, df_share, input_file):

    input_path = Path(cnf.PROJECT_ROOT / f"resources/yaml-files/{input_file}")
    output_path = Path(cnf.PROJECT_ROOT / f"results/yaml-files/{input_file}")

    yaml = YAML()
    yaml.indent(mapping=4, sequence=4, offset=2)

    # Load YAML
    with open(input_path, "r") as f:
        yaml_data = yaml.load(f)

    overrides = yaml_data["overrides"]["ev_battery_limits"]["locations"]
    new_locations = CommentedMap()

    for region in subregions:
        country = region.split("_")[0]
        country_key = f"{country}.techs"

        if country_key not in overrides:
            continue

        techs_dict = overrides[country_key]
        sub_dict = CommentedMap()

        # Get shares
        heavy_share = df_share[
            (df_share["locs"] == region)
            & (df_share["techs"] == "demand_heavy_transport")
        ]["value"]

        light_share = df_share[
            (df_share["locs"] == region)
            & (df_share["techs"] == "demand_light_transport")
        ]["value"]

        heavy_share = heavy_share.iloc[0] if not heavy_share.empty else 0
        light_share = light_share.iloc[0] if not light_share.empty else 0

        for tech, tech_data in techs_dict.items():
            cap = tech_data["constraints"]["storage_cap_max"]

            if tech in ["heavy_duty_ev_battery", "bus_ev_battery"]:
                share = heavy_share
            else:
                share = light_share

            sub_cap = float(cap * share)

            tech_entry = CommentedMap()
            constraints = CommentedMap()

            constraints["storage_cap_max"] = sub_cap

            # 👉 Add inline comment (unit)
            constraints.yaml_add_eol_comment("100'000 MWh", key="storage_cap_max")

            tech_entry["constraints"] = constraints
            sub_dict[tech] = tech_entry

        new_locations[f"{region}.techs"] = sub_dict

    yaml_data["overrides"]["ev_battery_limits"]["locations"] = new_locations

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save YAML
    with open(output_path, "w") as f:
        yaml.dump(yaml_data, f)

    print(f"New YAML saved to: {output_path}")


def build_ers_timeseries(
    ers_elec_demand_path,
    ers_usage_profile_path,
    subregions,
    truck_efficiency_kwh_per_km=1.8,  # default, adjust if needed
    ers_eff=0.9,
    demand_type="Stationary",  # or "Dynamic"
):
    """
    Returns:
        pd.DataFrame with hourly demand per subregion for year 2018
    """

    # -------------------------
    # 1. Read data
    # -------------------------
    df_demand = pd.read_csv(ers_elec_demand_path, sep=",")
    df_usage = pd.read_csv(ers_usage_profile_path, sep=",")

    # Parse timestamps
    df_demand["Timestamp"] = pd.to_datetime(
        df_demand["Timestamp"], format="%Y-%m-%d %H:%M:%S"
    )

    df_usage["Timestamp"] = pd.to_datetime(
        df_usage["Timestamp"], format="%Y-%m-%d %H:%M:%S"
    )

    df_demand = df_demand.set_index("Timestamp")
    df_usage = df_usage.set_index("Timestamp")

    # -------------------------
    # 2. Compute total demand per subregion
    # -------------------------
    total_demand = {}

    for region in subregions:
        col = f"{region}_{demand_type}_kWh"

        if col in df_demand.columns:
            total_kwh = df_demand[col].sum()
        else:
            total_kwh = 0

        # Convert to 100,000 MWh
        total_scaled = total_kwh / 1e8
        total_demand[region] = total_scaled

    total_demand = pd.Series(total_demand)

    # -------------------------
    # 3. Normalize usage profiles
    # -------------------------
    usage_norm = {}

    for region in subregions:
        col = f"{region}_COUNT"

        if col in df_usage.columns:
            profile = df_usage[col].copy()
        else:
            # If missing → zero profile
            profile = pd.Series(0, index=df_usage.index)

        total = profile.sum()

        if total > 0:
            usage_norm[region] = profile / total
        else:
            usage_norm[region] = profile  # stays zero

    usage_norm = pd.DataFrame(usage_norm)

    # -------------------------
    # 4. Redistribute demand
    # -------------------------
    # Convert energy → activity using efficiency
    # (interpretation: total energy / efficiency gives "km-equivalent")
    activity = total_demand * ers_eff / truck_efficiency_kwh_per_km  # 100 Mio vkm

    # Redistribute across time
    ts = usage_norm.multiply(activity, axis=1)

    # -------------------------
    # 5. Reindex to full 2018
    # -------------------------
    full_index = pd.date_range(
        start="2018-01-01 00:00:00", end="2018-12-31 23:00:00", freq="h"
    )

    # Map original profile to 2018 by repeating pattern
    ts = ts.copy()
    ts["hour"] = ts.index.hour
    ts["dayofyear"] = ts.index.dayofyear

    template = ts.groupby(["dayofyear", "hour"]).mean()

    new_index = pd.DataFrame(
        {"dayofyear": full_index.dayofyear, "hour": full_index.hour}, index=full_index
    )

    ts_2018 = new_index.join(template, on=["dayofyear", "hour"])
    ts_2018 = ts_2018[subregions]

    # Fill any gaps
    ts_2018 = ts_2018.fillna(0)

    return ts_2018


def split_harmonise_file(ers_elec_demand_path, ers_scenario):
    """Split dynamics fro stationary."""
    df_in = pd.read_csv(ers_elec_demand_path, sep=",")

    # 1. Keep only columns containing "Dynamic"
    for charge_type in ["Dynamic", "Stationary"]:
        df = df_in.loc[:, df_in.columns.str.contains(charge_type)].copy()

        # 2. Create new timestamp
        start_time = pd.Timestamp("2018-01-01 00:00:00")
        freq = "h"  # adjust if needed

        new_time = pd.date_range(start=start_time, periods=len(df), freq=freq)

        # Optional trimming (example: one year hourly)
        max_periods = 8760
        if len(df) > max_periods:
            df = df.iloc[:max_periods]
            new_time = new_time[:max_periods]

        # 3. Insert timestamp as first column
        df.insert(0, "timestep", new_time)

        # 4. Truncate column names after second "_"
        def truncate_after_second_underscore(col):
            if col == "timestep":
                return col
            parts = col.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:2])
            return col

        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols] / 100_000_000  # from kWh to 100 GWh

        df.columns = [truncate_after_second_underscore(c) for c in df.columns]

        output_dir = cnf.PROJECT_ROOT / "results/timeseries"
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            os.path.join(
                output_dir,
                f"heavy-duty-{charge_type.lower()}-charging-{ers_scenario}.csv",
            ),
            index=False,
            float_format="%.8f",
        )


def generate_ers_timevarying_constraints(ers_elec_demand_path, ers_scenario):
    """Split dynamics from stationary."""
    df_in = pd.read_csv(ers_elec_demand_path, sep=",")

    for charge_type in ["Dynamic", "Stationary"]:
        df = df_in.loc[:, df_in.columns.str.contains(charge_type)].copy()

        # 2. Create new timestamp
        start_time = pd.Timestamp("2018-01-01 00:00:00")
        freq = "h"
        new_time = pd.date_range(start=start_time, periods=len(df), freq=freq)

        # 3. Insert timestamp (no pre-trimming — keep extra rows for wrap-around)
        df.insert(0, "timestep", new_time)

        # 4. Rename columns
        def truncate_after_second_underscore(col):
            if col == "timestep":
                return col
            parts = col.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:2])
            return col

        df.columns = [truncate_after_second_underscore(c) for c in df.columns]

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include="number").columns

        # -----------------------------
        # STEP 0: Correction of time zone
        # -----------------------------
        df = apply_timezone_correction(df, wrap_around=True)

        # -----------------------------
        # STEP 0b: Trim to target year AFTER timezone correction
        # -----------------------------
        year_start = pd.Timestamp("2018-01-01 00:00:00")
        year_end = pd.Timestamp("2018-12-31 23:00:00")

        df["timestep"] = pd.to_datetime(df["timestep"])
        df = df.loc[
            (df["timestep"] >= year_start) & (df["timestep"] <= year_end)
        ].copy()
        df = df.reset_index(drop=True)

        # Re-identify numeric cols after correction (columns may have been reordered)
        numeric_cols = df.select_dtypes(include="number").columns

        # -----------------------------
        # STEP 1: Compute max per column (before normalization)
        # -----------------------------
        max_values = df[numeric_cols].max()

        # Convert units: kWh → 100,000 MW
        max_values_scaled = max_values / 100_000_000

        df_max = max_values_scaled.reset_index()
        df_max["unit"] = "0.1 TW"
        df_max.columns = ["locs", "max_value", "unit"]

        # -----------------------------
        # STEP 2: Normalize profiles
        # -----------------------------
        df[numeric_cols] = df[numeric_cols].div(max_values)

        # -----------------------------
        # Save outputs
        # -----------------------------
        output_dir = cnf.PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            os.path.join(
                output_dir,
                "timeseries",
                f"heavy-duty-{charge_type.lower()}-charging-normalised-{ers_scenario}.csv",
            ),
            index=False,
            float_format="%.8f",
        )

        df_max.to_csv(
            os.path.join(
                output_dir,
                f"heavy-duty-{charge_type.lower()}-charging-max-{ers_scenario}.csv",
            ),
            index=False,
            float_format="%.8f",
        )


def create_ers_cap_max_yaml(ers_scenario_list, charge_type):
    output_dir = cnf.PROJECT_ROOT / "results"

    # -----------------------------
    # Output path
    # -----------------------------
    yaml_dir = cnf.PROJECT_ROOT / "results/yaml-files"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    output_path = yaml_dir / f"ers_cap_max_{charge_type.lower()}.yaml"

    # -----------------------------
    # Write YAML manually (full control)
    # -----------------------------
    with open(output_path, "w") as f:

        # --- ers_exists ---
        f.write("overrides:\n")

        for ers_scenario in ers_scenario_list:
            df_max = pd.read_csv(
                os.path.join(
                    output_dir,
                    f"heavy-duty-{charge_type.lower()}-charging-max-{ers_scenario}.csv",
                )
            )

            # -----------------------------
            # Prepare data
            # -----------------------------
            locs = sorted(df_max["locs"].tolist())
            locs_str = ", ".join(locs)

            f.write(f"    ers_exists_{ers_scenario}:\n")
            f.write("        locations:\n")
            f.write(f"            {locs_str}:\n")
            f.write("                techs:\n")
            f.write("                    heavy_duty_ers_battery:\n")
            f.write("                    heavy_duty_transport_ers:\n")
            f.write("                    electric_road_system:\n\n")

            # --- ers_max_cap ---
            f.write(f"    ers_max_cap_{ers_scenario}:\n")
            f.write("        techs:\n")
            f.write("            electric_road_system:\n")
            f.write("                constraints:\n")
            f.write(
                f"                    energy_cap_time_varying: file=heavy-duty-dynamic-charging-normalised-{ers_scenario}.csv\n\n"
            )
            f.write("        locations:\n")

            for _, row in df_max.iterrows():
                loc = row["locs"]
                value = round(float(row["max_value"]), 8)

                f.write(f"            {loc}.techs:\n")
                f.write("                electric_road_system:\n")
                f.write("                    constraints:\n")
                f.write(f"                        energy_cap_max: {value}\n")


def normalise_profiles(df):
    # -----------------------------
    # 1. Identify time column
    # -----------------------------
    time_col = None
    for col in ["timestep", "timesteps", "datetime"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        raise ValueError("No 'timestep' or 'timesteps' column found.")

    # -----------------------------
    # 2. Separate numeric data
    # -----------------------------
    numeric_cols = df.columns.drop(time_col)

    # Ensure numeric (just in case)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # -----------------------------
    # 3. Compute max per column
    # -----------------------------
    max_vals = df[numeric_cols].abs().max()

    # Avoid division by zero
    max_vals = max_vals.replace(0, np.nan)

    # -----------------------------
    # 4. Normalize
    # -----------------------------
    df_norm = df.copy()
    df_norm[numeric_cols] = df[numeric_cols].abs() / max_vals

    return df_norm


def plot_profiles_window(
    dfs_dict,
    time_window,
    agg="mean",  # "mean", "sum", or None
    regions=None,  # list like ["DEU_1", "FRA_1"]
):
    """
    dfs_dict: dict of name -> dataframe
        e.g. {"demand": df1, "charging": df2, ...}
    time_window: ["start", "end"]
    """

    start, end = pd.to_datetime(time_window)

    plt.figure(figsize=(10, 5))

    for name, df in dfs_dict.items():

        # -----------------------------
        # Identify time column
        # -----------------------------
        for col in ["timestep", "timesteps", "datetime"]:
            if col in df.columns:
                time_col = col
                break
        else:
            raise ValueError("No valid time column found (timestep/timesteps/datetime)")

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # -----------------------------
        # Filter time window
        # -----------------------------
        df = df[(df[time_col] >= start) & (df[time_col] <= end)]

        # -----------------------------
        # Select data
        # -----------------------------
        data_cols = df.columns.drop(time_col)

        if regions is not None:
            data_cols = [c for c in data_cols if c in regions]

        if agg == "mean":
            y = df[data_cols].mean(axis=1)
        elif agg == "sum":
            y = df[data_cols].sum(axis=1)
        else:
            # plot first region if no aggregation
            y = df[data_cols[0]]

        # -----------------------------
        # Plot
        # -----------------------------
        plt.plot(df[time_col], y, label=name)

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Normalized value")
    plt.title(f"Profiles ({start} to {end})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def expand_vehicle_cost(df, final_year=2050, step=5, fit_start_year=None, target=None):
    """
    Expand vehicle cost data by group, projecting costs to a final year using polynomial fitting
    and optionally enforcing cost ceilings based on a target dict.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns 'year', 'value', 'category', 'vehicle_type', 'cost_class'.
    final_year : int, optional
        The final year to project to (default is 2050).
    step : int, optional
        Step size in years (default is 5).
    fit_start_year : int or None, optional
        Year from which to start fitting (default is None).
    target : dict or None, optional
        Dictionary mapping cost_class to target minimum values (e.g., {"inv": 0.0, "om_var": None}).

    Returns:
    -------
    pd.DataFrame
        Expanded DataFrame with projected vehicle costs and relative changes.
    """  # noqa: D205
    df = df.copy()
    df["year"] = df["year"].astype(int)
    results = []

    group_cols = ["category", "vehicle_type", "cost_class"]
    grouped = df.groupby(group_cols)

    for (cat, vtype, cost_class), group in grouped:
        group_sorted = group.sort_values("year").reset_index(drop=True)
        years = group_sorted["year"].values
        costs = group_sorted["value"].values

        # Compute relative change (€/year)
        rel_changes = [np.nan]
        for i in range(1, len(years)):
            dy = years[i] - years[i - 1]
            dcost = costs[i] - costs[i - 1]
            rel_changes.append(dcost / dy)
        group_sorted["relative_change"] = rel_changes

        # Add final year row if missing
        if final_year not in group_sorted["year"].values:
            group_sorted = pd.concat(
                [
                    group_sorted,
                    pd.DataFrame(
                        {
                            "year": [final_year],
                            "value": [np.nan],
                            "category": [cat],
                            "vehicle_type": [vtype],
                            "cost_class": [cost_class],
                            "relative_change": [0.0],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Prepare data for fitting
        rel_df = group_sorted.dropna(subset=["relative_change"])
        if fit_start_year:
            rel_df = rel_df[rel_df["year"] >= fit_start_year]

        coeffs = np.polyfit(rel_df["year"], rel_df["relative_change"], deg=2)
        poly = np.poly1d(coeffs)

        # Expand to future years
        existing_years = set(group_sorted["year"])
        last_data_year = max(y for y in years if y < final_year)

        expansion_years = list(range(last_data_year + step, final_year + 1, step))
        current_year = last_data_year
        current_cost = group_sorted[group_sorted["year"] == current_year][
            "value"
        ].to_numpy()[0]
        new_rows = []

        # Retrieve cost floor from target dict (if any)
        cost_floor = None
        if target and cost_class in target:
            cost_floor = target[cost_class]
            if cost_floor is None:
                cost_floor = -np.inf  # no constraint

        for year in expansion_years:
            dy = year - current_year
            rel_change = poly(year)
            projected_cost = current_cost + rel_change * dy

            # Prevent cost increase if target is specified
            if cost_floor is not None and projected_cost > current_cost:
                projected_cost = current_cost
                rel_change = 0.0

            new_rows.append(
                {
                    "year": year,
                    "value": projected_cost,
                    "category": cat,
                    "vehicle_type": vtype,
                    "cost_class": cost_class,
                    "relative_change": rel_change,
                }
            )

            current_year = year
            current_cost = projected_cost

        combined = pd.concat(
            [
                group_sorted[group_sorted["year"] <= last_data_year],
                pd.DataFrame(new_rows),
            ],
            ignore_index=True,
        )
        results.append(combined)

    return pd.concat(results, ignore_index=True)


def write_yaml_transport_costs(data, year):
    """Write the transport data to a YAML file."""
    cost_df = data[data["year"] == str(year)]
    override_name = f"transport_tech_cost_{year}"

    yaml_data = CommentedMap()
    yaml_data["overrides"] = CommentedMap()
    yaml_data["overrides"][override_name] = CommentedMap()
    yaml_data["overrides"][override_name]["locations"] = CommentedMap()

    for _, row in cost_df.iterrows():
        country = row["country"]
        demand_tech = row["demand_tech"]
        inv = round(row["inv"], 4)
        om_fix = round(row["om_fix"], 4)
        om_var = round(row["om_var"], 4)

        loc_key = f"{country}.techs"
        if loc_key not in yaml_data["overrides"][override_name]["locations"]:
            yaml_data["overrides"][override_name]["locations"][loc_key] = CommentedMap()

        tech_block = CommentedMap()
        cost_block = CommentedMap()
        monetary = CommentedMap()

        monetary["energy_cap"] = inv
        monetary.yaml_add_eol_comment("(10 EUR2015/km_per_hour)", "energy_cap")
        # Conditionally add om_fix
        if om_fix != 0.0:
            monetary["om_annual"] = om_fix
            monetary.yaml_add_eol_comment("(10 EUR2015/km)", "om_annual")

        # Conditionally add om_var
        if om_var != 0.0:
            monetary["om_prod"] = om_var
            monetary.yaml_add_eol_comment("(10 EUR2015/km)", "om_prod")

        cost_block["monetary"] = monetary
        tech_block["costs"] = cost_block
        yaml_data["overrides"][override_name]["locations"][loc_key][
            demand_tech
        ] = tech_block

    # Write YAML with 4-space indent and preserved comments
    yaml = YAML()
    yaml.indent(mapping=4, sequence=4, offset=2)
    with open(
        os.path.join(
            cnf.PROJECT_ROOT,
            "results",
            "yaml-files",
            f"transport-tech-costs-{str(year)}.yaml",
        ),
        "w",
    ) as f:
        yaml.dump(yaml_data, f)


def battery_fleet(vehicle_number, dict_battery_size, loc_list=None):
    """Computes the total fleet battery per vehicle type."""
    num_vehicle = vehicle_number.copy()
    num_vehicle["battery_cap"] = num_vehicle["techs"].map(dict_battery_size)

    # Calculate the total fleet battery size for each row
    num_vehicle["fleet_battery_cap"] = (
        num_vehicle["vehicles"] * num_vehicle["battery_cap"] / 100000000
    )  # 100'000 MWh

    # Filter locs
    if loc_list is not None:
        num_vehicle = num_vehicle[num_vehicle["locs"].isin(loc_list)]

    return num_vehicle[["locs", "techs", "fleet_battery_cap"]]


def build_and_save_ers_yaml(df_lengths, ers_inv_cost, ers_fix_cost, output_path):
    """
    Build ERS cost YAML and save it with controlled indentation.

    Parameters
    ----------
    df_lengths : pd.DataFrame
        columns: ['locs', 'length_km']
    ers_inv_cost : float
        €/km/kW
    ers_fix_cost : float
        €/km/kW
    output_path : str or Path
        YAML file path
    """

    df = df_lengths.copy()

    # conversion factor:
    # €/km/kW * km -> €/kW
    # €/kW -> €/MW (*1000)
    # EUR -> 10,000 EUR (/10000)
    factor = 0.1  # (1000 / 10000)

    # compute costs per location
    df["energy_cap"] = ers_inv_cost * df["length_km"] * factor
    df["om_annual"] = ers_fix_cost * df["length_km"] * factor

    yaml_dict = {"overrides": {"ers_costs": {"locations": {}}}}

    for _, row in df.iterrows():
        loc = row["locs"]

        yaml_dict["overrides"]["ers_costs"]["locations"][loc] = {
            "techs": {
                "electric_road_system": {
                    "costs": {
                        "monetary": {
                            "energy_cap": float(row["energy_cap"]),
                            "om_annual": float(row["om_annual"]),
                        }
                    }
                }
            }
        }

    # write YAML with proper indentation
    yaml = YAML()
    yaml.indent(mapping=4, sequence=4, offset=2)
    yaml.default_flow_style = False

    with open(output_path, "w") as f:
        yaml.dump(yaml_dict, f)

    # return yaml_dict


def generate_yaml_ev_battery_cap_max(df, output_file_path):
    """Writes the battery limits in the yaml for each vehicle and country."""
    # Dictionary to map Transport_Type to the desired YAML keys
    transport_type_mapping = {
        "light_duty_transport": "light_duty_ev_battery",
        "heavy_duty_transport": "heavy_duty_ev_battery",
        "passenger_car_transport": "passenger_car_ev_battery",
        "motorcycle_transport": "motorcycle_ev_battery",
        "bus_transport": "bus_ev_battery",
    }

    # Initialize the dictionary to structure the YAML data
    yaml_data = {"overrides": {"ev_battery_limits": {"locations": {}}}}

    # Iterate over the unique countries
    for country in df["locs"].unique():
        # Filter the DataFrame for the current country
        country_data = df[df["locs"] == country]

        # Create a dictionary for the country
        country_dict = {}

        for _, row in country_data.iterrows():
            tech_key = transport_type_mapping[row["techs"]]
            storage_cap_max = row["fleet_battery_cap"]  # Convert kWh to MWh

            # Populate the dictionary with the relevant data
            if country not in country_dict:
                country_dict[country] = {"techs": {}}
            country_dict[country]["techs"][tech_key] = {
                "constraints": {
                    "storage_cap_max": storage_cap_max,
                }
            }

        # Update the main YAML data structure
        yaml_data["overrides"]["ev_battery_limits"]["locations"].update(country_dict)

    # Write the dictionary to a YAML file with increased indentation and comments
    with open(output_file_path, "w") as file:
        file.write("overrides:\n")
        file.write("    ev_battery_limits:\n")
        file.write("        locations:\n")

        for country, data in yaml_data["overrides"]["ev_battery_limits"][
            "locations"
        ].items():
            file.write(f"            {country}.techs:\n")
            for tech, tech_data in data["techs"].items():
                file.write(f"                {tech}:\n")
                file.write("                    constraints:\n")
                storage_cap_max = tech_data["constraints"]["storage_cap_max"]
                comment = " # 100'000 MWh"
                file.write(
                    f"                        storage_cap_max: {storage_cap_max}{comment}\n"
                )


def generate_yaml_ers_battery_cap_max(df, output_file_path):
    """Writes the battery limits in the yaml for each vehicle and country."""
    # Dictionary to map Transport_Type to the desired YAML keys
    transport_type_mapping = {
        "heavy_duty_transport": "heavy_duty_ers_battery",
    }

    # Initialize the dictionary to structure the YAML data
    yaml_data = {"overrides": {"ers_battery_limits": {"locations": {}}}}

    # Iterate over the unique countries
    for country in df["locs"].unique():
        # Filter the DataFrame for the current country
        country_data = df[df["locs"] == country]

        # Create a dictionary for the country
        country_dict = {}

        for _, row in country_data.iterrows():
            tech_key = transport_type_mapping[row["techs"]]
            storage_cap_max = row["fleet_battery_cap"]  # Convert kWh to MWh

            # Populate the dictionary with the relevant data
            if country not in country_dict:
                country_dict[country] = {"techs": {}}
            country_dict[country]["techs"][tech_key] = {
                "constraints": {
                    "storage_cap_max": storage_cap_max,
                }
            }

        # Update the main YAML data structure
        yaml_data["overrides"]["ers_battery_limits"]["locations"].update(country_dict)

    # Write the dictionary to a YAML file with increased indentation and comments
    with open(output_file_path, "w") as file:
        file.write("overrides:\n")
        file.write("    ers_battery_limits:\n")
        file.write("        locations:\n")

        for country, data in yaml_data["overrides"]["ers_battery_limits"][
            "locations"
        ].items():
            file.write(f"            {country}.techs:\n")
            for tech, tech_data in data["techs"].items():
                file.write(f"                {tech}:\n")
                file.write("                    constraints:\n")
                storage_cap_max = tech_data["constraints"]["storage_cap_max"]
                comment = " # 100'000 MWh"
                file.write(
                    f"                        storage_cap_max: {storage_cap_max}{comment}\n"
                )


def split_vehicle_numbers_by_share(vehicle_num, df_share, tolerance=1e-6):
    """
    Split national vehicle numbers across subregions using predefined shares.

    Parameters
    ----------
    vehicle_num : pd.DataFrame
        Columns:
        country_code,
        light_duty_transport,
        heavy_duty_transport,
        passenger_car_transport,
        motorcycle_transport,
        bus_transport

    df_share : pd.DataFrame
        Columns:
        locs, techs, unit, value

        techs must contain:
        demand_heavy_transport
        demand_light_transport

    tolerance : float
        Allowed deviation from share sum = 1

    Returns
    -------
    pd.DataFrame
        Columns:
        locs, techs, vehicles
    """

    share_map = {
        "bus_transport": "demand_heavy_transport",
        "heavy_duty_transport": "demand_heavy_transport",
        "light_duty_transport": "demand_light_transport",
        "passenger_car_transport": "demand_light_transport",
        "motorcycle_transport": "demand_light_transport",
    }

    shares = df_share.copy()
    vehicles = vehicle_num.copy()

    # derive country code from locs
    shares["country_code"] = shares["locs"].str.split("_").str[0]

    # validate shares sum to 1 by country + share type
    check = shares.groupby(["country_code", "techs"])["value"].sum().reset_index()

    bad = check[np.abs(check["value"] - 1) > tolerance]

    if not bad.empty:
        raise ValueError(
            "Share totals not equal to 1 within tolerance:\n"
            + bad.to_string(index=False)
        )

    output = []

    for _, row in vehicles.iterrows():

        country = row["country_code"]

        for vehicle_type, share_type in share_map.items():

            national_total = int(row[vehicle_type])

            subset = shares[
                (shares["country_code"] == country) & (shares["techs"] == share_type)
            ].copy()

            if subset.empty:
                raise ValueError(
                    f"Missing shares for country={country}, share={share_type}"
                )

            # raw allocation
            subset["vehicles_raw"] = subset["value"] * national_total

            # floor
            subset["vehicles"] = np.floor(subset["vehicles_raw"]).astype(int)

            # distribute remainder
            remainder = national_total - subset["vehicles"].sum()

            if remainder > 0:
                subset["fraction"] = subset["vehicles_raw"] - subset["vehicles"]

                subset = subset.sort_values("fraction", ascending=False)

                idx = subset.index[:remainder]
                subset.loc[idx, "vehicles"] += 1

            subset = subset.sort_values("locs")

            result_part = subset[["locs"]].copy()
            result_part["techs"] = vehicle_type
            result_part["vehicles"] = subset["vehicles"].values

            output.append(result_part)

    result = pd.concat(output, ignore_index=True)

    return result


def iso3_to_tz(iso3):
    """
    Direct ISO3 → timezone resolver (no ISO2 step exposed)
    """
    country = pycountry.countries.get(alpha_3=iso3)
    if country is None:
        raise ValueError(f"Unknown ISO3 country: {iso3}")
    return pytz.country_timezones[country.alpha_2][0]


def normalise_profiles_tz(df, apply_timezone=False):
    """
    Normalize profiles with optional ISO3-based timezone correction.
    Output DataFrame has the same structure as input (same columns, same timestep format).
    """

    # -----------------------------
    # 1. Detect time column
    # -----------------------------
    time_col = None
    for col in ["timestep", "timesteps", "datetime"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        raise ValueError("No time column found.")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Preserve original column order for restoration later
    value_cols = [c for c in df.columns if c != time_col]

    # -----------------------------
    # 2. OPTIONAL timezone correction (ISO3-aware)
    # -----------------------------
    if apply_timezone:
        long = df.melt(id_vars=time_col, var_name="locs", value_name="value")
        long["country"] = long["locs"].str.split("_").str[0]

        tz_map = {c: iso3_to_tz(c) for c in long["country"].unique()}

        # Build (iso3, naive_ts) -> utc_ts lookup, one tz at a time
        unique_times = df[time_col].drop_duplicates().sort_values()
        utc_map = {}
        for iso3, tz in tz_map.items():
            try:
                localized = unique_times.dt.tz_localize(
                    tz, ambiguous="infer", nonexistent="shift_forward"
                )
            except Exception:
                localized = unique_times.dt.tz_localize(
                    tz,
                    ambiguous=np.zeros(len(unique_times), dtype=bool),
                    nonexistent="shift_forward",
                )
            utc_times = localized.dt.tz_convert("UTC").dt.tz_localize(None)
            for naive, utc in zip(unique_times, utc_times):
                utc_map[(iso3, naive)] = utc

        # Vectorised map (no iterrows)
        long[time_col] = pd.Series(
            [utc_map[(c, t)] for c, t in zip(long["country"], long[time_col])],
            index=long.index,
        )

        # DST spring-forward causes two local hours to collapse into the same UTC
        # slot — average them so pivot has no duplicate index entries
        long = long.groupby([time_col, "locs"], as_index=False)["value"].mean()

        df = long.pivot(index=time_col, columns="locs", values="value").reset_index()
        df.columns.name = None
        # -----------------------------
        # 2b. PATCH: wrap trimmed time series back into NaN holes
        # -----------------------------

        def wrap_series(s):
            s = s.copy()

            # detect NaNs created by shift
            nan_mask = s.isna()

            if not nan_mask.any():
                return s

            valid = s.dropna()
            if valid.empty:
                return s

            # split head/tail
            n_missing = nan_mask.sum()

            # circular fill assumption (energy-preserving shift)
            wrapped_values = pd.concat(
                [valid.iloc[-n_missing:], valid, valid.iloc[:n_missing]]
            )

            # align back to original index
            s.loc[nan_mask] = wrapped_values.iloc[:n_missing].values

            return s

        value_cols = [c for c in df.columns if c != time_col]
        numeric_cols = value_cols.copy()

        for col in numeric_cols:
            df[col] = wrap_series(df[col])
    # -----------------------------
    # 3. Normalize
    # -----------------------------
    numeric_cols = [c for c in df.columns if c != time_col]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    max_vals = df[numeric_cols].abs().max().replace(0, np.nan)

    df_norm = df.copy()
    df_norm[numeric_cols] = df[numeric_cols].abs() / max_vals

    # -----------------------------
    # 4. Restore original structure
    # -----------------------------
    df_norm = df_norm[[time_col] + value_cols]

    if df_norm[time_col].dt.tz is not None:
        df_norm[time_col] = df_norm[time_col].dt.tz_localize(None)
    df_norm[time_col] = df_norm[time_col].dt.floor("s").astype(str).str[:19]

    return df_norm


def apply_timezone_correction(df, wrap_around=True):
    """
    Shift each country's profile by its UTC offset (in hours) using a circular
    shift (np.roll), keeping the timestep index clean and continuous.
    No gaps or duplicates are created.
    """
    import numpy as np

    time_col = next(
        (c for c in ["timestep", "timesteps", "datetime"] if c in df.columns), None
    )
    if time_col is None:
        raise ValueError("No time column found.")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    value_cols = [c for c in df.columns if c != time_col]

    for col in value_cols:
        iso3 = col.split("_")[0].replace("\xa0", "").strip()
        tz_str = iso3_to_tz(iso3)

        # Use a fixed winter date to get a clean integer UTC offset (avoids DST)
        ref = pd.Timestamp("2018-01-15").tz_localize(tz_str)
        offset_hours = int(ref.utcoffset().total_seconds() / 3600)

        # Circular shift: wraps bottom values to top, no gaps, no duplicates
        df[col] = np.roll(df[col].to_numpy(), offset_hours)

    return df


def detect_nbsp_full(df):
    results = []

    def check_value(val, context):
        if isinstance(val, str) and "\xa0" in val:
            results.append(
                {
                    "context": context,
                    "full_value": repr(val),
                    "cleaned_value": val.replace("\xa0", "").strip(),
                }
            )

    # Check columns
    for col in df.columns:
        check_value(col, "column_name")

    # Check index
    for idx in df.index:
        check_value(idx, "index")

    # Check data
    for col in df.columns:
        if df[col].dtype == object:
            for val in df[col].dropna().unique():
                check_value(val, f"value_in_{col}")

    return results


def plot_heavy_transport_mix_map(gdf, scenario=None, max_radius=1.2):
    """
    Plot proportional pie charts of heavy-duty transport technology mix per subregion.
    Pie radius encodes total flow (billion_km); wedge angles encode technology mix.
    Reads model_results/flow_out_sum.csv, filters carrier=='transport' and techs
    containing 'heavy_duty' (excluding 'battery').
    Saves figure to results/figures/.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge

    # --- 1. Load and filter data ---
    df = pd.read_csv(cnf.PROJECT_ROOT / "model_results/flow_out_sum.csv")
    df = df[
        (df["carriers"] == "transport")
        & (df["techs"].str.contains("heavy_duty"))
        & (~df["techs"].str.contains("battery"))
    ]
    if scenario is not None:
        df = df[df["scenario"] == scenario]

    # --- 2. Prepare base map ---
    gdf = gdf.copy()

    def clean_id(shape_id):
        return "_".join(shape_id.split("_")[:2])

    gdf["node"] = gdf["shape_id"].apply(clean_id)
    land_gdf = gdf[gdf["shape_class"] == "land"]
    maritime_gdf = gdf[gdf["shape_class"] == "maritime"]

    # --- 3. Tech color palette ---
    tech_colors = {
        "heavy_duty_transport_ev": "#2196F3",  # blue
        "heavy_duty_transport_fcev": "#4CAF50",  # green
        "heavy_duty_transport_ice": "#FF5722",  # orange-red
        "heavy_duty_transport_ers": "#9C27B0",  # purple
    }
    fallback = ["#FFC107", "#00BCD4", "#795548", "#607D8B"]
    for i, t in enumerate(sorted(df["techs"].unique())):
        if t not in tech_colors:
            tech_colors[t] = fallback[i % len(fallback)]

    # --- 4. Compute per-region totals and scale radii (area ∝ total flow) ---
    totals = df.groupby("locs")["flow_out_sum"].sum()
    max_total = totals.max()

    def scaled_radius(total):
        return max_radius * np.sqrt(total / max_total)

    # --- 5. Centroids indexed by node ---
    nodes = land_gdf.set_index("node").geometry.centroid

    # --- 6. Base map ---
    fig, ax = plt.subplots(figsize=(10, 10))

    if not maritime_gdf.empty:
        maritime_gdf.plot(ax=ax, color="lightblue", edgecolor="none")
    land_gdf.plot(ax=ax, color="#F4E2B8", edgecolor="#927961", linewidth=0.5)

    # Equal aspect ratio ensures Wedge patches appear as circles, then lock limits
    ax.set_aspect("equal")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    # --- 7. Proportional pie charts via Wedge patches ---
    grouped = df.groupby(["locs", "techs"])["flow_out_sum"].sum().reset_index()

    for loc, group in grouped.groupby("locs"):
        if loc not in nodes.index:
            continue

        centroid = nodes[loc]
        if isinstance(centroid, gpd.GeoSeries):
            centroid = centroid.iloc[0]
        cx, cy = centroid.x, centroid.y

        sizes = group["flow_out_sum"].values
        total = sizes.sum()
        if total == 0:
            continue

        r = scaled_radius(total)
        colors = [tech_colors.get(t, "grey") for t in group["techs"]]
        start = 90.0  # start from 12 o'clock

        for frac, color in zip(sizes / total, colors):
            sweep = frac * 360.0
            wedge = Wedge(
                center=(cx, cy),
                r=r,
                theta1=start - sweep,
                theta2=start,
                facecolor=color,
                linewidth=0.3,
                edgecolor="white",
            )
            ax.add_patch(wedge)
            start -= sweep

    # --- 8. Color legend (technology types) ---
    present_techs = sorted(df["techs"].unique())
    color_handles = [
        mpatches.Patch(color=tech_colors[t], label=t.replace("_", " "))
        for t in present_techs
    ]
    color_legend = ax.legend(
        handles=color_handles,
        title="Technology",
        fontsize=9,
        title_fontsize=10,
        loc="lower left",
        framealpha=0.9,
    )
    ax.add_artist(color_legend)

    # --- 9. Size legend (total flow reference circles) ---
    ref_fracs = [0.25, 0.5, 1.0]
    ref_vals = [max_total * f for f in ref_fracs]

    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="lightgrey",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=4 + 12 * np.sqrt(f),
            label=f"{v:.1f} Gkm",
        )
        for f, v in zip(ref_fracs, ref_vals)
    ]
    ax.legend(
        handles=size_handles,
        title="Total flow (billion km)",
        fontsize=9,
        title_fontsize=10,
        loc="upper right",
        framealpha=0.9,
    )

    ax.set_axis_off()
    plt.tight_layout()

    # --- 10. Save ---
    out_dir = cnf.PROJECT_ROOT / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = scenario if scenario else "all"
    out_path = out_dir / f"heavy_transport_mix_{tag}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return fig, ax


def plot_electricity_mix_map(
    gdf, yaml_path, scenario=None, max_radius=1.2, linewidth_scale=25
):
    """
    Plot proportional pie charts of electricity generation mix per subregion,
    with transmission connection lines from a links YAML.
    Pie radius encodes total electricity produced (TWh); wedge angles encode tech mix.
    Reads model_results/flow_out_sum.csv, filters carrier=='electricity'.
    Saves figure to results/figures/.
    """
    import yaml as _yaml
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge
    from shapely.geometry import LineString
    from collections import defaultdict

    # --- 1. Load and filter data ---
    df = pd.read_csv(cnf.PROJECT_ROOT / "model_results/flow_out_sum.csv")
    df = df[df["carriers"] == "electricity"]
    if scenario is not None:
        df = df[df["scenario"] == scenario]

    # --- 2. Prepare base map ---
    gdf = gdf.copy()

    def clean_id(shape_id):
        return "_".join(shape_id.split("_")[:2])

    gdf["node"] = gdf["shape_id"].apply(clean_id)
    land_gdf = gdf[gdf["shape_class"] == "land"]
    maritime_gdf = gdf[gdf["shape_class"] == "maritime"]

    # --- 3. Tech color palette ---
    tech_colors = {
        "wind_onshore": "#4CAF50",
        "wind_offshore": "#00897B",
        "open_field_pv": "#FDD835",
        "roof_mounted_pv": "#FF8F00",
        "hydro_reservoir": "#42A5F5",
        "hydro_run_of_river": "#1565C0",
        "pumped_hydro": "#5C6BC0",
        "nuclear": "#AB47BC",
        "ccgt": "#FF7043",
        "ccgt_ccs": "#BF360C",
        "coal_power_plant": "#546E7A",
        "chp_biofuel_extraction": "#8D6E63",
        "chp_biofuel_extraction_ccs": "#4E342E",
        "chp_biogas": "#A5D6A7",
        "chp_methane_extraction": "#FFB74D",
        "chp_methane_extraction_ccs": "#E65100",
        "chp_wte_back_pressure": "#78909C",
        "chp_wte_back_pressure_ccs": "#37474F",
        "biofuel_to_liquids": "#BCAAA4",
        "battery": "#B0BEC5",
    }
    fallback = ["#F06292", "#26C6DA", "#D4E157", "#FF8A65"]
    for i, t in enumerate(sorted(df["techs"].unique())):
        if t not in tech_colors:
            tech_colors[t] = fallback[i % len(fallback)]

    # --- 4. Per-region totals and scaled radii (area ∝ total production) ---
    totals = df.groupby("locs")["flow_out_sum"].sum()
    max_total = totals.max()

    def scaled_radius(total):
        return max_radius * np.sqrt(total / max_total)

    # --- 5. Centroids indexed by node ---
    nodes = land_gdf.set_index("node").geometry.centroid

    # --- 6. Transmission lines from YAML ---
    with open(yaml_path) as f:
        yaml_data = _yaml.safe_load(f)

    links = yaml_data.get("links", {})
    capacity = defaultdict(float)
    for key, value in links.items():
        locs_part = key.split(".")[0]
        loc1, loc2 = locs_part.split(",")
        pair = tuple(sorted([loc1, loc2]))
        for tech in value.values():
            cap = tech.get("constraints", {}).get("energy_cap_min", 0)
            capacity[pair] += cap

    line_records = []
    for (n1, n2), cap in capacity.items():
        if n1 in nodes.index and n2 in nodes.index:
            c1 = nodes[n1]
            c2 = nodes[n2]
            if isinstance(c1, gpd.GeoSeries):
                c1 = c1.iloc[0]
            if isinstance(c2, gpd.GeoSeries):
                c2 = c2.iloc[0]
            line_records.append({"geometry": LineString([c1, c2]), "cap": cap})

    lines_gdf = (
        gpd.GeoDataFrame(line_records, crs=gdf.crs)
        if line_records
        else gpd.GeoDataFrame()
    )

    # --- 7. Base map ---
    fig, ax = plt.subplots(figsize=(12, 12))

    if not maritime_gdf.empty:
        maritime_gdf.plot(ax=ax, color="lightblue", edgecolor="none")
    land_gdf.plot(ax=ax, color="#F4E2B8", edgecolor="#927961", linewidth=0.5)

    # Transmission lines drawn under pies
    if not lines_gdf.empty:
        lines_gdf.plot(
            ax=ax,
            linewidth=lines_gdf["cap"] * linewidth_scale,
            color="steelblue",
            alpha=0.6,
        )

    # Equal aspect so Wedge patches appear circular, then lock limits
    ax.set_aspect("equal")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    # --- 8. Proportional pie charts via Wedge patches ---
    grouped = df.groupby(["locs", "techs"])["flow_out_sum"].sum().reset_index()

    for loc, group in grouped.groupby("locs"):
        if loc not in nodes.index:
            continue

        centroid = nodes[loc]
        if isinstance(centroid, gpd.GeoSeries):
            centroid = centroid.iloc[0]
        cx, cy = centroid.x, centroid.y

        sizes = group["flow_out_sum"].values
        total = sizes.sum()
        if total == 0:
            continue

        r = scaled_radius(total)
        colors = [tech_colors.get(t, "grey") for t in group["techs"]]
        start = 90.0

        for frac, color in zip(sizes / total, colors):
            sweep = frac * 360.0
            wedge = Wedge(
                center=(cx, cy),
                r=r,
                theta1=start - sweep,
                theta2=start,
                facecolor=color,
                linewidth=0.2,
                edgecolor="white",
            )
            ax.add_patch(wedge)
            start -= sweep

    # --- 9. Color legend (two columns for 20 techs) ---
    present_techs = sorted(df["techs"].unique())
    color_handles = [
        mpatches.Patch(color=tech_colors[t], label=t.replace("_", " "))
        for t in present_techs
    ]
    color_legend = ax.legend(
        handles=color_handles,
        title="Technology",
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
        framealpha=0.9,
        ncol=2,
    )
    ax.add_artist(color_legend)

    # --- 10. Size legend ---
    ref_fracs = [0.25, 0.5, 1.0]
    ref_vals = [max_total * f for f in ref_fracs]
    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="lightgrey",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=4 + 12 * np.sqrt(f),
            label=f"{v:.2f} TWh",
        )
        for f, v in zip(ref_fracs, ref_vals)
    ]
    ax.legend(
        handles=size_handles,
        title="Total electricity (TWh)",
        fontsize=8,
        title_fontsize=9,
        loc="upper right",
        framealpha=0.9,
    )

    ax.set_axis_off()
    plt.tight_layout()

    # --- 11. Save ---
    out_dir = cnf.PROJECT_ROOT / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = scenario if scenario else "all"
    out_path = out_dir / f"electricity_mix_{tag}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return fig, ax


def plot_duals_electricity(scenario=None, n_cols=10):
    """
    Plot daily-averaged electricity dual values for each region in a grid of subplots.
    Each subplot shows:
      - background bands: p0-p10 / p10-p50 / p50-p90 / p90-p100 of that region's daily means
      - shaded grey envelope: intra-day p10-p90 range (across the 3h timesteps within each day)
      - solid line: daily mean
    Saves figure to results/figures/.
    """
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

    # --- 1. Load data ---
    df = pd.read_csv(
        cnf.PROJECT_ROOT / "model_results/duals_balance.csv",
        usecols=["scenario", "locs", "carriers", "timesteps", "dual_value"],
    )
    df = df[df["carriers"] == "electricity"].copy()

    if scenario is not None:
        df = df[df["scenario"] == scenario]

    df["timesteps"] = pd.to_datetime(df["timesteps"])
    df["date"] = df["timesteps"].dt.normalize()

    scenario_name = df["scenario"].iloc[0]

    # --- 2. Aggregate 3h -> daily mean / p10 / p90 ---
    grp = df.groupby(["locs", "date"])["dual_value"]
    daily = pd.concat(
        [
            grp.mean().rename("mean"),
            grp.quantile(0.10).rename("p10"),
            grp.quantile(0.90).rename("p90"),
        ],
        axis=1,
    ).reset_index()

    # --- 3. Layout ---
    regions = sorted(daily["locs"].unique())
    n_regions = len(regions)
    n_rows = int(np.ceil(n_regions / n_cols))

    # --- 4. Colors ---
    band_colors = ["#d1e5f0", "#92c5de", "#fdd0a2", "#f4a582"]
    band_labels = ["0–10th", "10–50th", "50–90th", "90–100th"]

    # --- 5. Figure ---
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.2, n_rows * 2.2),
        sharey=False,
        sharex=False,
    )
    axes_flat = axes.flatten()

    for i, loc in enumerate(regions):
        ax = axes_flat[i]

        loc_df = daily[daily["locs"] == loc].sort_values("date")
        dates = loc_df["date"].values
        mean_v = loc_df["mean"].values
        p10_v = loc_df["p10"].values
        p90_v = loc_df["p90"].values

        # Background bands: percentiles of this region's daily means
        dm_p10 = np.percentile(mean_v, 10)
        dm_p50 = np.percentile(mean_v, 50)
        dm_p90 = np.percentile(mean_v, 90)
        ymin, ymax = 0, 70

        for ylo, yhi, color in zip(
            [ymin, dm_p10, dm_p50, dm_p90],
            [dm_p10, dm_p50, dm_p90, ymax],
            band_colors,
        ):
            ax.axhspan(ylo, yhi, facecolor=color, alpha=0.7, linewidth=0)

        # Intra-day p10–p90 envelope
        ax.fill_between(dates, p10_v, p90_v, color="#4a4a4a", alpha=0.20, linewidth=0)

        # Daily mean line
        ax.plot(dates, mean_v, color="#1a1a1a", linewidth=0.7)

        ax.set_title(loc, fontsize=7, pad=2)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(ymin, ymax)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=4)
        ax.tick_params(axis="y", labelsize=5)
        ax.tick_params(axis="x", pad=1)

    # Hide unused cells
    for j in range(n_regions, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # --- 6. Legend ---
    band_handles = [
        mpatches.Patch(
            facecolor=c, edgecolor="none", alpha=0.7, label=f"{l} pct of daily mean"
        )
        for c, l in zip(band_colors, band_labels)
    ]
    envelope_handle = mpatches.Patch(
        facecolor="#4a4a4a",
        edgecolor="none",
        alpha=0.20,
        label="Intra-day p10–p90 range",
    )
    line_handle = Line2D([], [], color="#1a1a1a", linewidth=1.2, label="Daily mean")

    fig.legend(
        handles=band_handles + [envelope_handle, line_handle],
        title=f"Electricity dual value  |  scenario: {scenario_name}",
        loc="lower center",
        ncol=3,
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"Electricity dual values (daily avg)  —  {scenario_name}", fontsize=13, y=1.01
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # --- 7. Save ---
    out_dir = cnf.PROJECT_ROOT / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"duals_electricity_{scenario_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return fig, axes
