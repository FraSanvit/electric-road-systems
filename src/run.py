
#%%
from utils import *
import config as cnf
import sys
import os

from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

import yaml
import re
from collections import defaultdict

sys.path.append(os.path.abspath(".."))

from resources.data.transport_cost import transport_cost

# %% EMODNET

projection_year = 2030
lifetime = 25
shape = gpd.read_parquet(cnf.shape_path)


df = windfarms_json_to_df(cnf.windfarm_path)
df_agg = aggregate_wind_capacity(df, shape, projection_year, lifetime)

plot_wind_farms_emodnet(df, shape)



#%% windeurope

df_wind = load_wind_farms_from_har_folder(cnf.windeurope_path)
df_allocated = allocate_wind_to_shapes(df_wind, shape, projection_year, lifetime)

plot_wind_farms_windeurope(df_wind, shape)


#%% compaison
plot_two_dfs_side_by_side_hatched_vertical_national(df_agg, df_allocated, label1="emodnet", label2="windeurope")

#%% disaggregated vkm freight transport

yaml_file = "D:\\transfer\\ERS\\2030\\model\\eurospores\\vehicle_group_constraints_2018.yaml"

country_totals = defaultdict(lambda: {"heavy": 0.0, "light": 0.0})

with open(yaml_file, "r") as f:
    data = yaml.safe_load(f)

# Navigate to group_constraints
group_constraints = (
    data.get("overrides", {})
        .get("annual_transport_distance", {})
        .get("group_constraints", {})
)

# Regex pattern to match keys
pattern = re.compile(r"(heavy|light)_([A-Z]{3}_\d+)_annual_distance")

for key, value in group_constraints.items():
    match = pattern.match(key)
    if not match:
        continue

    transport_type, loc = match.groups()

    # Extract country code (ALB_1 -> ALB)
    country = loc.split("_")[0]

    # Extract value from carrier_con_equals
    carrier_dict = value.get("carrier_con_equals", {})
    
    if transport_type == "heavy":
        distance = carrier_dict.get("heavy_transport", 0.0)
    else:
        distance = carrier_dict.get("light_transport", 0.0)

    country_totals[country][transport_type] += distance

# ---- Print results ----
print("Country totals (100 Mio km):\n")
for country in sorted(country_totals):
    heavy = country_totals[country]["heavy"]
    light = country_totals[country]["light"]
    total = heavy + light

    print(f"{country}:")
    print(f"   Heavy: {heavy:.3f}")
    print(f"   Light: {light:.3f}")
    print(f"   Total: {total:.3f}")
    print()


# %% ERS temporary

input_path = "C:\\Users\\sanvi\\OneDrive - Delft University of Technology\\Research Projects\\ERS\\ERS inputs"
file_name = "250KW_HGV_DYN_STAT_FULL.csv"

def generate_ers_constraint(input_path, file_name):
    def _split_harmonise_file(input_path, file_name):
        """Split dynamics fro stationary."""
        df_in = pd.read_csv(os.path.join(input_path, file_name), index_col=False)

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
            df.to_csv(
                os.path.join(input_path, f"heavy-duty-{charge_type.lower()}-charging.csv"), index=False
            )

            # Time-varying constraints

            df_normalized = df.copy()
            cols = df_normalized.columns.drop("timestep")

            df_normalized[cols] = df[cols] / df[cols].max()
            df_normalized.to_csv(
                os.path.join(input_path, f"ers-charging-{charge_type.lower()}-rate.csv"), index=False
            )

            return df
        
    def compute_max_cap(df):
        df_max = (
            df.drop(columns="timestep")
            .max()
            .rename("max_cap_ers_100gw")
            .reset_index()
            .rename(columns={"index": "locs"})
        )
        return df_max
    
    def save_ers_yaml(df_max, folder_path, filename="ers_overrides.yaml"):
        """
        Create a YAML file for ers_max_cap and ers_charging_fixed overrides.
        
        Parameters:
        - df_max: DataFrame with columns ['locs', 'max_cap_ers_100gw']
        - folder_path: str or Path where the YAML will be saved
        - filename: name of the YAML file (default: 'ers_overrides.yaml')
        """
        
        # Ensure folder exists
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        
        # Base structure
        overrides = {
            "overrides": {
                "ers_charging_fixed": {
                    "techs": {
                        "electric_road_system": {
                            "constraints": {
                                "energy_cap_equals_time_varying": "file=heavy-duty-dynamic-charging-ratio.csv"
                            }
                        }
                    }
                },
                "ers_max_cap": {
                    "locations": {}
                }
            }
        }
        
        # Fill in ers_max_cap from df_max
        for _, row in df_max.iterrows():
            loc = row["locs"]
            max_cap = row["max_cap_ers_100gw"]
            overrides["overrides"]["ers_max_cap"]["locations"][f"{loc}.techs"] = {
                "electric_road_system": {
                    "constraints": {
                        "energy_cap_max": max_cap
                    }
                }
            }
        
        # Save YAML
        yaml_file = folder / filename
        with open(yaml_file, "w") as f:
            yaml.dump(
                overrides,
                f,
                sort_keys=False,
                default_flow_style=False,
                indent=4
            )
        
        print(f"YAML saved to: {yaml_file}")

    df_max = compute_max_cap(_split_harmonise_file(input_path, file_name))

    return



#%% extend hydrogen and CO2 demand to subregions

expand_yaml_locs("additional-fuel-group-constraints-2018", cnf.SUBREGIONS, output_path=None)
expand_yaml_locs("biogas-supply-2018", cnf.SUBREGIONS, output_path=None)

DEMAND_SHARE_TRANSPORT_DICT = {
    "passenger_car_transport": [
        "passenger_car_transport_ev",
        "passenger_car_transport_fcev",
        "passenger_car_transport_ice"
    ],
    "bus_transport": [
        "bus_transport_ev",
        "bus_transport_fcev",
        "bus_transport_ice"
    ],
    "heavy_duty_transport": [
        "heavy_duty_transport_ev",
        "heavy_duty_transport_fcev",
        "heavy_duty_transport_ice"
    ],
    "light_duty_transport": [
        "light_duty_transport_ev",
        "light_duty_transport_fcev",
        "light_duty_transport_ice"
    ],
    "motorcycle_transport": [
        "motorcycle_transport_ev",
        "motorcycle_transport_fcev",
        "motorcycle_transport_ice"
    ]
}



#%% Disaggregate transport demand to subregions

df = pd.read_csv(Path(cnf.PROJECT_ROOT / f"resources/transport-data/annual-road-transport-distance-demand-iceland.csv"))
df_share = transport_shares_to_df("vehicle_group_constraints_2018")
df_transport_demand = disaggregate_transport(df, df_share)

save_hourly_profiles_by_tech(df_transport_demand, 2018)

file_list = ["bus-charging.csv",
            "heavy-duty-transport-charging.csv",
            "light-duty-transport-charging.csv",
            "motorcycle-charging.csv",
            "v1g-charging-profile-low.csv",
            "v2g-charging-profile-low.csv",
            "uncoordinated-charging-profile-norm.csv"]

expand_profiles_to_subregions(file_list, cnf.SUBREGIONS, 2018)

save_demand_share_yaml_literal(cnf.SUBREGIONS, DEMAND_SHARE_TRANSPORT_DICT, "transport_demand_share")

#%% Brownfield capapcity


df_br = load_brownfield_parquets()
df_br = map_techs(df_br, TECH_DICT)
write_brownfield_override(df_br, "/yaml-files/brownfield_new.yaml")


#%% EV battery cap max

distribute_ev_battery_caps(
    subregions=cnf.SUBREGIONS,
    df_share=df_share,
    input_file="ev_battery_cap_max.yaml")

#%% Generate demand, capacity max, and time-varying constraint for ERS

ers_elec_demand_path = cnf.PROJECT_ROOT / "resources/ers_data/250KW_HGV_DYN_STAT_FULL.csv"
ers_usage_profile_path = cnf.PROJECT_ROOT / "resources/ers_data/HGV_COUNT_FULL.csv"

build_ers_timeseries(
    ers_elec_demand_path,
    ers_usage_profile_path,
    cnf.SUBREGIONS,
    truck_efficiency_kwh_per_km=1.8,  # default, adjust if needed
    ers_eff = 0.9,
    demand_type="Stationary",  # or "Dynamic"
)

#%% Create timevarying max constraints for ERS charging technologies

ers_scenario_list = ["200", "250", "300"]

for ers_scenario in ers_scenario_list:
    ers_elec_demand_path = cnf.PROJECT_ROOT / f"resources/ers_data/{ers_scenario}KW_HGV_DYN_STAT_FULL.csv"
    split_harmonise_file(ers_elec_demand_path, ers_scenario)

    generate_ers_timevarying_constraints(ers_elec_demand_path, ers_scenario)

create_ers_cap_max_yaml(ers_scenario_list, "dynamic")

#%% Comparison Heavy-duty and ERS profiles

heavy_duty_demand = pd.read_csv(cnf.PROJECT_ROOT / "results/timeseries/heavy-duty-transport-demand.csv")
heavy_duty_charging = pd.read_csv(cnf.PROJECT_ROOT / "results/timeseries/heavy-duty-transport-charging.csv")

heavy_duty_dynamic = pd.read_csv(cnf.PROJECT_ROOT / "results/timeseries/heavy-duty-dynamic-charging-250.csv")
heavy_duty_stationary = pd.read_csv(cnf.PROJECT_ROOT / "results/timeseries/heavy-duty-stationary-charging-250.csv")

heavy_duty_demand_norm = normalise_profiles(heavy_duty_demand)
heavy_duty_charging_norm = normalise_profiles(heavy_duty_charging)

heavy_duty_dynamic_norm = normalise_profiles(heavy_duty_dynamic)
heavy_duty_stationary_norm = normalise_profiles(heavy_duty_stationary)

plot_profiles_window(
    {
        "demand": heavy_duty_demand_norm,
        "charging": heavy_duty_charging_norm,
        "dynamic": heavy_duty_dynamic_norm,
        "stationary": heavy_duty_stationary_norm,
    },
    ["2018-03-12 00:00", "2018-03-19 00:00"],
    agg=None,
    regions=["DEU_1"]
)


#%% Transport tech costs

# ERS truck

extra_erc_cost = 20000
battery_capacity_ev = 680 # kWh - assumption
battery_capacity_ers = 200 # kWh e.g. 800 - {400} = 400
battery_cost = 168.954 # EUR/kWh, https://doi.org/10.1038/s41560-024-01531-9 174 2020EUR to 2015 (2.9% inflation)
battery_cost_difference = battery_cost * (battery_capacity_ers - battery_capacity_ev) # EUR, the cost reduction for ERS trucks compared to EV trucks thanks to the smaller battery

# Methodology: the unit cost per vehicle is reidstributed on the peak demand to have a real cost
# of the entire fleet

light_duty_cost_path = cnf.PROJECT_ROOT / f"resources/data/transport_cost_burke_2024.csv"
light_duty_cost = pd.read_csv(os.path.join(light_duty_cost_path))

target_dict = {
    "inv": 0.0,
    "om_var": None,  # will be interpreted as no floor constraint
}

cost_light_transport = expand_vehicle_cost(
    light_duty_cost, final_year=2050, step=5, fit_start_year=2030, target=target_dict
)
data_dict = deepcopy(transport_cost)
exchange_rate = 0.8099 * 0.919  # from 2024USD to 2015USD and from 2015USD to 2015EUR

for year in [2020, 2030, 2050]:
    for category in cost_light_transport.category.unique():
        for vehicle_type in cost_light_transport.vehicle_type.unique():
            for cost_class in cost_light_transport.cost_class.unique():
                if cost_class == "om_var":
                    multiplier = exchange_rate / 1.60934  # convertion of miles
                else:
                    multiplier = exchange_rate
                value = cost_light_transport[
                    (cost_light_transport["year"] == year)
                    & (cost_light_transport["category"] == category)
                    & (cost_light_transport["vehicle_type"] == vehicle_type)
                    & (cost_light_transport["cost_class"] == cost_class)
                ]["value"].to_numpy()[0]

                data_dict[category][str(year)][vehicle_type][cost_class] = (
                    value * exchange_rate
                )  # USD to EUR

# Add ERS truck cost

ers = {
        "2020": {
            "ers": {"inv": 315330 + extra_erc_cost + battery_cost_difference, "om_var": 0.0, "om_fix": 14617},
        },
        "2030": {
            "ers": {"inv": 153180 + extra_erc_cost + battery_cost_difference, "om_var": 0.0, "om_fix": 14617},
        },
        "2050": {
            "ers": {"inv": 139380 + extra_erc_cost + battery_cost_difference, "om_var": 0.0, "om_fix": 14617},
        },
    }

for year, tech_dict in ers.items():
    data_dict["heavy_duty_transport"][year]["ers"] = tech_dict["ers"]

demand_dict = {
    "passenger-car-transport-demand": "passenger_car_transport",
    "light-duty-transport-demand": "light_duty_transport",
    "bus-transport-demand": "bus_transport",
    "motorcycle-transport-demand": "motorcycle_transport",
    "heavy-duty-transport-demand": "heavy_duty_transport",
}
peak_demand_dict = {}
df_num_vehicles = pd.read_csv(cnf.PROJECT_ROOT / "resources/data/number-road-vehicles.csv")
peak_vehicles_dict = {}

# Find peak demand transport data

for demand in demand_dict.keys():  # noqa: PLC0206
    peak_demand_dict[demand] = {}
    # Read the CSV file
    df_demand = pd.read_csv(cnf.PROJECT_ROOT / f"resources/transport-data/{demand}.csv")
    peak_demand_dict[demand] = {
        col: df_demand[col].min() for col in df_demand.columns[1:]
    }  # 0.1 TWh

    num_vehicles = df_num_vehicles[["country_code", demand_dict[demand]]]

    num_vehicles = num_vehicles.set_index("country_code")  # Only needed once

    peak_vehicles_dict[demand_dict[demand]] = {
        country: (-peak) / num_vehicles.loc[country, demand_dict[demand]]  # 100 Mio vkm / vehicle
        for country, peak in peak_demand_dict[demand].items()
        if country in num_vehicles.index
    }

df_peak_vehicles = pd.DataFrame(peak_vehicles_dict)
df_peak_vehicles = df_peak_vehicles.reset_index().rename(columns={"index": "locs"})

adjusted_transport_costs = {  # 10 EUR2015/vkm_per_hour
    demand: {
        year: {
            tech: {
                country: {
                    "inv": cost["inv"]
                    / 1000000000
                    / df_peak_vehicles.set_index("locs").loc[country, demand],
                    "om_fix": cost["om_fix"]
                    / 1000000000
                    / df_peak_vehicles.set_index("locs").loc[country, demand],
                    "om_var": cost["om_var"] / 10,
                }
                for country in df_peak_vehicles["locs"]
                if demand in df_peak_vehicles.columns
                and not pd.isna(df_peak_vehicles.set_index("locs").loc[country, demand])
                and df_peak_vehicles.set_index("locs").loc[country, demand] != 0
            }
            for tech, cost in techs.items()
        }
        for year, techs in years.items()
    }
    for demand, years in data_dict.items()
}

records = []

# Unit of measurement (inv, om_fix): 10 EUR2015/vkm_per_hour; om_var: EUR2015/vkm
for demand_type, years in adjusted_transport_costs.items():
    for year_str, techs in years.items():
        for technology, countries in techs.items():
            for country, cost_components in countries.items():
                records.append(
                    {
                        "demand_type": demand_type,
                        "year": year_str,
                        "technology": technology,
                        "country": country,
                        **cost_components,  # Unpacks inv, om_fix, om_var
                    }
                )

df_costs = pd.DataFrame(records)
df_costs["demand_tech"] = df_costs["demand_type"] + "_" + df_costs["technology"]

# I have computed the cost per peak demand per each country, and I extend the same costs for each subregion
# Computing the peak cost per subregion would have brought to the same results since the number of vehicles would have been reallocated according to the demand.

subregion_map = defaultdict(list)

for sub in cnf.SUBREGIONS:
    country = sub.split("_")[0]
    subregion_map[country].append(sub)

df_expanded = (
    df_costs
    .assign(country=lambda df: df["country"].map(subregion_map))
    .explode("country")
)

output_path = cnf.PROJECT_ROOT / "results/transport-cost/transport_costs.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df_costs.to_csv(output_path, index=False)

write_yaml_transport_costs(df_expanded, 2020)
write_yaml_transport_costs(df_expanded, 2030)
write_yaml_transport_costs(df_expanded, 2050)

#%% Cost of infrastructure ERS

number_of_trucks_per_km = 8
power_per_truck = 200 # kW
ers_cost_inv = 1_700_000 # EUR/km
ers_cost_om_fix = 0.88 # EUR/km/year 

heavy_duty_dynamic = pd.read_csv(cnf.PROJECT_ROOT / "results/timeseries/heavy-duty-dynamic-charging-250.csv")
ers_locs = list(heavy_duty_dynamic.columns[1:]) # skip the "timesetep" column

output_path = cnf.PROJECT_ROOT / "results/yaml-files/ers_costs.yaml"
df_lengths = pd.read_csv(cnf.PROJECT_ROOT / "resources/ers_data/ers_length.csv")
df_lengths = df_lengths[df_lengths["locs"].isin(ers_locs)].reset_index()

ers_cost_inv_per_peak_power = ers_cost_inv / (number_of_trucks_per_km * power_per_truck) # EUR/kW
ers_cost_om_fix_per_peak_power = ers_cost_om_fix / (number_of_trucks_per_km * power_per_truck) # EUR/kW/year

build_and_save_ers_yaml(
    df_lengths,
    ers_cost_inv_per_peak_power,
    ers_cost_om_fix_per_peak_power,
    output_path
)

#%% EV and ERS battery limits

df = pd.read_csv(Path(cnf.PROJECT_ROOT / f"resources/transport-data/annual-road-transport-distance-demand-iceland.csv"))
df_share = transport_shares_to_df("vehicle_group_constraints_2018")

vehicle_num = pd.read_csv(cnf.PROJECT_ROOT / "resources/data/number-road-vehicles.csv")

df_vehicle_split = split_vehicle_numbers_by_share(
    vehicle_num=vehicle_num,
    df_share=df_share
)

ev_battery_limit_dict = {}

ev_battery_limit_dict["2030"] = battery_fleet(df_vehicle_split, cnf.EV_BATTERY_SIZE_2030)
ev_battery_limit_dict["2050"] = battery_fleet(df_vehicle_split, cnf.EV_BATTERY_SIZE_2050)

for year, df in ev_battery_limit_dict.items():
    generate_yaml_ev_battery_cap_max(
        df, os.path.join(cnf.PROJECT_ROOT, "results", "yaml-files", f"ev_battery_cap_max_{year}.yaml")
    )

heavy_duty_dynamic = pd.read_csv(cnf.PROJECT_ROOT / "results/timeseries/heavy-duty-dynamic-charging-250.csv")
ers_locs = list(heavy_duty_dynamic.columns[1:]) # skip the "timesetep" column

ers_battery_limit_dict = {}

ers_battery_limit_dict["2030"] = battery_fleet(df_vehicle_split, cnf.ERS_BATTERY_SIZE_2030, loc_list=ers_locs).dropna()
ers_battery_limit_dict["2050"] = battery_fleet(df_vehicle_split, cnf.ERS_BATTERY_SIZE_2050, loc_list=ers_locs).dropna()

for year, df in ers_battery_limit_dict.items():
    generate_yaml_ers_battery_cap_max(
        df, os.path.join(cnf.PROJECT_ROOT, "results", "yaml-files", f"ers_battery_cap_max_{year}.yaml")
    )

#%% Storage cap max ERS trucks


#%% Links

map_path = "C:\\Users\\sanvi\\GitHub\\friendly-maritime-shapes\\results\\ehighways\\ehighways.parquet"
gdf = gpd.read_parquet(map_path)
yaml_path = "D:\\transfer\\ERS\\2030\\links.yaml"

def plot_grid_from_yaml(yaml_path, gdf, linewidth_scale=10):
    import yaml
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from shapely.geometry import LineString

    # --- 1. Prepare base data ---
    gdf = gdf.copy()

    # Clean node id from shape_id
    def clean_id(shape_id):
        return "_".join(shape_id.split("_")[:2])

    gdf["node"] = gdf["shape_id"].apply(clean_id)

    land_gdf = gdf[gdf["shape_class"] == "land"]
    maritime_gdf = gdf[gdf["shape_class"] == "maritime"]

    # --- 2. Read YAML ---
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    links = data.get("links", {})

    # --- 3. Aggregate capacities ---
    capacity = defaultdict(float)

    for key, value in links.items():
        locs = key.split(".")[0]
        loc1, loc2 = locs.split(",")

        # avoid duplicates (A,B) vs (B,A)
        pair = tuple(sorted([loc1, loc2]))

        for tech in value.values():
            cap = tech.get("constraints", {}).get("energy_cap_min", 0)
            capacity[pair] += cap

    # --- 4. Get node positions (centroids) ---
    nodes = land_gdf.set_index("node").geometry.centroid

    # --- 5. Build line geometries ---
    lines = []
    for (n1, n2), cap in capacity.items():
        if n1 in nodes.index and n2 in nodes.index:
            line = LineString([nodes[n1], nodes[n2]])
            lines.append({"geometry": line, "cap": cap})

    lines_gdf = gpd.GeoDataFrame(lines, crs=gdf.crs)

    # --- 6. Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Maritime background
    if not maritime_gdf.empty:
        maritime_gdf.plot(ax=ax, color="lightblue", edgecolor="none")

    # Land
    land_gdf.plot(ax=ax, color="#F4E2B8", edgecolor="#927961", linewidth=0.5)

    # Lines
    if not lines_gdf.empty:
        lines_gdf.plot(
            ax=ax,
            linewidth=lines_gdf["cap"] * linewidth_scale,
            color="red",
            alpha=0.7
        )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_randomized_grid(
    yaml_path,
    gdf,
    perc_range=(-0.5, 0.5),
    linewidth_scale=8,
    seed=None,
    min_abs_change=0.05
):
    import yaml
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import LineString

    # --- reproducibility ---
    if seed is not None:
        np.random.seed(seed)

    # --- 1. Prepare data ---
    gdf = gdf.copy()

    def clean_id(shape_id):
        return "_".join(shape_id.split("_")[:2])

    gdf["node"] = gdf["shape_id"].apply(clean_id)

    land_gdf = gdf[gdf["shape_class"] == "land"]
    maritime_gdf = gdf[gdf["shape_class"] == "maritime"]

    # --- 2. Read YAML (only to get pairs) ---
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    links = data.get("links", {})

    # --- 3. Extract unique pairs ---
    pairs = set()
    for key in links.keys():
        locs = key.split(".")[0]
        loc1, loc2 = locs.split(",")
        pair = tuple(sorted([loc1, loc2]))
        pairs.add(pair)

    # --- 4. Assign random variation (guaranteed variation) ---
    lower, upper = perc_range

    random_values = {}
    for pair in pairs:
        val = np.random.uniform(lower, upper)

        # enforce minimum variation
        while abs(val) < min_abs_change:
            val = np.random.uniform(lower, upper)

        random_values[pair] = 1 + val

    # --- DEBUG (optional) ---
    # print(random_values)

    # --- 5. Node positions ---
    nodes = land_gdf.set_index("node").geometry.centroid

    # --- 6. Build line geometries ---
    lines = []
    for (n1, n2), val in random_values.items():
        if n1 in nodes.index and n2 in nodes.index:
            delta = val - 1
            line = LineString([nodes[n1], nodes[n2]])
            lines.append({"geometry": line, "value": val, "delta": delta})

    lines_gdf = gpd.GeoDataFrame(lines, crs=gdf.crs)

    # --- 7. Normalize linewidths ---
    if not lines_gdf.empty:
        max_delta = lines_gdf["delta"].abs().max()
        if max_delta == 0:
            lines_gdf["width"] = 1
        else:
            lines_gdf["width"] = lines_gdf["delta"].abs() / max_delta

    # --- 8. Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Maritime
    if not maritime_gdf.empty:
        maritime_gdf.plot(ax=ax, color="lightblue", edgecolor="none")

    # Land
    land_gdf.plot(ax=ax, color="#F4E2B8", edgecolor="#927961", linewidth=0.5)

    if not lines_gdf.empty:
        positive = lines_gdf[lines_gdf["delta"] >= 0]
        negative = lines_gdf[lines_gdf["delta"] < 0]

        # Expansion (red)
        if not positive.empty:
            positive.plot(
                ax=ax,
                linewidth=positive["width"] * linewidth_scale,
                color="red",
                alpha=0.7,
                label="Expansion"
            )

        # Reduction (green)
        if not negative.empty:
            negative.plot(
                ax=ax,
                linewidth=negative["width"] * linewidth_scale,
                color="green",
                alpha=0.7,
                label="Reduction"
            )

    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig, ax, random_values


def plot_randomized_grid(
    yaml_path,
    gdf,
    perc_range=(-0.5, 0.5),
    linewidth_scale=8,
    seed=None,
    min_abs_change=0.05
):
    import yaml
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import LineString

    # --- reproducibility ---
    if seed is not None:
        np.random.seed(seed)

    # --- 1. Prepare data ---
    gdf = gdf.copy()

    def clean_id(shape_id):
        return "_".join(shape_id.split("_")[:2])

    gdf["node"] = gdf["shape_id"].apply(clean_id)

    land_gdf = gdf[gdf["shape_class"] == "land"]
    maritime_gdf = gdf[gdf["shape_class"] == "maritime"]

    # --- 2. Read YAML ---
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    links = data.get("links", {})

    # --- 3. Extract unique pairs ---
    pairs = set()
    for key in links.keys():
        locs = key.split(".")[0]
        loc1, loc2 = locs.split(",")
        pair = tuple(sorted([loc1, loc2]))
        pairs.add(pair)

    # --- 4. Random transmission variation ---
    lower, upper = perc_range
    random_values = {}

    for pair in pairs:
        val = np.random.uniform(lower, upper)
        while abs(val) < min_abs_change:
            val = np.random.uniform(lower, upper)
        random_values[pair] = 1 + val

    # --- 5. Node positions ---
    nodes = land_gdf.set_index("node").geometry.centroid

    # --- 6. Build lines ---
    lines = []
    for (n1, n2), val in random_values.items():
        if n1 in nodes.index and n2 in nodes.index:
            delta = val - 1
            line = LineString([nodes[n1], nodes[n2]])
            lines.append({"geometry": line, "value": val, "delta": delta})

    lines_gdf = gpd.GeoDataFrame(lines, crs=gdf.crs)

    # --- 7. Normalize linewidths ---
    if not lines_gdf.empty:
        max_delta = lines_gdf["delta"].abs().max()
        if max_delta == 0:
            lines_gdf["width"] = 1
        else:
            lines_gdf["width"] = lines_gdf["delta"].abs() / max_delta

    # --- 8. Plot base ---
    fig, ax = plt.subplots(figsize=(10, 10))

    if not maritime_gdf.empty:
        maritime_gdf.plot(ax=ax, color="lightblue", edgecolor="none")

    land_gdf.plot(ax=ax, color="#F4E2B8", edgecolor="#927961", linewidth=0.5)

    # --- 9. Plot lines ---
    if not lines_gdf.empty:
        positive = lines_gdf[lines_gdf["delta"] >= 0]
        negative = lines_gdf[lines_gdf["delta"] < 0]

        if not positive.empty:
            positive.plot(
                ax=ax,
                linewidth=positive["width"] * linewidth_scale,
                color="red",
                alpha=0.7,
                label="Expansion"
            )

        if not negative.empty:
            negative.plot(
                ax=ax,
                linewidth=negative["width"] * linewidth_scale,
                color="green",
                alpha=0.7,
                label="Reduction"
            )

    # --- 10. Add pie charts at centroids ---
    for idx, row in land_gdf.iterrows():
        centroid = row.geometry.centroid
        x, y = centroid.x, centroid.y

        # random share (0–18%)
        blue_share = np.random.uniform(0, 0.18)
        yellow_share = 1 - blue_share

        # pie
        ax.pie(
            [blue_share, yellow_share],
            colors=["blue", "yellow"],
            radius=0.3,   # adjust size
            center=(x, y),
            wedgeprops={"linewidth": 0.2, "edgecolor": "black"}
        )

    # --- 11. Final ---
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig, ax, random_values

map_path = "C:\\Users\\sanvi\\GitHub\\friendly-maritime-shapes\\results\\ehighways\\ehighways.parquet"
gdf = gpd.read_parquet(map_path)
yaml_path = "D:\\transfer\\ERS\\2030\\links.yaml"

plot_heavy_transport_mix_map(gdf, scenario="2030-1x-250-3h")
plot_electricity_mix_map(gdf, yaml_path, scenario="2030-1x-250-3h")