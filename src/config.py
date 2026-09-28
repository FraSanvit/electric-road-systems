"""Configuration file"""

from pathlib import Path

CRS = "EPSG:4326"
PROJECT_YEAR = 2030
LIFETIME = 25


PROJECT_ROOT = Path(__file__).resolve().parents[1]

scenario_name = "ehighways"
windfarm_path = PROJECT_ROOT / f"resources/data/windfarms.json"
windfarm_europe_path = PROJECT_ROOT / f"resources/data/windeurope_offshore.json"
windeurope_path = PROJECT_ROOT / f"resources/data/windeurope"
shape_path = PROJECT_ROOT / f"resources/{scenario_name}/{scenario_name}.parquet"

SUBREGIONS = [
    "ALB_1","AUT_1","AUT_2","AUT_3","BEL_1","BGR_1","BIH_1","CHE_1","CHE_2",
    "CYP_1","CZE_1","CZE_2","DEU_1","DEU_2","DEU_3","DEU_4","DEU_5","DEU_6","DEU_7",
    "DNK_1","DNK_2","ESP_1","ESP_10","ESP_11","ESP_2","ESP_3","ESP_4","ESP_5",
    "ESP_6","ESP_7","ESP_8","ESP_9","EST_1","FIN_1","FIN_2","FRA_1","FRA_10",
    "FRA_11","FRA_12","FRA_13","FRA_14","FRA_15","FRA_2","FRA_3","FRA_4","FRA_5",
    "FRA_6","FRA_7","FRA_8","FRA_9","GBR_1","GBR_2","GBR_3","GBR_4","GBR_5",
    "GBR_6","GRC_1","GRC_2","HRV_1","HUN_1","IRL_1","ISL_1","ITA_1","ITA_2",
    "ITA_3","ITA_4","ITA_5","ITA_6","LTU_1","LUX_1","LVA_1","MKD_1","MNE_1",
    "NLD_1","NOR_1","NOR_2","NOR_3","NOR_4","NOR_5","NOR_6","NOR_7","POL_1",
    "POL_2","POL_3","POL_4","POL_5","PRT_1","PRT_2","ROU_1","ROU_2","ROU_3",
    "SRB_1","SVK_1","SVN_1","SWE_1","SWE_2","SWE_3","SWE_4"
]

DEMAND_DICT = {
    "coaches-and-buses": "bus_transport",
    "heavy-duty-vehicles": "heavy_duty_transport",
    "light-duty-vehicles": "light_duty_transport",
    "motorcycles": "motorcycle_transport",
    "passenger-cars": "passenger_car_transport",
}

EV_BATTERY_SIZE_2050 = {
    "bus_transport": 500,
    "heavy_duty_transport": 800,
    "light_duty_transport": 100,
    "motorcycle_transport": 15,
    "passenger_car_transport": 80,
}

EV_BATTERY_SIZE_2030 = {
    "bus_transport": 330,
    "heavy_duty_transport": 680,
    "light_duty_transport": 80,
    "motorcycle_transport": 10,
    "passenger_car_transport": 60,
}

ERS_BATTERY_SIZE_2050 = {
    "heavy_duty_transport": 200,
}

ERS_BATTERY_SIZE_2030 = {
    "heavy_duty_transport": 200,
}

# https://emodnet.ec.europa.eu/geonetwork/srv/ita/catalog.search#/metadata/8201070b-4b0b-4d54-8910-abcea5dce57f