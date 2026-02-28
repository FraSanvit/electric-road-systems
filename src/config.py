"""Configuration file"""

from pathlib import Path

CRS = "EPSG:4326"
PROJECT_YEAR = 2030
LIFETIME = 25


PROJECT_ROOT = Path(__file__).resolve().parents[1]

scenario_name = "ehighways"
windfarm_path = PROJECT_ROOT / f"resource/data/windfarms.json"
windfarm_europe_path = PROJECT_ROOT / f"resource/data/windeurope_offshore.json"
windeurope_path = PROJECT_ROOT / f"resource/data/windeurope"
shape_path = PROJECT_ROOT / f"resource/{scenario_name}/{scenario_name}.parquet"



# https://emodnet.ec.europa.eu/geonetwork/srv/ita/catalog.search#/metadata/8201070b-4b0b-4d54-8910-abcea5dce57f