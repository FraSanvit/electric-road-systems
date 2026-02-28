#%%
from utils import *
import config as cnf
# %%

projection_year = 2030
lifetime = 25
shape = gpd.read_parquet(cnf.shape_path)


df = windfarms_json_to_df(cnf.windfarm_path)
df_agg = aggregate_wind_capacity(df, shape, projection_year, lifetime)

plot_wind_farms_emodnet(df, shape)



#%% windeurope

df_wind = load_wind_farms_from_har_folder(windeurope_path)
df_allocated = allocate_wind_to_shapes(df_wind, shape, projection_year, lifetime)

plot_wind_farms_windeurope(df_wind, shape)


#%% compaison
plot_two_dfs_side_by_side_hatched_vertical_national(df_agg, df_allocated, label1="emodnet", label2="windeurope")

