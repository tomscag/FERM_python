#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

from src.ferm.model import FERM
from src.ferm.preprocessing import (
    filter_flows_by_continent, 
    split_flows_by_period,    
    )
from src.ferm.utils import (
    load_migration_data, 
    load_population_data,
    load_country_geometries_global,
    load_niche_data,
    build_master_country_table,
    add_niche
    )
from src.ferm.config import Config



#%%----------------------------------------------------------------------------

config = Config(
    niche_method = "zscore_log",
    niche_type = "gdp_per_capita_2018",
    target_continent = "Asia",    
    num_particles = int(1e3),
    sigma = 1.0,
    verbose=True,
    )

niche_path = config.gdp_path

df = load_migration_data(config.flow_path)
pop = load_population_data(config.pop_path)

world_gdf, country_geo = load_country_geometries_global()

if config.target_continent == "Americas":
    continent_gdf = world_gdf[world_gdf["CONTINENT"].isin(["North America", "South America", "Central America"])].copy()
elif config.target_continent is None:
    continent_gdf = world_gdf.copy()
else:
    continent_gdf = world_gdf[world_gdf["CONTINENT"] == config.target_continent].copy()




# ----------------------------
# Load niche and build master table
# ----------------------------

niche_df = load_niche_data(niche_path, niche_type=config.niche_type)

master_country_df = build_master_country_table(
    country_geo,
    pop,
    niche_df=niche_df,
    niche_col=config.niche_type
)

flows_df, country_df = filter_flows_by_continent(
    master_country_df, 
    df, 
    niche_type=config.niche_type, 
    continent=config.target_continent
    )

# pair_lookup = split_flows_by_period(
#     df_model, 
#     master_country_df
#     )


country_df = add_niche(country_df, niche_col=config.niche_type, method=config.niche_method)


####
import pandas as pd
def ensure_columns(df, required, df_name="DataFrame"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")

def prepare_nodes(country_df, flows_df) -> pd.DataFrame:
    """ 
    Extract nodes
    """
    used_codes = sorted(set(flows_df["country_from"]).union(set(flows_df["country_to"])))
    nodes = country_df[country_df["code"].isin(used_codes)].drop_duplicates("code").copy()
    ensure_columns(nodes, ["code", "country_name", "lat", "lon", "population"], "nodes metadata")
    missing_codes = sorted(set(used_codes) - set(nodes["code"]))
    if missing_codes:
        print(f"Warning: missing metadata for these countries: {missing_codes}")
    return nodes

nodes = prepare_nodes(country_df, flows_df)
#%% Run FERM
model = FERM(
    config.niche_type,
    config.niche_method,    
     )

res = model.run(
    nodes = nodes,
    num_particles = int(1e4),#config.num_particles, 
    sigma = config.sigma, 
    niche_col = "niche", # modify name
    verbose = True)







