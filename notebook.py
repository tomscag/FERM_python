#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

from src.ferm.model import FERM, RM
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
    add_niche,
    prepare_nodes
    )
from src.ferm.config import Config



#%%

config = Config(
    niche_method = "zscore_log",
    niche_type = "gdp_per_capita_2018",
    target_continent = "Asia",    
    num_particles = int(1e3),
    sigma = 1.0,
    verbose=True,
    )

niche_path = config.gdp_path

migrations = load_migration_data(config.flow_path)
populations = load_population_data(config.pop_path)

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

master_country = build_master_country_table(
    country_geo,
    populations,
    niche_df=niche_df,
    niche_col=config.niche_type
)

flows, country = filter_flows_by_continent(
    master_country, 
    migrations, 
    niche_type=config.niche_type, 
    continent=config.target_continent
    )

# pair_lookup = split_flows_by_period(
#     df_model, 
#     master_country_df
#     )


country = add_niche(country, niche_col=config.niche_type, method=config.niche_method)


nodes = prepare_nodes(country, flows)

#%% Run FERM
ferm = FERM(
    nodes,
    flows,
     )

res = ferm.run(
    num_particles = int(1e4),#config.num_particles, 
    sigma = config.sigma, 
    niche_col = "niche", # modify name
    verbose = True)


#%% Run RM
rm = RM(
        nodes,
        flows,
        )


res = rm.run()







