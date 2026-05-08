#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    prepare_nodes,
    iso2_to_iso3,
    )
from src.ferm.config import Config


import pandas as pd
import numpy as np
def create_feature_matrix(niche_path, normalize=True):
    
    niche = load_niche_data(niche_path, niche_type="gdp_per_capita_2018")
    niche.set_index(keys='iso3', inplace=True)

    origin = niche['gdp_per_capita_2018'].to_numpy()[:, None]
    destination = niche['gdp_per_capita_2018'].to_numpy()[None, :]
    
    if normalize:
        # Normalization (zscore log)
        diff = np.log1p(origin) - np.log1p(destination)
        diff = (diff - np.nanmean(diff))/np.nanstd(diff)
    else:
        diff = origin - destination
    df = pd.DataFrame(
            data = diff,
            index = niche.index,
            columns = niche.index
        )    
    return df
    

def load_stock_matrix(stock_path, normalize:True) -> pd.DataFrame:
    
    df = pd.read_csv(config.stock_path, index_col=0)
    
    if normalize:
        data = df.to_numpy()
        pass
    
    
    return df
    


#%% Data preparation

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


# Split into periods
pair_lookup = split_flows_by_period(
    flows, 
    master_country
    )


country = add_niche(
    country, 
    niche_col=config.niche_type, 
    method=config.niche_method
    )


nodes = prepare_nodes(country, flows)

nodes  = nodes.drop(nodes[nodes['iso3'] == 'TWN'].index)
#%% Test 

features = create_feature_matrix(config.niche_path)

ferm = FERM(
    nodes,
    flows,
    features,
     )

res = ferm.run(
    num_particles = int(1e4),
    sigma = 5.0, 
    niche_col = "niche", 
    verbose = False)

rm = RM(
        nodes,
        flows,
        )

res_rm = rm.run()


#%% Analysis periods 
from src.ferm.plotting import plot_rm_vs_ferm_error_scatter, plot_timeseries_migrants

only_label="precovid"

comparisons_radiation = {}
comparisons_ferm = {}

for label, flows_partial in pair_lookup.items():
    
    flows_partial.rename({'total_migrants':'num_migrants'}, axis=1, inplace=True)
    
    if only_label is not None and label != only_label:
        continue
    if len(flows_partial) == 0:
        continue
    print(f"Computing period: {label}")
    
    nodes_rm = prepare_nodes(country, flows_partial)
    nodes_rm  = nodes_rm.drop(nodes_rm[nodes_rm['iso3'] == 'TWN'].index)
    
    rm = RM(
        nodes=nodes_rm,
        flows=flows_partial,
    )
    results = rm.run()
    
    comparisons_radiation[label] = {
        "nodes": nodes_rm,
        "D": results.distance_matrix,
        "S": results.intervening_population_matrix,
        "P": results.probability_matrix,
        "comparison": results.comparison
    }
    
    ferm = FERM(
        nodes=nodes_rm,
        flows=flows_partial,
        features=features
    )

    results = ferm.run(
        num_particles = 500000, # config.num_particles,
        sigma = 5.0, #config.sigma, 
        niche_col = "niche", 
        verbose = False)

    comparisons_ferm[label] = {
        "nodes": nodes_rm,
        "D": results.distance_matrix,
        "P": results.probability_matrix,
        "comparison": results.comparison
    }


#%% Plotting
comp_rm = comparisons_radiation[only_label]["comparison"]
comp_f = comparisons_ferm[only_label]["comparison"]

#TODO fix this column name
comp_rm.rename({'num_migrants':'total_migrants'}, axis=1, inplace=True)
comp_f.rename({'num_migrants':'total_migrants'}, axis=1, inplace=True)

plot_rm_vs_ferm_error_scatter(
    comp_rm,
    comp_f,
    label=only_label,
    metric="abs_log",
    niche_type= config.niche_type
)


#%% Plot time series migrants
plot_timeseries_migrants(migrations)




