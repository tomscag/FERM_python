import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Enriched Radiation Model
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

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
        prepare_nodes,
        iso2_to_iso3,
        )
    from src.ferm.config import Config

    FIGDIR = Path("./figures")
    return (
        Config,
        FERM,
        FIGDIR,
        RM,
        add_niche,
        build_master_country_table,
        filter_flows_by_continent,
        load_country_geometries_global,
        load_migration_data,
        load_niche_data,
        load_population_data,
        mo,
        np,
        pd,
        plt,
        prepare_nodes,
        split_flows_by_period,
    )


@app.cell
def _(
    Config,
    load_country_geometries_global,
    load_migration_data,
    load_population_data,
):
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

    flows_reported = load_migration_data(config.flow_path) # Facebook data
    populations = load_population_data(config.pop_path)

    world_gdf, country_geo = load_country_geometries_global()

    if config.target_continent == "Americas":
        continent_gdf = world_gdf[world_gdf["CONTINENT"].isin(["North America", "South America", "Central America"])].copy()
    elif config.target_continent is None:
        continent_gdf = world_gdf.copy()
    else:
        continent_gdf = world_gdf[world_gdf["CONTINENT"] == config.target_continent].copy()
    return config, country_geo, flows_reported, niche_path, populations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load niche and build master table
    """)
    return


@app.cell
def _(
    add_niche,
    build_master_country_table,
    config,
    country_geo,
    create_feature_matrix,
    filter_flows_by_continent,
    flows_reported,
    load_niche_data,
    niche_path,
    populations,
    prepare_nodes,
    split_flows_by_period,
):
    niche_df = load_niche_data(niche_path, niche_type=config.niche_type)

    master_country = build_master_country_table(
        country_geo,
        populations,
        niche_df=niche_df,
        niche_col="gdp_per_capita_2018"
    )

    flows, country = filter_flows_by_continent(
        master_country, 
        flows_reported, 
        niche_type="gdp_per_capita_2018", 
        continent="Asia"
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
    return country, features, flows, nodes, pair_lookup


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run FERM
    """)
    return


@app.cell
def _(FERM, RM, features, flows, nodes):
    _ferm = FERM(
        nodes,
        flows,
        features,
         )

    res_ferm = _ferm.run(
        num_particles = int(1e4),
        sigma = 5.0, 
        niche_col = "niche", 
        verbose = False)

    _rm = RM(
            nodes,
            flows,
            )

    res_rm = _rm.run()
    return res_ferm, res_rm


@app.cell
def _(res_rm):
    res_rm.comparison
    return


@app.cell
def _(FIGDIR, plt, res_ferm, res_rm):
    def plot_residuals(res_rm, res_ferm):
        plt.hist(res_rm.comparison["residual"], bins=2000, alpha=0.6, label="RM")
        plt.hist(res_ferm.comparison["residual"], bins=2000, alpha=0.6, label="FERM")
        # plt.xscale("log")
        plt.yscale("log")
        plt.xlim([-50_000, 50_000])
        plt.xlabel("residual", fontsize=20)
        plt.ylabel("count", fontsize=20)
        plt.legend()
        plt.savefig(FIGDIR / "comparison_residuals_histogram.pdf")
        plt.show()
    plot_residuals(res_rm, res_ferm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run FERM for selected periods
    """)
    return


@app.cell
def _(pair_lookup):
    pair_lookup
    return


@app.cell
def _(FERM, FIGDIR, RM, country, features, pair_lookup, plt, prepare_nodes):
    from src.ferm.plotting import plot_rm_vs_ferm_error_scatter, plot_timeseries_migrants

    only_period=None#"precovid"
    _plot = True

    results_rm = {}
    results_ferm = {}

    for period, flows_partial in pair_lookup.items():
    
        # flows_partial.rename({'total_migrants':'num_migrants'}, axis=1, inplace=True)

        if only_period is not None and period != only_period:
            continue
        if len(flows_partial) == 0:
            continue
        print(f"Computing period: {period}")

        nodes_rm = prepare_nodes(country, flows_partial)
        nodes_rm  = nodes_rm.drop(nodes_rm[nodes_rm['iso3'] == 'TWN'].index)

        rm = RM(
            nodes=nodes_rm,
            flows=flows_partial,
        )
        results_rm[period] = rm.run()

        ferm = FERM(
            nodes=nodes_rm,
            flows=flows_partial,
            features=features
        )

        results_ferm[period] = ferm.run(
            num_particles = 500_000, # config.num_particles,
            sigma = 5.0, #config.sigma, 
            niche_col = "niche", 
            verbose = False)

        if _plot:
            _ax = plot_rm_vs_ferm_error_scatter(
                comp_rm = results_rm[period].comparison,
                comp_ferm = results_ferm[period].comparison,
                metric = "abs_log",
            )
            _ax.set_title(f"{period}")
            _ax.set_xlim([-0.2, 6.3])
            _ax.set_ylim([-0.2, 6.3])
            plt.savefig(FIGDIR / f"comparison_residuals_scatter_{period}.png")
            # plt.show()

    return only_period, plot_rm_vs_ferm_error_scatter, results_ferm, results_rm


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    FIGDIR,
    only_period,
    plot_rm_vs_ferm_error_scatter,
    plt,
    results_ferm,
    results_rm,
):
    ax = plot_rm_vs_ferm_error_scatter(
        comp_rm = results_rm[only_period].comparison,
        comp_ferm = results_ferm[only_period].comparison,
        metric = "abs_log",
    )
    plt.savefig(FIGDIR / "comparison_residuals_scatter.pdf")
    plt.show()

    #%% Plot time series migrants
    # plot_timeseries_migrants(flows_reported)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell
def _(config, load_niche_data, np, pd):
    def create_feature_matrix(niche_path, normalize=True) -> pd.DataFrame:

        niche = load_niche_data(niche_path, niche_type="gdp_per_capita_2018")
        niche.set_index(keys='iso3', inplace=True)

        origin = niche['gdp_per_capita_2018'].to_numpy()[:, None]
        destination = niche['gdp_per_capita_2018'].to_numpy()[None, :]

        if normalize:
            # Normalization (zscore log)
            diff = np.log1p(destination) - np.log1p(origin)
            diff = (diff - np.nanmean(diff))/np.nanstd(diff)
        else:
            diff = destination - origin
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

    return (create_feature_matrix,)


if __name__ == "__main__":
    app.run()
