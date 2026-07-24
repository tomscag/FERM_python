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
    from typing import Literal

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
        Literal,
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
    return config, country_geo, flows_reported, populations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load niche and build master table
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    add_niche,
    build_master_country_table,
    config,
    country_geo,
    filter_flows_by_continent,
    flows_reported,
    load_niche_data,
    populations,
    prepare_nodes,
    split_flows_by_period,
):
    niche_df, niche_name = load_niche_data(niche_type=config.niche_type)
    # TODO: questa funzione non accetta una niche relazionale
    master_country = build_master_country_table(
        country_geo,
        populations,
        niche_df=niche_df,
        niche_col=niche_name,
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
    return country, flows, master_country, nodes, pair_lookup


@app.cell
def _(master_country):
    master_country
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build relational features
    """)
    return


@app.cell
def _(Literal, load_niche_data, np, pd):
    def load_relational_features(
        relational_feature: Literal["sci","comrelig","hdi_2020"] = "sci",
        node_feature: Literal["gdp_per_capita_2018"] = "gdp_per_capita_2018",
        normalize: str = "log_zscore",
        fillna: bool = True,
    ) -> pd.DataFrame:

        # Node feature 
        df, _ = load_niche_data(niche_type=node_feature)
        df = df.set_index("iso3")
        df["log_gdp"] = np.log(df[node_feature])
        df["log_gdp_norm"] = (df["log_gdp"]- df["log_gdp"].mean())/df["log_gdp"].std()

        # Relational feature
        dfr, niche_name = load_niche_data(niche_type=relational_feature) 

        dfr = dfr.pivot(
            index="iso3_o",
            columns="iso3_d",
            values=niche_name,
        )
        if fillna:
            # print(1-(np.isnan(df.to_numpy()).sum()-235)/(235*234)) # Count nan
            dfr.fillna(dfr.mean().mean(), inplace=True)

        # Combine (add) node features with relational features
        common_cols = df.index.intersection(dfr.index)
        dfr = dfr.loc[common_cols, common_cols]
        dfr = dfr.add(df.loc[common_cols, "log_gdp_norm"],axis="index")
     
        return dfr

    rel_feat = load_relational_features(relational_feature="sci")
    rel_feat
    return (rel_feat,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run FERM
    """)
    return


@app.cell
def _(FERM, RM, flows, nodes, rel_feat):
    _ferm = FERM(
        nodes,
        flows,
        features = rel_feat,
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
    # res_ferm.comparison.groupby( res_ferm.comparison["year"]).mean(numeric_only=True)
    res_rm.comparison
    return


@app.cell
def _(FIGDIR, plot_rm_vs_ferm_error_scatter, plt, res_ferm, res_rm):
    _res_ferm = res_ferm.comparison.groupby( [res_ferm.comparison["country_from"], res_ferm.comparison["country_to"]]).mean(numeric_only=True)
    _res_rm = res_ferm.comparison.groupby( [res_rm.comparison["country_from"], res_rm.comparison["country_to"]]).mean(numeric_only=True)
    _ax = plot_rm_vs_ferm_error_scatter(
        comp_rm = _res_rm,
        comp_ferm = _res_ferm,
        metric = "abs_log",
    )
    plt.savefig(FIGDIR / "comparison_residuals_scatter.pdf")
    plt.show()
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
    pair_lookup.keys()
    return


@app.cell
def _(FERM, FIGDIR, RM, country, pair_lookup, plt, prepare_nodes, rel_feat):
    from src.ferm.plotting import plot_rm_vs_ferm_error_scatter, plot_timeseries_migrants

    only_period="precovid"#"precovid"
    _plot = True

    results_rm = {}
    results_ferm = {}

    for period, flows_partial in pair_lookup.items():

        flows_partial.rename({'total_migrants':'num_migrants'}, axis=1, inplace=True)

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
            features=rel_feat
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
    return (plot_rm_vs_ferm_error_scatter,)


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
def _(FIGDIR, plot_rm_vs_ferm_error_scatter, plt, res_ferm, res_rm):
    _period = "precovid"
    _ax = plot_rm_vs_ferm_error_scatter(
        comp_rm = res_rm.comparison,
        comp_ferm = res_ferm.comparison,
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
def _(config, pd):
    def load_stock_matrix(stock_path, normalize:True) -> pd.DataFrame:

        df = pd.read_csv(config.stock_path, index_col=0)

        if normalize:
            data = df.to_numpy()
            pass
        return df

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scaled Social Connectdness Index
    """)
    return


@app.cell
def _(pd, plt):
    df = pd.read_csv("./Gravity_csv_V202211/Gravity_V202211_bilateral_nonbinary.csv")
    df = df.loc[df.year == 2021,:]



    xmin, xmax = df["scaled_sci_2021"].min(), df["scaled_sci_2021"].max()
    func = lambda x : 2*(x-xmin)/(xmax-xmin) -1
    df["scaled_sci_2021_norm"] = df["scaled_sci_2021"].apply(func)

    df = df.loc[:, ["year","iso3_o", "iso3_d", "scaled_sci_2021_norm"]]
    df = df.drop("year", axis=1)
    # df.head()
    df.to_csv("./data/features/sci_gravity_2021_norm.csv", index=False, float_format='%.3f')

    ## Common religion
    # xmin, xmax = df["comrelig"].min(), df["comrelig"].max()
    # func = lambda x : 2*(x-xmin)/(xmax-xmin) -1
    # df["comrelig_2021_norm"] = df["comrelig"].apply(func)

    # df = df.loc[:, ["year","iso3_o", "iso3_d", "comrelig_2021_norm"]]
    # df = df.drop("year", axis=1)
    # df.head()
    # df.to_csv("./data/features/comrelig_2021_norm.csv", index=False, float_format='%.3f')

    # Plot
    plt.hist(df["scaled_sci_2021_norm"], bins=100)
    plt.yscale("log")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
