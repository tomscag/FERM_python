import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


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
    from matplotlib.axes import Axes
    from pycirclize import Circos

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
        Axes,
        Circos,
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
    return country, flows, nodes, pair_lookup


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select period
    """)
    return


@app.cell
def _(flows):
    flows_sel = flows.loc[(flows.year==2019)  & (flows.month <=12)]
    flows_sel = flows_sel.groupby(["country_from", "country_to"], as_index=False)["num_migrants"].sum()
    flows_sel
    return (flows_sel,)


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

        #/ Node feature matrix
        df, _ = load_niche_data(niche_type=node_feature)
        df["log_gdp"] = np.log(df[node_feature])
        df["log_gdp_norm"] = (df["log_gdp"]- df["log_gdp"].mean())/df["log_gdp"].std()
        df = df.rename(columns={"iso3":"iso3_o"})
        df["iso3_d"] = df["iso3_o"]

        # Build node feature matrix
        df = df.pivot(index="iso3_o", columns="iso3_d", values="log_gdp_norm")
        df = df.ffill(axis=0).bfill(axis=0) # Fill rows
        # print(df)

        #/ Relational feature matrix
        dfr, niche_name = load_niche_data(niche_type=relational_feature) 

        dfr = dfr.pivot(
            index="iso3_o",
            columns="iso3_d",
            values=niche_name,
        )
        if fillna:
            # print(1-(np.isnan(df.to_numpy()).sum()-235)/(235*234)) # Count nan
            dfr.fillna(dfr.mean().mean(), inplace=True)

        # Fill to zero the diagonal terms
        for label in dfr.index:
            dfr.loc[label, label] = 0

        # Combine (add) node features with relational features
        common_cols = df.index.intersection(dfr.index)
        dfr = dfr.loc[common_cols, common_cols]
        df = df.loc[common_cols, common_cols]

        df_all = dfr + df

        # dfr = dfr.add(df.loc[common_cols, "log_gdp_norm"],axis="index")

        return df_all, df

    df_feat_all, df_feat_nodes = load_relational_features(
        relational_feature="sci",
        node_feature="gdp_per_capita_2018",
    )
    df_feat_all
    return df_feat_all, df_feat_nodes


@app.cell
def _(df_feat_all):
    df_feat_all.loc["JPN","JPN"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run FERM
    """)
    return


@app.cell
def _(FERM, RM, df_feat_all, df_feat_nodes, flows_sel, nodes):
    SIGMA = 5.0
    NUM_PARTICLES = int(10e4)

    # GDP + SCI
    _ferm = FERM(
        nodes,
        flows_sel,
        features = df_feat_all,
         )

    res_ferm = _ferm.run(
        num_particles = NUM_PARTICLES,
        sigma = SIGMA, 
        verbose = False).comparison

    # GDP
    _ferm = FERM(
        nodes,
        flows_sel,
        features = df_feat_nodes,
         )

    res_ferm_GDP = _ferm.run(
        num_particles = NUM_PARTICLES,
        sigma = SIGMA, 
        verbose = False).comparison

    # RM
    _rm = RM(
            nodes,
            flows_sel,
            )

    res_rm = _rm.run().comparison
    return res_ferm, res_ferm_GDP, res_rm


@app.cell
def _():
    return


@app.cell
def _(np, pd, res_ferm, res_ferm_GDP, res_rm):
    def coefficient_of_determination(df:pd.DataFrame, mode:str="log") -> float:

        if mode == "log":
            df = df.loc[ (df["num_migrants"] >0) & (df["predicted_migrants"] >0),:]
            SS_res = ((np.log10(df["num_migrants"]) - np.log10(df["predicted_migrants"]))**2).sum()    
            SS_tot = ((np.log10(df["num_migrants"]) - np.log10(df["num_migrants"]).mean())**2).sum()
        elif mode=="normal":
            SS_res = ((df["num_migrants"] - df["predicted_migrants"])**2).sum()    
            SS_tot = ((df["num_migrants"] - df["num_migrants"].mean())**2).sum()

        R2 = 1 - SS_res/SS_tot
        return R2

    mode = "normal"
    R2_ferm = coefficient_of_determination(res_ferm, mode=mode)
    R2_ferm_GDP = coefficient_of_determination(res_ferm_GDP, mode=mode)
    R2_rm = coefficient_of_determination(res_rm, mode=mode)


    print(f"FERM ALL:\t{R2_ferm:.3f}")
    print(f"FERM GDP:\t{R2_ferm_GDP:.3f}")
    print(f"RM:\t\t\t{R2_rm:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatter plot data versus models
    """)
    return


@app.cell
def _(Axes, FIGDIR, np, pd, plt, res_ferm, res_rm):
    def plot_scatter(df:pd.DataFrame, label:str="FERM", ax:Axes=None):
        """
        Plot data versus model predictions
        """
        # Filter dataset
        df = df.loc[
        (df["num_migrants"]>0) & (df["predicted_migrants"]>0),:]
    
        n_bins = 12
        if ax is None:
            ax: Axes
            _, ax = plt.subplots(figsize=(6,6))

        x = df["num_migrants"].to_numpy()
        y = df["predicted_migrants"].to_numpy()
    
        ax.scatter(
            x=x, 
            y=y,
            color='k',        
            alpha=0.25,
            zorder=1,
        )

        # Logarithmically spaced x bins.
        bin_edges = np.geomspace(
            x.min(),
            x.max(),
            n_bins + 1,
        )

        # Assign each observation to a bin.
        bin_indices = np.digitize(x, bin_edges) - 1

        box_data = []
        box_positions = []
        box_widths = []

        for i in range(n_bins):
            values = y[bin_indices == i]

            if values.size == 0:
                continue

            left = bin_edges[i]
            right = bin_edges[i + 1]

            # Geometric midpoint, appropriate for a log-scaled axis.
            position = np.sqrt(left * right)

            box_data.append(values)
            box_positions.append(position)

            # Width expressed in the original x coordinates.
            box_widths.append(0.55 * (right - left))

        ax.boxplot(
            box_data,
            positions=box_positions,
            widths=box_widths,
            vert=True,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={
                "facecolor": "tab:orange",
                "alpha": 0.9,
                "edgecolor": "tab:blue",
            },
            medianprops={
                "color": "tab:blue",
                "linewidth": 2,
            },
            whiskerprops={
                "color": "tab:blue",
            },
            capprops={
                "color": "tab:blue",
            },
            zorder=2,
        )

    
    
        ax.plot([1e-1,1e6],[1e-1,1e6],color="tab:red",linestyle="--",alpha=0.95)
        ax.set_ylabel("Migrants (model)", fontsize=18)
        ax.set_xlabel("Migrants (data)", fontsize=18)
        ax.set(xscale="log", yscale="log")
        ax.set_xlim([1e-1,1e6])
        ax.set_ylim([1e-1,1e6])
        ax.set_title(label, color="tab:red", fontsize=16)

        return ax


    # plot_scatter(res_rm_agg)
    df_list, labels = [res_rm, res_ferm], ["RM", "FERM"]
    fig, _axes = plt.subplots(1,2,figsize=(12,6))

    for _idx, _ax in enumerate(_axes):
        plot_scatter(df=df_list[_idx], ax=_ax, label=labels[_idx])

    plt.savefig(FIGDIR / "scatter_plot.pdf", bbox_inches="tight")
    plt.show()


    return


@app.cell
def _(data):
    # pd.DataFrame(data={"a":res_ferm["predicted_migrants"], "b":res_ferm["predicted_migrants"].round()})


    data

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatter plot residuals
    """)
    return


@app.cell
def _(Axes, FIGDIR, np, plt, res_ferm, res_rm):
    EPS: float = 1.0
    def plot_scatter_residuals(
        df1,
        df2,
        method: str = "abs_log_ratio"
    ) -> Axes:

        df1["residuals"] = np.log(
            np.abs(df1["predicted_migrants"] - df1["num_migrants"]))
        df2["residuals"] = np.log(
            np.abs(df2["predicted_migrants"] - df2["num_migrants"]))

        df1["abs_log_ratio"] = np.abs(
            np.log10((df1["predicted_migrants"] + EPS) / (df1["num_migrants"] + EPS)))
        df2["abs_log_ratio"] = np.abs(
            np.log10((df2["predicted_migrants"] + EPS) / (df2["num_migrants"] + EPS)))
    
        ax: Axes
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(
            x=df1[method], y=df2[method], alpha=0.2, color='k')
        lim = max(df1[method].max(), df2[method].max())
        ax.plot([0, lim], [0, lim], linestyle="--", color="tab:blue")
        ax.set_xlabel(f"errors RM", fontsize=16)
        ax.set_ylabel(f"errors FERM", fontsize=16)
        ax.grid(True, linestyle=":")

        return ax

    plot_scatter_residuals(res_rm, res_ferm, "abs_log_ratio")
    plt.savefig(FIGDIR / "comparison_residuals_scatter_ferm.pdf")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chord diagram
    """)
    return


@app.cell
def _(np, res_ferm_agg, res_rm_agg):
    # Performance column
    r = np.log10(np.abs(res_ferm_agg["residual"]))
    r[r<0] = 0
    res_ferm_agg["abs_log_residuals"] = r
    r = np.log10(np.abs(res_rm_agg["residual"]))
    r[r<0] = 0
    res_rm_agg["abs_log_residuals"] = r
    return


@app.cell
def _(Circos, FIGDIR, pd, plt, res_rm):
    import matplotlib.colors as mcolors

    def plot_migration_chord(
        df: pd.DataFrame,
        origin_col: str = "country_from",
        destination_col: str = "country_to",
        value_col: str = "num_migrants",
        performance_col:str = "residuals",
        min_flux: float = 0,
        top_n: int | None = None,
        figsize: tuple[float, float] = (12, 12),
    ):
        """
        Plot directed migration flows as a chord diagram.

        Parameters
        ----------
        df
            Long-form DataFrame containing origin, destination and flow.
        origin_col
            Name of the origin column.
        destination_col
            Name of the destination column.
        value_col
            Name of the migration-flow column.
        performance_col
            Name of the performance column.
        min_flux
            Exclude flows smaller than this value.
        top_n
            Retain only the top N countries by total incoming plus outgoing flow.
        figsize
            Matplotlib figure size.

        Returns
        -------
        fig
            Matplotlib figure.
        matrix
            Origin-destination matrix used in the plot.
        """

        required = {origin_col, destination_col, value_col}

        if missing := required.difference(df.columns):
            raise ValueError(f"Missing columns: {sorted(missing)}")

        data = df[
            [origin_col, destination_col, value_col]
        ].copy()

        data[value_col] = pd.to_numeric(
            data[value_col],
            errors="coerce",
        )

        data = data.dropna(
            subset=[origin_col, destination_col, value_col]
        )

        data = data.loc[data[value_col] >= min_flux]

        if top_n is not None:
            outgoing = data.groupby(origin_col)[value_col].sum()
            incoming = data.groupby(destination_col)[value_col].sum()

            total_flux = outgoing.add(incoming, fill_value=0)

            selected = total_flux.nlargest(top_n).index

            data = data.loc[
                data[origin_col].isin(selected)
                & data[destination_col].isin(selected)
            ]

        matrix = data.pivot_table(
            index=origin_col,
            columns=destination_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )

        countries = matrix.index.union(matrix.columns)

        matrix = matrix.reindex(
            index=countries,
            columns=countries,
            fill_value=0,
        )

        # Self-migration links are generally not informative.
        # common = matrix.index.intersection(matrix.columns)
        # matrix.loc[common, common] = 0


        cmap = plt.colormaps["RdYlGn_r"]

        vmin = 4 
        vmax = 5.8
        df[performance_col][df[performance_col]<=vmin] = vmin

        norm = mcolors.Normalize(
            # vmin=df[performance_col].min(),
            # vmax=df[performance_col].max(),
            vmin = vmin,
            vmax = vmax,
        )

        # Performance feature
        feature_lookup = (
            df.set_index([origin_col, destination_col])[performance_col]
        )    
        link_map = [
            (origin, destination, cmap(norm(value))) 
            for (origin, destination), value in feature_lookup.items()
        ]

        sector_colors = {
            name: [0.8,0.8,0.8]
            for name in matrix.index.union(matrix.columns)
        }

        # Draw chord diagram
        circos = Circos.chord_diagram(
            matrix,
            space=2,
            cmap=sector_colors,
            label_kws={
                "size": 15,
            },
            link_cmap=link_map,
            link_kws={
                "alpha": 0.45,
                "direction": 1,
            },        
        )
        circos.colorbar(        
            vmin=vmin,
            # vmax=df[performance_col].max(),
            vmax=vmax,
            cmap=cmap,
            label=performance_col,
        )

        fig = circos.plotfig()
        fig.set_size_inches(*figsize)

        return fig, matrix

    fig, matrix = plot_migration_chord(
        df=res_rm, 
        min_flux=30_000,
        performance_col="abs_log_residuals",
    )
    plt.savefig(FIGDIR / f"chord_diagram_rm.png")
    plt.show()
    return


@app.cell
def _(FIGDIR, plot_rm_vs_ferm_error_scatter, plt):
    # _res_ferm
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
    def plot_histogram_residuals(res_rm, res_ferm):
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
    plot_histogram_residuals(res_rm, res_ferm)
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
def _(plt, res_ferm):
    _fig, _ax = plt.subplots(figsize=(5,5))
    _ax.scatter(res_ferm.comparison["predicted_migrants"],
    res_ferm.comparison["num_migrants"])
    return


@app.cell
def _(FIGDIR, plot_rm_vs_ferm_error_scatter, plt, res_ferm_agg, res_rm_agg):
    _period = "precovid"
    _ax = plot_rm_vs_ferm_error_scatter(
        comp_rm = res_rm_agg,
        comp_ferm = res_ferm_agg,
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
def _(pd):
    df = pd.read_csv("./Gravity_csv_V202211/Gravity_V202211_bilateral_nonbinary.csv")
    df = df.loc[df.year == 2021,:]
    df

    # Sci Min-max threshold
    # thres = 1e6
    # df.loc[df["scaled_sci_2021"] >= 1e6, "scaled_sci_2021"] = 0
    # xmin, xmax = df["scaled_sci_2021"].min(), df["scaled_sci_2021"].max()
    # func = lambda x : 2*(x-xmin)/(xmax-xmin) -1
    # df["scaled_sci_2021_minmax"] = df["scaled_sci_2021"].apply(func)
    # df = df.loc[:, ["iso3_o", "iso3_d", "scaled_sci_2021_minmax"]]
    # df.to_csv(f"./data/features/scaled_sci_2021_minmax_threshold_{thres:.1e}.csv", index=False, float_format='%.3f')

    ## Sci lognorm
    # df["scaled_sci_2021_lognorm"] = (np.log10(df["scaled_sci_2021"]) - np.log10(df["scaled_sci_2021"]).mean())/np.log10(df["scaled_sci_2021"]).std()
    # df = df.loc[:, ["iso3_o", "iso3_d", "scaled_sci_2021_lognorm"]]
    # df.to_csv("./data/features/sci_gravity_2021_lognorm.csv", index=False, float_format='%.3f')



    ## Sci Min-max
    # xmin, xmax = df["scaled_sci_2021"].min(), df["scaled_sci_2021"].max()
    # func = lambda x : 2*(x-xmin)/(xmax-xmin) -1
    # df["scaled_sci_2021_norm"] = df["scaled_sci_2021"].apply(func)

    # df = df.loc[:, ["iso3_o", "iso3_d", "scaled_sci_2021_norm"]]
    # # df.head()
    # df.to_csv("./data/features/sci_gravity_2021_norm.csv", index=False, float_format='%.3f')

    ## Common religion
    # xmin, xmax = df["comrelig"].min(), df["comrelig"].max()
    # func = lambda x : 2*(x-xmin)/(xmax-xmin) -1
    # df["comrelig_2021_norm"] = df["comrelig"].apply(func)

    # df = df.loc[:, ["iso3_o", "iso3_d", "comrelig_2021_norm"]]
    # df.head()
    # df.to_csv("./data/features/comrelig_2021_norm.csv", index=False, float_format='%.3f')

    # Plot
    # plt.hist(df["scaled_sci_2021_minmax"], bins=100)
    # plt.yscale("log")
    # plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
