import marimo

__generated_with = "0.24.0"
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
def _(niche_df):
    niche_df
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
    nodes
    return country, flows, niche_df, nodes, pair_lookup


@app.cell
def _():
    return


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

        return df_all, df, dfr

    df_feat_all, df_feat_nodes, dfr_feat = load_relational_features(
        relational_feature="sci",
        node_feature="gdp_per_capita_2018",
    )
    df_feat_all
    return df_feat_all, df_feat_nodes


@app.cell
def _(df_feat_all, nodes):
    # plt.imshow(df_feat_all.to_numpy())
    # plt.colorbar()
    # plt.show()
    df_feat_all.loc["JPN","JPN"]
    nodes.loc[nodes.iso3=="XKX"]
    nodes.drop(labels=236, inplace=True)
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
    return NUM_PARTICLES, SIGMA, res_ferm, res_ferm_GDP, res_rm


@app.cell
def _():
    return


@app.cell
def _(np, pd):
    def coefficient_of_determination(df:pd.DataFrame, mode:str="normal") -> float:

        if mode == "log":
            df = df.loc[ (df["num_migrants"] >0) & (df["predicted_migrants"] >0),:]
            SS_res = ((np.log10(df["num_migrants"]) - np.log10(df["predicted_migrants"]))**2).sum()    
            SS_tot = ((np.log10(df["num_migrants"]) - np.log10(df["num_migrants"]).mean())**2).sum()
        elif mode=="normal":
            SS_res = ((df["num_migrants"] - df["predicted_migrants"])**2).sum()    
            SS_tot = ((df["num_migrants"] - df["num_migrants"].mean())**2).sum()

        R2 = 1 - SS_res/SS_tot
        return R2

    return (coefficient_of_determination,)


@app.cell
def _(coefficient_of_determination, res_ferm, res_ferm_GDP, res_rm):
    mode = "normal"
    R2_ferm = coefficient_of_determination(res_ferm, mode=mode)
    R2_ferm_GDP = coefficient_of_determination(res_ferm_GDP, mode=mode)
    R2_rm = coefficient_of_determination(res_rm, mode=mode)
    R2s = [R2_rm, R2_ferm_GDP, R2_ferm]

    print(f"FERM ALL:\t{R2_ferm:.3f}")
    print(f"FERM GDP:\t{R2_ferm_GDP:.3f}")
    print(f"RM:\t\t\t{R2_rm:.3f}")
    return R2s, mode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plotting functions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatter plot: data versus predictions
    """)
    return


@app.cell
def _(Axes, np, pd, plt):
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
            whis=0,
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


    return (plot_scatter,)


@app.cell
def _(FIGDIR, R2s, plot_scatter, plt, res_ferm, res_ferm_GDP, res_rm):
    # plot_scatter(res_rm_agg)
    df_list, labels = [res_rm, res_ferm_GDP, res_ferm], ["RM", "FERM GDP", "FERM GDP+SCI"]
    fig, _axes = plt.subplots(1,3,figsize=(18,6))

    for _idx, _ax in enumerate(_axes):
        _ax = plot_scatter(df=df_list[_idx], ax=_ax, label=labels[_idx])
        _ax.text(x=0.1, y=0.9, s=f"R² = {R2s[_idx]:.3f}", transform=_ax.transAxes, fontsize=18)
        if _idx != 0:
            _ax.set_ylabel(" ")

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
    ## Scatter plot: residuals
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scaled Social Connectdness Index
    """)
    return


@app.cell
def _():
    # df = pd.read_csv("./Gravity_csv_V202211/Gravity_V202211_bilateral_nonbinary.csv")
    # df = df.loc[df.year == 2021,:]
    # df

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis US states
    """)
    return


@app.cell
def _(pd):
    gdp_df_states = pd.read_csv("data/US_census_data/features/GDP/gdp_per_capita_2014_states.csv")
    sci_df_states = pd.read_csv("data/US_census_data/features/SCI/us_states_SCI_2026_norm.csv")
    sci_df_states.head()
    return gdp_df_states, sci_df_states


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build relational features US
    """)
    return


@app.cell
def _(gdp_df_states, load_relational_features_US, sci_df_states):
    df_feat_all_us, df_us, dfr_us = load_relational_features_US(
        df_rel=sci_df_states,
        df=gdp_df_states,
        fillna=False,
    )
    df_feat_all_us
    return df_feat_all_us, df_us


@app.cell
def _():
    return


@app.cell
def _(np, pd):
    def load_relational_features_US(
        df_rel: pd.DataFrame = None,
        df: pd.DataFrame = None,
        niche_name_nodes: str = "gdp_per_capita_2014",
        niche_name_relational: str = "scaled_sci_2016_norm",
        normalize: str = "log_zscore",
        fillna: bool = True,
    ) -> pd.DataFrame:

        #/ Node feature matrix
        df["log_gdp"] = np.log(df[niche_name_nodes])
        df["log_gdp_norm"] = (df["log_gdp"]- df["log_gdp"].mean())/df["log_gdp"].std()
        df = df.rename(columns={"state_fips":"state_fips_o"})
        df["state_fips_d"] = df["state_fips_o"]

        # Build node feature matrix
        df = df.pivot(index="state_fips_o", columns="state_fips_d", values="log_gdp_norm")
        df = df.ffill(axis=0).bfill(axis=0) # Fill rows
        # print(df)

        #/ Relational feature matrix
        dfr = df_rel.pivot(
            index="state_from",
            columns="state_to",
            values=niche_name_relational,
        )
        # print(dfr)

        if fillna:
            # print(1-(np.isnan(df.to_numpy()).sum()-235)/(235*234)) # Count nan
            dfr.fillna(dfr.mean().mean(), inplace=True)

        # Fill to zero the diagonal terms
        for label in dfr.index:
            dfr.loc[label, label] = 0
        # print(dfr)

        # Combine (add) node features with relational features
        common_cols = df.index.intersection(dfr.index)
        dfr = dfr.loc[common_cols, common_cols]
        df = df.loc[common_cols, common_cols]

        df_all = dfr + df

        # dfr = dfr.add(df.loc[common_cols, "log_gdp_norm"],axis="index")

        return df_all, df, dfr


    return (load_relational_features_US,)


@app.cell
def _():
    # df_feat_all
    return


@app.cell
def _(pd):
    nodes_us = pd.read_csv("./data/US_census_data/center_of_population/CenPop2020_Mean_ST.txt")
    nodes_us.head(5)
    return (nodes_us,)


@app.cell
def _(pd):
    year = "1516"
    flows_us = pd.read_csv(f"./data/US_census_data/migrations/{year}migrationdata/stateoutflow{year}.csv")

    # Filter "non state" rows (fips code < 59)
    flows_us = flows_us.loc[
        ((flows_us["y1_statefips"].astype(int) < 59) & (flows_us["y2_statefips"].astype(int) < 59)),:]

    # Filter out sci within the same country
    flows_us = flows_us.loc[
        flows_us["y1_statefips"] != flows_us["y2_statefips"],:
        ]
    flows_us.reset_index(inplace=True)
    flows_us.rename(columns={"n2":"num_migrants"}, inplace=True)
    flows_us.head(5)
    return (flows_us,)


@app.cell
def _():
    return


@app.cell
def _(res_ferm_us):
    res_ferm_us
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run models
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    FERM,
    NUM_PARTICLES,
    RM,
    SIGMA,
    coefficient_of_determination,
    df_feat_all_us,
    df_us,
    flows_us,
    mode,
    nodes_us,
):
    # GDP + SCI
    _ferm = FERM(
        nodes_us,
        flows_us,
        features = df_feat_all_us,
        node_col="statefp",
         )

    res_ferm_us = _ferm.run(
        num_particles = NUM_PARTICLES,
        sigma = SIGMA, 
        origin_col="y1_statefips",
        dest_col="y2_statefips",
        flow_col="num_migrants",
        verbose = False
    ).comparison

    # GDP
    _ferm = FERM(
        nodes_us,
        flows_us,
        features = df_us,
        node_col="statefp",
         )

    res_ferm_us_GDP = _ferm.run(
        num_particles = NUM_PARTICLES,
        sigma = 15.0, 
        origin_col="y1_statefips",
        dest_col="y2_statefips",
        flow_col="num_migrants",
        verbose = False
    ).comparison

    # Radiation model
    _rm = RM(
            nodes_us,
            flows_us,
            node_col="statefp",
            )

    res_rm_us = _rm.run(
        origin_col="y1_statefips",
        dest_col="y2_statefips",
        flow_col="num_migrants",
    ).comparison

    R2_ferm_us = coefficient_of_determination(res_ferm_us, mode=mode)
    R2_ferm_GDP_us = coefficient_of_determination(res_ferm_us_GDP, mode=mode)
    R2_rm_us = coefficient_of_determination(res_rm_us, mode=mode)
    R2s_us = [R2_rm_us, R2_ferm_GDP_us, R2_ferm_us]
    return R2s_us, res_ferm_us, res_ferm_us_GDP, res_rm_us


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot results US
    """)
    return


@app.cell
def _(
    FIGDIR,
    R2s_us,
    plot_scatter,
    plt,
    res_ferm_us,
    res_ferm_us_GDP,
    res_rm_us,
):
    # plot_scatter(res_rm_agg)
    _df_list, _labels = [res_rm_us, res_ferm_us_GDP, res_ferm_us], ["RM", "FERM GDP", "FERM GDP+SCI"]
    _fig, _axes = plt.subplots(1,3,figsize=(18,6))

    for _idx, _ax in enumerate(_axes):
        _ax = plot_scatter(df=_df_list[_idx], ax=_ax, label=_labels[_idx])
        _ax.text(x=0.1, y=0.9, s=f"R² = {R2s_us[_idx]:.3f}", transform=_ax.transAxes, fontsize=18)
        if _idx != 0:
            _ax.set_ylabel(" ")

    plt.savefig(FIGDIR / "scatter_plot_US.pdf", bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis US Counties
    """)
    return


@app.cell
def _(pd):
    gdp_df_counties = pd.read_csv("data/US_census_data/features/GDP/counties/gdp_per_capita_2018_counties.csv",
                                  dtype={'gdp_per_capita_2018': float}, na_values='(NA)')
    sci_df_counties = pd.read_csv("data/US_census_data/features/SCI/us_counties_SCI_2026_norm.csv")
    # sci_df_counties = pd.read_csv("data/US_census_data/features/SCI/us_counties_SCI_2026_minus_log.csv")
    sci_df_counties.head()
    return gdp_df_counties, sci_df_counties


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build relational features
    """)
    return


@app.cell
def _(np, pd):
    def load_relational_features_US_counties(
        df_rel: pd.DataFrame = None,
        df: pd.DataFrame = None,
        niche_name_nodes: str = "gdp_per_capita_2018",
        niche_name_relational: str = "scaled_sci_2026_norm",
        normalize: str = "log_zscore",
        fillna: bool = True,
    ) -> pd.DataFrame:

        # Transform county fips to string
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        df_rel["county_from"] = df_rel["county_from"].astype(str).str.zfill(5)
        df_rel["county_to"] = df_rel["county_to"].astype(str).str.zfill(5)
        # print(df_rel)

        #/ Node feature matrix
        df["log_gdp"] = np.log(df[niche_name_nodes])
        df["log_gdp_norm"] = (df["log_gdp"]- df["log_gdp"].mean())/df["log_gdp"].std()
        df = df.rename(columns={"county_fips":"county_fips_o"})
        df["county_fips_d"] = df["county_fips_o"]

        # Build node feature matrix
        df = df.pivot(index="county_fips_o", columns="county_fips_d", values="log_gdp_norm")
        df = df.ffill(axis=0).bfill(axis=0) # Fill rows
        # print(df)

        #/ Relational feature matrix
        dfr = df_rel.pivot(
            index="county_from",
            columns="county_to",
            values=niche_name_relational,
        )
        # print(dfr)

        if fillna:
            # print(1-(np.isnan(df.to_numpy()).sum()-235)/(235*234)) # Count nan
            dfr.fillna(dfr.mean().mean(), inplace=True)

        # Fill to zero the diagonal terms
        for label in dfr.index:
            dfr.loc[label, label] = 0
        # print(dfr)

        # Combine (add) node features with relational features
        common_cols = df.index.intersection(dfr.index)
        dfr = dfr.loc[common_cols, common_cols]
        df = df.loc[common_cols, common_cols]

        df_all = dfr + df

        # dfr = dfr.add(df.loc[common_cols, "log_gdp_norm"],axis="index")

        return df_all, df, dfr


    return (load_relational_features_US_counties,)


@app.cell
def _():
    return


@app.cell
def _(
    gdp_df_counties,
    load_relational_features_US_counties,
    pd,
    sci_df_counties,
):
    _year = "1516"  # 1415 default
    flows_us_c = pd.read_csv(f"./data/US_census_data/migrations/{_year}migrationdata/countyoutflow{_year}.csv")

    #\Filter "non state" rows (fips code <= 56). State code 57-59 are for other flows and foreign flows
    flows_us_c = flows_us_c.loc[
        ((flows_us_c["y1_statefips"].astype(int) <= 56) & (flows_us_c["y2_statefips"].astype(int) <=56)),:]

    flows_us_c.rename(columns={"n2":"num_migrants"}, inplace=True)

    #\ Create string fips columns for source and target 
    flows_us_c["y1_fips"] = (
        flows_us_c["y1_statefips"].astype(str).str.zfill(2) + flows_us_c["y1_countyfips"].astype(str).str.zfill(3)
    )

    flows_us_c["y2_fips"] = (
        flows_us_c["y2_statefips"].astype(str).str.zfill(2) + flows_us_c["y2_countyfips"].astype(str).str.zfill(3)
    )
    flows_us_c = flows_us_c.loc[:, ["y1_fips", "y2_fips", "num_migrants"]]

    #\ Filter out sci within the same country
    flows_us_c = flows_us_c.loc[
        flows_us_c["y1_fips"] != flows_us_c["y2_fips"],:
        ]
    #####################
    #####################
    #####################

    #\ Set Nan to zero (-1: "Suppressed data value" as for doc)
    print(f"Nan values: {sum(flows_us_c["num_migrants"] == -1)}")
    flows_us_c.loc[flows_us_c["num_migrants"] == -1, "num_migrants"] = 0  
    flows_us_c.reset_index(drop=True, inplace=True)
    flows_us_c.head(5)


    nodes_us_c = pd.read_csv("./data/US_census_data/center_of_population/CenPop2020_Mean_CO.txt")
    nodes_us_c["fips"] = (nodes_us_c["STATEFP"].astype(str).str.zfill(2) + nodes_us_c["COUNTYFP"].astype(str).str.zfill(3))
    nodes_us_c.drop(columns=["STATEFP","COUNTYFP"], inplace=True)
    nodes_us_c.rename(columns={"POPULATION":"population", "LATITUDE":"lat", "LONGITUDE":"lon"}, inplace=True)
    # nodes_us_c = nodes_us_c.set_index("fips")
    # nodes_us_c.head(5)

    # For testing
    # nodes_us_c = nodes_us_c.iloc[:750,:]

    nodes_us_c = nodes_us_c.set_index("fips")

    # Select just ones in flows_us_c
    intersection = (set(flows_us_c["y1_fips"]) | set(flows_us_c["y2_fips"])) &  set(nodes_us_c.index)

    nodes_us_c = nodes_us_c.loc[list(intersection),:]
    flows_us_c = flows_us_c.loc[flows_us_c["y1_fips"].isin(intersection) & flows_us_c["y2_fips"].isin(intersection),:]


    # Intersection with relational features
    df_all_us_c, df_us_c, dfr_us_c = load_relational_features_US_counties(
        df_rel=sci_df_counties,
        df=gdp_df_counties,
        fillna=False,
    )

    common_idx = nodes_us_c.index.intersection(df_all_us_c.index)
    nodes_us_c = nodes_us_c.loc[common_idx]
    df_all_us_c = df_all_us_c.loc[common_idx, common_idx]

    nodes_us_c.reset_index(drop=False, inplace=True, names="fips")
    return dfr_us_c, flows_us_c, nodes_us_c


@app.cell
def _():
    # plt.imshow(df_all_us_c.to_numpy()[0:30,:30])
    # plt.colorbar()
    # plt.show()

    # dfr_us_c
    # flows_us_c.reset_index(drop=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run models
    """)
    return


@app.cell
def _(FERM, NUM_PARTICLES, dfr_us_c, flows_us_c, nodes_us_c):
    VERBOSE = False
    SIGMA1 = 5
    # GDP + SCI
    # _ferm = FERM(
    #     nodes_us_c,
    #     flows_us_c,
    #     features = df_all_us_c,
    #     node_col="fips",
    #      )

    # res_ferm_us_c = _ferm.run(
    #     num_particles = NUM_PARTICLES,
    #     sigma = SIGMA1, 
    #     origin_col="y1_fips",
    #     dest_col="y2_fips",
    #     flow_col="num_migrants",
    #     verbose = VERBOSE
    # ).comparison

    # GDP
    _ferm = FERM(
        nodes_us_c,
        flows_us_c,
        features = dfr_us_c,
        node_col="fips",
         )

    res_ferm_us_GDP_c = _ferm.run(
        num_particles = NUM_PARTICLES,
        sigma = SIGMA1, 
        origin_col="y1_fips",
        dest_col="y2_fips",
        flow_col="num_migrants",
        verbose = VERBOSE
    ).comparison

    # Radiation model
    # _rm = RM(
    #         nodes_us_c,
    #         flows_us_c,
    #         node_col="fips",
    #         )

    # res_rm_us_c = _rm.run(
    #     origin_col="y1_fips",
    #     dest_col="y2_fips",
    #     flow_col="num_migrants",
    # ).comparison
    return (res_ferm_us_GDP_c,)


@app.cell
def _(coefficient_of_determination, mode, res_rm_us_c):
    ress = coefficient_of_determination(res_rm_us_c, mode=mode)
    print(ress)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # res_rm_us_c.loc[:,["y1_fips","y2_fips","num_migrants","predicted_migrants"]].to_csv(f"./results/predicted_flows+Radiation.csv",index=False)
    # res_ferm_us_GDP_c.loc[:,["y1_fips","y2_fips","num_migrants","predicted_migrants"]].to_csv(f"./results/predicted_flows+SCI+Sigma_{SIGMA1}+year_{year}.csv", index=False)
    return


@app.cell
def _():
    # _fig, _ax = plt.subplots()
    # plot_scatter(df=res_ferm_us_GDP_c, ax=_ax, label="test")
    return


@app.cell
def _(
    coefficient_of_determination,
    mode,
    res_ferm_us_GDP_c,
    res_ferm_us_c,
    res_rm_us_c,
):
    R2_ferm_us_c = coefficient_of_determination(res_ferm_us_c, mode=mode)
    R2_ferm_GDP_us_c = coefficient_of_determination(res_ferm_us_GDP_c, mode=mode)
    R2_rm_us_c = coefficient_of_determination(res_rm_us_c, mode=mode)
    R2s_us_c = [R2_rm_us_c, R2_ferm_us_c, R2_ferm_GDP_us_c]
    R2s_us_c
    return (R2s_us_c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot results counties
    """)
    return


@app.cell
def _(pd):
    res_rm_c = pd.read_csv("./results/predicted_flows+Radiation.csv")
    res_ferm_c = pd.read_csv(f"./results/predicted_flows+SCI+Sigma_5+year_1516.csv")
    return


@app.cell
def _():
    return


@app.cell
def _(FIGDIR, R2s_us_c, plot_scatter, plt, res_ferm_us_GDP_c, res_rm_us_c):
    # _df_list, _labels = [res_rm_us_c, res_ferm_us_c, res_ferm_us_GDP_c], ["RM", "FERM GDP+SCI", "FERM SCI"]
    _df_list, _labels = [res_rm_us_c, res_ferm_us_GDP_c], ["RM",  "FERM SCI"]
    _fig, _axes = plt.subplots(1,3,figsize=(18,6))

    for _idx, _ax in enumerate(_axes):
        _ax = plot_scatter(df=_df_list[_idx], ax=_ax, label=_labels[_idx])
        _ax.text(x=0.1, y=0.9, s=f"R² = {R2s_us_c[_idx]:.3f}", transform=_ax.transAxes, fontsize=18)
        _ax.set_xlim([1e1,1e5])
        _ax.set_ylim([0.5e0,1e5]) 
        if _idx != 0:
            _ax.set_ylabel(" ")

    plt.savefig(FIGDIR / "scatter_plot_US_counties.pdf", bbox_inches="tight")
    plt.show()

    return


@app.cell
def _():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    import matplotlib.colors as colors

    return ccrs, colors, shpreader


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
    ## Figure 1 paper
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot migrations and SCI
    """)
    return


@app.cell
def _(pd):
    sci_df = pd.read_csv("data/US_census_data/features/SCI/us_counties_SCI_2026.csv",
                         usecols=[2,3,4], dtype={"user_region":str, "friend_region":str, "scaled_sci":float})
    sci_df
    return (sci_df,)


@app.cell
def _(flows_us_c, sci_df):
    COUNTY_FIPS = "48479"
    df2 = flows_us_c.loc[
        (flows_us_c["y1_fips"] == COUNTY_FIPS) & 
        (flows_us_c["y2_fips"].map(lambda x: x[0:2]) == COUNTY_FIPS[0:2]),:
    ]
    df2["num_migrants_scaled"] = df2["num_migrants"]/df2["num_migrants"].max()
    df2

    # df2.loc[(df2.loc[:,"y1_fips"]==COUNTY_FIPS) & (df2.loc[:,"y2_fips"]=="48029"),
    #                     "num_migrants_scaled"
    #                     ]

    df = sci_df.loc[
        (sci_df["user_region"] == COUNTY_FIPS) & 
        (sci_df["friend_region"].map(lambda x: x[0:2]) == COUNTY_FIPS[0:2]),:
    ].reset_index(drop=True)
    df = df.set_index("friend_region")
    df
    return (COUNTY_FIPS,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(ccrs, colors, flows_us_c, plt, sci_df, shpreader):
    COUNTY_SHP = "./data/US_census_data/shapefile/cb_2019_us_all_500k/cb_2019_us_county_500k/cb_2019_us_county_500k.shp"

    def map_single_state(
        COUNTY_FIPS:str = "48453"
    ) -> plt.Axes:

        fips_state:str = COUNTY_FIPS[0:2]
    
        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": ccrs.LambertConformal()}
        )


        reader = shpreader.Reader(COUNTY_SHP)
        county_dct = {
            (item.attributes["STATEFP"]+item.attributes["COUNTYFP"]):item.geometry 
            for item in list(reader.records())
        }

        # Draw counties
        print("Draw counties")
        for item in reader.records():
        
            if fips_state == str(item.attributes["STATEFP"]):
                # print(fips)
                ax.add_geometries(
                    [item.geometry],
                    crs=ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor="k",
                    linewidth=0.4,
                )

        #/ Color SCI for a specific county
    
        # Create colormap for Social Connecdness Index
        cmap = plt.get_cmap("Blues")
        norm = colors.Normalize(
            vmin=0,
            vmax=0.5e5,
        )
    
        # Define dataframes
        df = sci_df.loc[
            (sci_df["user_region"] == COUNTY_FIPS) & 
            (sci_df["friend_region"].map(lambda x: x[0:2]) == COUNTY_FIPS[0:2]),:
        ].reset_index(drop=True)
        df = df.set_index("friend_region")

        df2 = flows_us_c.loc[
            (flows_us_c["y1_fips"] == COUNTY_FIPS) & 
            (flows_us_c["y2_fips"].map(lambda x: x[0:2]) == COUNTY_FIPS[0:2]),:
        ]
    
        df2["num_migrants_scaled"] = df2["num_migrants"]/df2["num_migrants"].max()*2.5
        # df2["num_migrants_scaled"] = np.log(df2["num_migrants"])

        # Select only counties for which we have migration flows
        # df = df.loc[df.index.intersection(df2["y2_fips"])]
    
        print("Adding SCI color")
        for item in reader.records():
            fips = str(item.attributes["STATEFP"]) + str(item.attributes["COUNTYFP"])
            if fips in df.index:

                sci = df.loc[fips,"scaled_sci"]
                ax.add_geometries(
                    [item.geometry],
                    crs=ccrs.PlateCarree(),
                    facecolor=cmap(norm(sci)),
                    edgecolor="k",
                    linewidth=0.4,
                )
            if fips == COUNTY_FIPS:
                ax.add_geometries(
                    [item.geometry],
                    crs=ccrs.PlateCarree(),
                    facecolor="r",
                    edgecolor="k",
                    linewidth=0.4,
                )


        print("Adding migration arrows")
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        lon1 = county_dct[COUNTY_FIPS].centroid.x
        lat1 = county_dct[COUNTY_FIPS].centroid.y
    
        for row in df2.iterrows():
            lon2 = county_dct[row[1].y2_fips].centroid.x
            lat2 = county_dct[row[1].y2_fips].centroid.y
            ax.annotate(
                "",
                xy=(lon2, lat2),
                xytext=(lon1, lat1),
                xycoords=transform,
                textcoords=transform,
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=df2.loc[ 
                        (df2.loc[:,"y1_fips"]==COUNTY_FIPS) & (df2.loc[:,"y2_fips"]==row[1].y2_fips),
                        "num_migrants_scaled"
                        ].item(),
                    connectionstyle="arc3,rad=0.2",
                ),
            )

        # Remove framing box
        ax.spines["geo"].set_visible(False)

        # Add SCI colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
        cbar = fig.colorbar(
            sm,
            ax=ax,
            shrink=0.8,
            pad=0.02,
        )
    
        cbar.set_label("Social Connectedness Index")
    
        return ax

    ax = map_single_state(COUNTY_FIPS="06037")
    # ax.set_extent([-107, -93, 25, 37]) # Texas (fips 48)
    ax.set_extent([-124.7, -114.0, 32.3, 42.1]) # California (fips 06)



    # Annotate Los Angeles 
    lon_la, lat_la = -118.2437, 34.0522
    _ax.plot(lon_la, lat_la, marker="o", color='k', markersize=5, transform=ccrs.PlateCarree())
    _ax.annotate(
        "Los Angeles",
        xy=(lon_la, lat_la),
        xytext=(-25, -20),
        textcoords="offset points",
        transform=ccrs.PlateCarree(),
        fontsize=10,
    )

    plt.show()
    # plt.savefig(FIGDIR / "paper/Figure 1/map1.pdf", bbox_inches="tight")
    

    
    return (COUNTY_SHP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot thresholds in counties
    """)
    return


@app.cell
def _(gaussian_max_sample_vec, np):
    thres = gaussian_max_sample_vec(mu=0, sigma=2.5,n=1_100_195,size=1)[0]
    int(np.round(thres,0))
    return


@app.cell
def _():
    return


@app.cell
def _(COUNTY_FIPS, sci_df_counties):
    sci_df_counties.loc[
        (sci_df_counties["county_from"] == COUNTY_FIPS) &
        (sci_df_counties["county_to"].map(lambda x:x[0:2]) == COUNTY_FIPS[0:2]),:
    ].set_index("county_from")
    return


@app.cell
def _(COUNTY_SHP, FIGDIR, ccrs, nodes_us_c, np, plt, shpreader):
    from src.ferm.model import gaussian_max_sample_vec

    def show_thresholds_single_state(
        COUNTY_FIPS:str = "06037"
    ) -> plt.Axes:

        fips_state:str = COUNTY_FIPS[0:2]
    
        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": ccrs.LambertConformal()}
        )

        reader = shpreader.Reader(COUNTY_SHP)
        county_dct = {
            (item.attributes["STATEFP"]+item.attributes["COUNTYFP"]):item.geometry 
            for item in list(reader.records())
        }

        # Draw counties
        print("Draw counties")
        for item in reader.records():
            fips = str(item.attributes["STATEFP"]) + str(item.attributes["COUNTYFP"])
            if fips_state == str(item.attributes["STATEFP"]):
                ax.add_geometries(
                    [item.geometry],
                    crs=ccrs.PlateCarree(),
                    facecolor='none',
                    edgecolor="k",
                    linewidth=0.4,
                )

            if fips == COUNTY_FIPS:
                ax.add_geometries(
                    [item.geometry],
                    crs=ccrs.PlateCarree(),
                    facecolor="r",
                    edgecolor="k",
                    linewidth=0.4,
                )

        # Define dataframes
        df = nodes_us_c.loc[
            nodes_us_c.loc[:,"fips"].map(lambda x:x[0:2]) == "06",:
        ].set_index("fips")

        # Put threshold in origin county
        thres_o = int(np.round(
            gaussian_max_sample_vec(
                mu=0,
                sigma=2.5,
                n=3_000_000, #df.loc[COUNTY_FIPS].population,
                size=1,
            )[0]
        ,0))
        ax.annotate(
            str(thres_o),
            xy=(county_dct[COUNTY_FIPS].centroid.x, county_dct[COUNTY_FIPS].centroid.y),
            xytext=(0, 0),
            textcoords="offset points",
            transform=ccrs.PlateCarree(),
            color = 'k',
            fontsize=10,
            ha="center",
            va="center",
        )
    
        for row in df.iterrows():
            if row[1].name != COUNTY_FIPS:
                thres = gaussian_max_sample_vec(
                    mu=0,
                    sigma=2.5,
                    n=row[1].population,
                    size=1,
                )[0]
                thres = int(np.round(thres,0)) # Round threshold
            
                lon = county_dct[row[1].name].centroid.x
                lat = county_dct[row[1].name].centroid.y        
                ax.annotate(
                    str(thres),
                    xy=(lon,lat),
                    xytext=(0, 0),
                    textcoords="offset points",
                    transform=ccrs.PlateCarree(),
                    color = 'g' if thres >= thres_o else 'r',
                    fontsize=10,
                    ha="center",
                    va="center",
                )


        # Remove framing box
        ax.spines["geo"].set_visible(False)
    
        return ax


    _ax = show_thresholds_single_state(COUNTY_FIPS="06037")
    _ax.set_extent([-124.7, -114.0, 32.3, 42.1]) # California (fips 06)

    # plt.show()
    plt.savefig(FIGDIR / "paper/Figure 1/thesholds.pdf", bbox_inches="tight")
    return (gaussian_max_sample_vec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Others
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hierarchical clustering SCI in US counties
    """)
    return


@app.cell
def _(ccrs, np, pd, plt, shpreader):
    def plot_us_counties_clusters(county_clusters:pd.DataFrame) -> plt.Axes:
        '''
        county_clusters
            dataframe containing the cluster identifiers for the US counties
            from agglomerative clustering
        '''

        # Define colors
        colors = np.vstack([
            plt.get_cmap("tab20").colors,
            plt.get_cmap("tab20b").colors,
            plt.get_cmap("tab20c").colors,
        ])
        cluster_colors = colors[:50]

        # Plotting
        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": ccrs.LambertConformal()}
        )
        ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())

        COUNTY_SHP = "./data/US_census_data/shapefile/cb_2019_us_all_500k/cb_2019_us_county_500k/cb_2019_us_county_500k.shp"
        reader = shpreader.Reader(COUNTY_SHP)

        # Draw counties
        for item in reader.records():
    
            fips = str(item.attributes["STATEFP"]) + str(item.attributes["COUNTYFP"])

            try:
                cluster = county_clusters.loc[fips]
            except KeyError:
                cluster = None

            if cluster is None:
                facecolor = "lightgray"
            else:
                facecolor = cluster_colors[cluster]

            ax.add_geometries(
                [item.geometry],
                crs=ccrs.PlateCarree(),
                facecolor=facecolor,
                edgecolor="white",
                linewidth=0.1,
            )

        # Draw states
        STATE_SHP = "./data/US_census_data/shapefile/cb_2019_us_all_500k/cb_2019_us_state_500k/cb_2019_us_state_500k.shp"
        reader = shpreader.Reader(STATE_SHP)
    
        for item in reader.records():
            ax.add_geometries(
                [item.geometry],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="black",
                linewidth=0.3,
            )

        # Remove framing box
        ax.spines["geo"].set_visible(False)

        return ax

    return (plot_us_counties_clusters,)


@app.cell
def _(pd):
    # https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
    # from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering 

    sci_df_counties_orig = pd.read_csv("data/US_census_data/features/SCI/us_counties_SCI_2026.csv", usecols=[2,3,4])
    sci_df_counties_orig.rename(columns={"user_region":"county_from", "friend_region":"county_to"}, inplace=True)
    sci_df_counties_orig.head()
    return AgglomerativeClustering, sci_df_counties_orig


@app.cell
def _(np, sci_df_counties_orig):
    _method = "mlog" # "maxmlog" "mlog"
    S = sci_df_counties_orig.pivot(index="county_from", columns="county_to", values="scaled_sci").astype(float)

    if _method == "maxmlog":
        S_log = np.log(S)
        S_log = S_log.to_numpy(copy=True)
        np.fill_diagonal(S_log,0)
        max_log = np.max(S_log)

    elif _method == "mlog":
        S_log = np.log(S)
        S_log = S_log.to_numpy(copy=True)
        np.fill_diagonal(S_log,0)
        S.iloc[:,:] = - S_log
    return (S,)


@app.cell
def _(AgglomerativeClustering, S, pd):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=50, linkage='average', metric="precomputed")
    labels = hierarchical_cluster.fit( S)
    county_clusters = pd.DataFrame(index=S.index, data=labels.labels_)
    county_clusters.index = county_clusters.index.astype(str).str.zfill(5)
    county_clusters
    return (county_clusters,)


@app.cell
def _(FIGDIR, county_clusters, plot_us_counties_clusters, plt):
    plot_us_counties_clusters(county_clusters)
    plt.savefig(FIGDIR / "hierarchical_clustering_US_counties_50_cluster_method_mlog.png")

    return


@app.cell
def _(np, pd):
    df3 = pd.read_csv("./data/US_census_data/features/SCI/us_counties_SCI_2026.csv")
    df3.head()

    df3["scaled_sci_norm"] = -np.log(df3["scaled_sci"])
    df3.rename(columns={"user_region": "county_from", "friend_region":"county_to","scaled_sci_norm":"scaled_sci_2026_norm"}, inplace=True)
    #df3
    #df3[["county_from","county_to","scaled_sci_2026_norm"]].to_csv("./data/US_census_data/features/SCI/us_counties_SCI_2026_minus_log.csv", index=False, float_format='%.5f')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
