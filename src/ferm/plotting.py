#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

EPS: float = 1.0

def summarize_comparison(comp, label):
    return {
        "period": label, "pairs": len(comp),
        "observed_total": comp["total_migrants"].sum(), "predicted_total": comp["predicted_migrants"].sum(),
        "MAE": np.mean(np.abs(comp["predicted_migrants"] - comp["total_migrants"])),
        "RMSE": np.sqrt(np.mean((comp["predicted_migrants"] - comp["total_migrants"])**2)),
        "Bias_pred_minus_obs": np.mean(comp["predicted_migrants"] - comp["total_migrants"]),
        "Pearson": comp["total_migrants"].corr(comp["predicted_migrants"], method="pearson"),
        "Spearman": comp["total_migrants"].corr(comp["predicted_migrants"], method="spearman"),
        "Pearson_log": np.log10(comp["total_migrants"] + 1).corr(np.log10(comp["predicted_migrants"] + 1), method="pearson"),
        "Spearman_log": np.log10(comp["total_migrants"] + 1).corr(np.log10(comp["predicted_migrants"] + 1), method="spearman"),
        "Median_abs_log_ratio": np.median(np.abs(np.log10((comp["predicted_migrants"] + 1) / (comp["total_migrants"] + 1))))
    }

def plot_timeseries_migrants(df):
    tmp = df.copy()
    monthly_totals = (
        tmp.dropna(subset=["migration_month"])
        .groupby(pd.to_datetime(tmp["migration_month"]).dt.to_period("M").dt.to_timestamp(), as_index=False)["num_migrants"]
        .sum().rename(columns={"migration_month": "month_ts"}).sort_values("month_ts")
    )
    plt.figure(figsize=(11, 5))
    plt.plot(monthly_totals["month_ts"], monthly_totals["num_migrants"], marker="o", linewidth=1.8)
    covid_date = pd.Timestamp("2020-03-01")
    plt.axvline(covid_date, color="red", linestyle="--", linewidth=1.8)
    plt.text(
        covid_date,
        monthly_totals["num_migrants"].max() * 0.95,
        "covid hit",
        color="red",
        rotation=90,
        va="top",
        ha="right"
    )
    war_date = pd.Timestamp("2022-02-24")
    plt.axvline(war_date, color="orange", linestyle="--", linewidth=1.8)
    plt.text(
        war_date,
        monthly_totals["num_migrants"].max() * 0.95,
        "war in Ukraine",
        color="orange",
        rotation=90,
        va="top",
        ha="right"
    )
    plt.title("Total migrants per month")
    plt.xlabel("Month")
    plt.ylabel("Total migrants")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
   
def prepare_comp_df(comp):
    out = comp.copy()
    if "residual" not in out.columns:
        out["residual"] = out["predicted_migrants"] - out["total_migrants"]
    if "abs_error" not in out.columns:
        out["abs_error"] = np.abs(out["residual"])
    if "log_ratio" not in out.columns:
        out["log_ratio"] = np.log10((out["predicted_migrants"] + EPS) / (out["total_migrants"] + EPS))
    if "abs_log_ratio" not in out.columns:
        out["abs_log_ratio"] = np.abs(out["log_ratio"])
    return out

def compare_rm_vs_ferm_routes(comp_rm, comp_ferm):
    rm = prepare_comp_df(comp_rm).copy()
    ferm = prepare_comp_df(comp_ferm).copy()
    keep_cols = ["country_from", "country_to", "country_from_name", "country_to_name", "total_migrants", "predicted_migrants", "residual", "abs_error", "log_ratio", "abs_log_ratio"]
    rm = rm[keep_cols].rename(columns={"predicted_migrants": "predicted_rm", "residual": "residual_rm", "abs_error": "abs_error_rm", "log_ratio": "log_ratio_rm", "abs_log_ratio": "abs_log_ratio_rm"})
    ferm = ferm[keep_cols].rename(columns={"predicted_migrants": "predicted_ferm", "residual": "residual_ferm", "abs_error": "abs_error_ferm", "log_ratio": "log_ratio_ferm", "abs_log_ratio": "abs_log_ratio_ferm"})
    merged = rm.merge(ferm[["country_from", "country_to", "predicted_ferm", "residual_ferm", "abs_error_ferm", "log_ratio_ferm", "abs_log_ratio_ferm"]], on=["country_from", "country_to"], how="inner")
    merged["improvement_abs_error"] = merged["abs_error_rm"] - merged["abs_error_ferm"]
    merged["improvement_abs_log"] = merged["abs_log_ratio_rm"] - merged["abs_log_ratio_ferm"]
    return merged

def add_coords_to_routes(df, country_geo):
    geo = country_geo[["code", "lat", "lon"]].drop_duplicates().copy()

    df = df.merge(
        geo.rename(columns={"code": "country_from", "lat": "lat_from", "lon": "lon_from"}),
        on="country_from", how="left"
    )
    df = df.merge(
        geo.rename(columns={"code": "country_to", "lat": "lat_to", "lon": "lon_to"}),
        on="country_to", how="left"
    )
    return df




def plot_rm_vs_ferm_error_scatter(comp_rm, comp_ferm, label="all", metric="abs_log", niche_type="gdp_per_capita_2018", continent=None):
    df = compare_rm_vs_ferm_routes(comp_rm, comp_ferm).copy()
    if metric == "abs_error":
        xcol, ycol, xlab, ylab = "abs_error_rm", "abs_error_ferm", "RM absolute error", "FERM absolute error"
    elif metric == "abs_log":
        xcol, ycol, xlab, ylab = "abs_log_ratio_rm", "abs_log_ratio_ferm", "RM absolute log-ratio error", "FERM absolute log-ratio error"
    else:
        raise ValueError("metric must be 'abs_error' or 'abs_log'")
    plt.figure(figsize=(7, 7))
    plt.scatter(df[xcol], df[ycol], alpha=0.4)
    lim = max(df[xcol].max(), df[ycol].max())
    plt.plot([0, lim], [0, lim], linestyle="--")
    share_better = (df[ycol] <= df[xcol]).mean() # share of points below the diagonal
    median_improvement = np.median(df[xcol] - df[ycol])
    text_str = (
        f"FERM better share: {share_better:.3f}\n"
        f"Median improvement ({metric}): {median_improvement:.4f}"
    )
    ax = plt.gca()
    ax.text(
        0.98, 0.02, text_str,
        transform=ax.transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"RM vs FERM route-level error — {(continent + ' ' if continent else '')} {label} ({niche_type})")
    plt.tight_layout()
    plt.show()

    return df

def plot_error_cdf_rm_vs_ferm(comp_rm, comp_ferm, label="all", metric="abs_log", niche_type="gdp_per_capita_2018", continent = None):
    df = compare_rm_vs_ferm_routes(comp_rm, comp_ferm).copy()
    if metric == "abs_error":
        rm = np.sort(df["abs_error_rm"].values)
        ferm = np.sort(df["abs_error_ferm"].values)
        xlabel = "Absolute error"
    elif metric == "abs_log":
        rm = np.sort(df["abs_log_ratio_rm"].values)
        ferm = np.sort(df["abs_log_ratio_ferm"].values)
        xlabel = "Absolute log-ratio error"
    else:
        raise ValueError("metric must be 'abs_error' or 'abs_log'")
    y_rm = np.arange(1, len(rm) + 1) / len(rm)
    y_ferm = np.arange(1, len(ferm) + 1) / len(ferm)
    plt.figure(figsize=(7, 5))
    plt.plot(rm, y_rm, label="RM")
    plt.plot(ferm, y_ferm, label="FERM")
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative share of OD pairs")
    plt.title(f"Distribution of route-level errors — {(continent + ' ' if continent else '')} {label} ({niche_type})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return df

def plot_ferm_vs_rm_route_change_ex(comp_rm, comp_ferm, country_geo, continent_gdf=None, top_n_improve=15, top_n_worsen=15, metric="abs_error", label="all", xlim=None, ylim=None):
    df = compare_rm_vs_ferm_routes(comp_rm, comp_ferm).copy()
    df = add_coords_to_routes(df, country_geo)

    if metric == "abs_error":
        score_col = "improvement_abs_error"
        title_metric = "|residual|"
    elif metric == "abs_log":
        score_col = "improvement_abs_log"
        title_metric = "|log-ratio error|"
    else:
        raise ValueError("metric must be 'abs_error' or 'abs_log'")

    improved = df[df[score_col] > 0].nlargest(top_n_improve, score_col).copy()
    worsened = df[df[score_col] < 0].nsmallest(top_n_worsen, score_col).copy()

    improved["route_change"] = "Improved"
    worsened["route_change"] = "Worsened"

    plot_df = pd.concat([improved, worsened], ignore_index=True)
    plot_df = plot_df.dropna(subset=["lat_from", "lon_from", "lat_to", "lon_to"]).copy()

    if len(plot_df) == 0:
        print("No routes to plot.")
        return plot_df

    fig, ax = plt.subplots(figsize=(12, 10))

    if continent_gdf is not None:
        continent_gdf.boundary.plot(ax=ax, linewidth=0.8, color="black")

    max_scale = np.max(np.abs(plot_df[score_col].values))

    for _, row in plot_df.iterrows():
        x1, y1 = row["lon_from"], row["lat_from"]
        x2, y2 = row["lon_to"], row["lat_to"]

        strength = abs(row[score_col])
        lw = 1 + 5 * (strength / max_scale)

        if row[score_col] > 0:
            color = "tab:green"
            rad = 0.15
        else:
            color = "tab:orange"
            rad = -0.15

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            alpha=0.8,
            connectionstyle=f"arc3,rad={rad}",
            clip_on=True
        )
        ax.add_patch(arrow)
        arrow.set_clip_path(ax.patch)

        ax.scatter([x1], [y1], s=14, color="black", zorder=3)
        ax.scatter([x2], [y2], s=14, color="black", zorder=3)

    legend_elements = [
        Line2D([0], [0], color="tab:green", lw=2, label="FERM improves over RM"),
        Line2D([0], [0], color="tab:orange", lw=2, label="FERM worse than RM"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    ax.set_title(f"FERM vs RM route changes ({title_metric}) — {label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return plot_df[[
        "country_from", "country_to",
        "country_from_name", "country_to_name",
        "total_migrants",
        "predicted_rm", "predicted_ferm",
        "residual_rm", "residual_ferm",
        "improvement_abs_error", "improvement_abs_log",
        "route_change"
    ]]

def plot_ferm_vs_rm_route_change_err(
    comp_rm,
    comp_ferm,
    country_geo,
    continent_gdf=None,
    top_n_improve=15,
    top_n_worsen=15,
    metric="abs_error",
    label="all",
    xlim=None,
    ylim=None
):
    df = compare_rm_vs_ferm_routes(comp_rm, comp_ferm).copy()
    df = add_coords_to_routes(df, country_geo)

    if metric == "abs_error":
        score_col = "improvement_abs_error"
        title_metric = "|residual|"
    elif metric == "abs_log":
        score_col = "improvement_abs_log"
        title_metric = "|log-ratio error|"
    else:
        raise ValueError("metric must be 'abs_error' or 'abs_log'")

    improved = df[df[score_col] > 0].nlargest(top_n_improve, score_col).copy()
    worsened = df[df[score_col] < 0].nsmallest(top_n_worsen, score_col).copy()

    improved["route_change"] = "Improved"
    worsened["route_change"] = "Worsened"

    plot_df = pd.concat([improved, worsened], ignore_index=True)
    plot_df = plot_df.dropna(subset=["lat_from", "lon_from", "lat_to", "lon_to"]).copy()

    if len(plot_df) == 0:
        print("No routes to plot.")
        return plot_df

    def classify_row(row):
        if row[score_col] > 0 and row["residual_ferm"] > 0:
            return "Better / Overestimated"
        elif row[score_col] > 0 and row["residual_ferm"] < 0:
            return "Better / Underestimated"
        elif row[score_col] < 0 and row["residual_ferm"] > 0:
            return "Worse / Overestimated"
        elif row[score_col] < 0 and row["residual_ferm"] < 0:
            return "Worse / Underestimated"
        else:
            return "Zero residual"

    plot_df["route_class"] = plot_df.apply(classify_row, axis=1)

    color_map = {
        "Better / Overestimated": "tab:blue",
        "Better / Underestimated": "tab:green",
        "Worse / Overestimated": "tab:red",
        "Worse / Underestimated": "tab:orange",
        "Zero residual": "gray",
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    if continent_gdf is not None:
        continent_gdf.boundary.plot(ax=ax, linewidth=0.8, color="black")

    max_scale = np.max(np.abs(plot_df[score_col].values))

    for _, row in plot_df.iterrows():
        x1, y1 = row["lon_from"], row["lat_from"]
        x2, y2 = row["lon_to"], row["lat_to"]

        strength = abs(row[score_col])
        lw = 1 + 5 * (strength / max_scale) if max_scale != 0 else 1.5

        color = color_map[row["route_class"]]

        if "Better" in row["route_class"]:
            rad = 0.15
        else:
            rad = -0.15

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            alpha=0.8,
            connectionstyle=f"arc3,rad={rad}",
            clip_on=True
        )
        ax.add_patch(arrow)
        arrow.set_clip_path(ax.patch)

        ax.scatter([x1], [y1], s=14, color="black", zorder=3)
        ax.scatter([x2], [y2], s=14, color="black", zorder=3)

    legend_elements = [
        Line2D([0], [0], color="tab:blue", lw=2, label="Better than RM / FERM overestimates"),
        Line2D([0], [0], color="tab:green", lw=2, label="Better than RM / FERM underestimates"),
        Line2D([0], [0], color="tab:red", lw=2, label="Worse than RM / FERM overestimates"),
        Line2D([0], [0], color="tab:orange", lw=2, label="Worse than RM / FERM underestimates"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    ax.set_title(f"FERM vs RM route changes ({title_metric}) — {label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return plot_df[[
        "country_from", "country_to",
        "country_from_name", "country_to_name",
        "total_migrants",
        "predicted_rm", "predicted_ferm",
        "residual_rm", "residual_ferm",
        "improvement_abs_error", "improvement_abs_log",
        "route_change", "route_class"
    ]]

def plot_ferm_vs_rm_route_change(
    comp_rm,
    comp_ferm,
    country_geo,
    continent_gdf=None,
    top_n_improve=15,
    top_n_worsen=15,
    metric="abs_error",
    label="all",
    xlim=None,
    ylim=None
):
    df = compare_rm_vs_ferm_routes(comp_rm, comp_ferm).copy()
    df = add_coords_to_routes(df, country_geo)

    if metric == "abs_error":
        score_col = "improvement_abs_error"
        title_metric = "|residual|"
    elif metric == "abs_log":
        score_col = "improvement_abs_log"
        title_metric = "|log-ratio error|"
    else:
        raise ValueError("metric must be 'abs_error' or 'abs_log'")

    improved = df[df[score_col] > 0].nlargest(top_n_improve, score_col).copy()
    worsened = df[df[score_col] < 0].nsmallest(top_n_worsen, score_col).copy()

    improved["route_change"] = "Improved"
    worsened["route_change"] = "Worsened"

    plot_df = pd.concat([improved, worsened], ignore_index=True)
    plot_df = plot_df.dropna(subset=["lat_from", "lon_from", "lat_to", "lon_to"]).copy()

    if len(plot_df) == 0:
        print("No routes to plot.")
        return plot_df

    def classify_row(row):
        if row[score_col] > 0:
            # Better than RM -> classify using RM sign
            if row["residual_rm"] < 0:
                return "Better / RM underestimates"
            elif row["residual_rm"] > 0:
                return "Better / RM overestimates"
            else:
                return "Better / RM exact"
        elif row[score_col] < 0:
            # Worse than RM -> classify using FERM sign
            if row["residual_ferm"] > 0:
                return "Worse / FERM overestimates"
            elif row["residual_ferm"] < 0:
                return "Worse / FERM underestimates"
            else:
                return "Worse / FERM exact"
        else:
            return "No change"

    plot_df["route_class"] = plot_df.apply(classify_row, axis=1)

    color_map = {
        "Better / RM underestimates": "tab:blue",
        "Better / RM overestimates": "tab:green",
        "Worse / FERM overestimates": "tab:red",
        "Worse / FERM underestimates": "tab:orange",
        "Better / RM exact": "tab:blue",
        "Worse / FERM exact": "tab:red",
        "No change": "gray",
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    if continent_gdf is not None:
        continent_gdf.boundary.plot(ax=ax, linewidth=0.8, color="black")

    max_scale = np.max(np.abs(plot_df[score_col].values))
    if max_scale == 0:
        max_scale = 1.0

    for _, row in plot_df.iterrows():
        x1, y1 = row["lon_from"], row["lat_from"]
        x2, y2 = row["lon_to"], row["lat_to"]

        strength = abs(row[score_col])
        lw = 1 + 5 * (strength / max_scale)

        color = color_map[row["route_class"]]
        rad = 0.15 if row[score_col] > 0 else -0.15

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            alpha=0.8,
            connectionstyle=f"arc3,rad={rad}",
            clip_on=True
        )
        ax.add_patch(arrow)
        arrow.set_clip_path(ax.patch)

        ax.scatter([x1], [y1], s=14, color="black", zorder=3)
        ax.scatter([x2], [y2], s=14, color="black", zorder=3)

    legend_elements = [
        Line2D([0], [0], color="tab:blue", lw=2, label="Better than RM / RM underestimates"),
        Line2D([0], [0], color="tab:green", lw=2, label="Better than RM / RM overestimates"),
        Line2D([0], [0], color="tab:red", lw=2, label="Worse than RM / FERM overestimates"),
        Line2D([0], [0], color="tab:orange", lw=2, label="Worse than RM / FERM underestimates"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    ax.set_title(f"FERM vs RM route changes ({title_metric}) — {label}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    return plot_df[[
        "country_from", "country_to",
        "country_from_name", "country_to_name",
        "total_migrants",
        "predicted_rm", "predicted_ferm",
        "residual_rm", "residual_ferm",
        "improvement_abs_error", "improvement_abs_log",
        "route_change", "route_class"
    ]]