#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


SUMMER_MONTHS = [6, 7, 8]
COVID_START = pd.Timestamp("2020-03-01")
WAR_START = pd.Timestamp("2022-03-01")



#------------------------------------------------------------------------------

def filter_flows_by_continent(master_country_df, df, niche_type, continent=None):
    country_df = master_country_df[["code", "iso3", "country_name", "continent", "lat", "lon", "population", niche_type]].drop_duplicates("code").copy()

    if continent is None:
        df_model = df.copy()
    elif continent == "Americas":
        allowed_codes = set(country_df[country_df["continent"].isin(["North America", "South America", "Central America"])].values.flatten())
        df_model = df[df["country_from"].isin(allowed_codes) & df["country_to"].isin(allowed_codes)].copy()
    else:
        allowed_codes = set(country_df.loc[country_df["continent"] == continent, "code"])
        df_model = df[df["country_from"].isin(allowed_codes) & df["country_to"].isin(allowed_codes)].copy()

    return df_model, country_df

def split_flows_by_period(df_model, master_country_df):

    df_precovid = df_model[df_model["migration_month"] < COVID_START].copy()
    df_postcovid = df_model[df_model["migration_month"] >= COVID_START].copy()
    df_prewar = df_model[df_model["migration_month"] < WAR_START].copy()
    df_postwar = df_model[df_model["migration_month"] >= WAR_START].copy()
    df_summer = df_model[df_model["month"].isin(SUMMER_MONTHS)].copy()
    df_nonsummer = df_model[~df_model["month"].isin(SUMMER_MONTHS)].copy()
    df_summer_precovid = df_precovid[df_precovid["month"].isin(SUMMER_MONTHS)].copy()
    df_nonsummer_precovid = df_precovid[~df_precovid["month"].isin(SUMMER_MONTHS)].copy()

    # Holdout split: tune on the first semester of 2019, report on the second.
    df_validation_2019_h1 = df_model[
        (df_model["migration_month"] >= pd.Timestamp("2019-01-01"))
        & (df_model["migration_month"] < pd.Timestamp("2019-07-01"))
    ].copy()
    df_test_2019_h2 = df_model[
        (df_model["migration_month"] >= pd.Timestamp("2019-07-01"))
        & (df_model["migration_month"] < pd.Timestamp("2020-01-01"))
    ].copy()

    pairs_all = harmonize_flow_country_names(aggregate_pairs(df_model), master_country_df)
    pairs_precovid = harmonize_flow_country_names(aggregate_pairs(df_precovid), master_country_df)
    pairs_postcovid = harmonize_flow_country_names(aggregate_pairs(df_postcovid), master_country_df)
    pairs_prewar = harmonize_flow_country_names(aggregate_pairs(df_prewar), master_country_df)
    pairs_postwar = harmonize_flow_country_names(aggregate_pairs(df_postwar), master_country_df)
    pairs_summer = harmonize_flow_country_names(aggregate_pairs(df_summer), master_country_df)
    pairs_nonsummer = harmonize_flow_country_names(aggregate_pairs(df_nonsummer), master_country_df)
    pairs_summer_precovid = harmonize_flow_country_names(aggregate_pairs(df_summer_precovid), master_country_df)
    pairs_nonsummer_precovid = harmonize_flow_country_names(aggregate_pairs(df_nonsummer_precovid), master_country_df)
    pairs_validation_2019_h1 = harmonize_flow_country_names(aggregate_pairs(df_validation_2019_h1), master_country_df)
    pairs_test_2019_h2 = harmonize_flow_country_names(aggregate_pairs(df_test_2019_h2), master_country_df)

    pair_lookup = {
        "2019-2022": pairs_all,
        "precovid": pairs_precovid,
        "postcovid": pairs_postcovid,
        "prewar": pairs_prewar,
        "postwar": pairs_postwar,
        "summer": pairs_summer,
        "non_summer": pairs_nonsummer,
        "summer_precovid": pairs_summer_precovid,
        "nonsummer_precovid": pairs_nonsummer_precovid,
        "validation_2019_h1": pairs_validation_2019_h1,
        "test_2019_h2": pairs_test_2019_h2,
    }

    return pair_lookup



def harmonize_flow_country_names(df, master_country_df):
    code_to_name = master_country_df[["code", "country_name"]].drop_duplicates("code").set_index("code")["country_name"].to_dict()
    out = df.copy()
    out["country_from_name"] = out["country_from"].map(code_to_name).fillna(out["country_from"].map(code_to_country))
    out["country_to_name"] = out["country_to"].map(code_to_name).fillna(out["country_to"].map(code_to_country))
    return out

def aggregate_pairs(df, flow_col="num_migrants"):
    grouped = df.groupby(["country_from", "country_to"], as_index=False)[flow_col].sum().rename(columns={flow_col: "total_migrants"})
    grouped["total_migrants"] = pd.to_numeric(grouped["total_migrants"], errors="coerce").fillna(0.0)
    return grouped[grouped["country_from"] != grouped["country_to"]].copy()

def code_to_country(code):
    return {"XK": "Kosovo", "TW": "Taiwan"}.get(code, code)






