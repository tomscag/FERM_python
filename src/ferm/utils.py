import numpy as np
import pandas as pd
import geopandas as gpd
import pycountry

from scipy import stats

NE_URL = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

#------------------------------------------------------------------------------

def ensure_columns(df, required, df_name="DataFrame"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def code_to_country(code):
    return {"XK": "Kosovo", "TW": "Taiwan"}.get(code, code)


def load_migration_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find migration file at {path}.")
    df = pd.read_csv(path)
    ensure_columns(df, ["country_from", "country_to", "num_migrants"], "migration data")
    for c in ["country_from", "country_to"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["num_migrants"] = pd.to_numeric(df["num_migrants"], errors="coerce").fillna(0.0)
    if "migration_month" in df.columns:
        df["migration_month"] = pd.to_datetime(df["migration_month"], errors="coerce")
        df["year"] = df["migration_month"].dt.year
        df["month"] = df["migration_month"].dt.month
    elif {"year", "month"}.issubset(df.columns):
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df["migration_month"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    else:
        raise ValueError("Migration data must contain 'migration_month' or both 'year' and 'month'.")
    return df[df["country_from"] != df["country_to"]].copy()


def load_population_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find population file at {path}.")
    wb = pd.read_csv(path, skiprows=4)
    cols_needed = ["Country Name", "Country Code", "2019", "2020", "2021", "2022"]
    ensure_columns(wb, cols_needed, "population data")
    wb = wb[cols_needed].copy()
    for c in ["2019", "2020", "2021", "2022"]:
        wb[c] = pd.to_numeric(wb[c], errors="coerce")
    wb["population"] = wb[["2019", "2020", "2021", "2022"]].mean(axis=1)
    pop = wb.rename(columns={"Country Name": "country_name_wb", "Country Code": "iso3"})[["country_name_wb", "iso3", "population"]]
    pop["iso3"] = pop["iso3"].astype(str).str.strip().str.upper()
    return pop.dropna(subset=["iso3", "population"]).drop_duplicates(subset="iso3")


def load_country_geometries_global():    

    world = gpd.read_file(NE_URL)[
        ["ADMIN", "CONTINENT", "ADM0_A3", "ISO_A2_EH", "ISO_A3_EH", "geometry"]
    ].copy()

    rows = []
    for _, row in world.iterrows():
        code = str(row["ISO_A2_EH"]).strip().upper()
        iso3 = str(row["ISO_A3_EH"]).strip().upper()
        country_name = str(row["ADMIN"]).strip()
        continent = str(row["CONTINENT"]).strip()

        rp = row["geometry"].representative_point()

        rows.append({
            "code": code,
            "iso3": iso3,
            "adm0_a3": row["ADM0_A3"],
            "country_name": country_name,
            "continent": continent,
            "lat": float(rp.y),
            "lon": float(rp.x),
            "coord_source": "representative_point"
        })

    country_geo = pd.DataFrame(rows)

    manual = pd.DataFrame([
        {
            "code": "XK",
            "iso3": "XKX",
            "adm0_a3": "XKX",
            "country_name": "Kosovo",
            "continent": "Europe",
            "lat": 42.6026,
            "lon": 20.9030,
            "coord_source": "manual_representative_point"
        },
        {
            "code": "TW",
            "iso3": "TWN",
            "adm0_a3": "TWN",
            "country_name": "Taiwan",
            "continent": "Asia",
            "lat": 23.6978,
            "lon": 120.9605,
            "coord_source": "manual_representative_point"
        },
    ])

    country_geo = pd.concat([country_geo, manual], ignore_index=True)
    country_geo = country_geo.drop_duplicates(subset="code", keep="last").copy()

    return world, country_geo


def country_to_iso3(name):
    if pd.isna(name):
        return pd.NA

    name = str(name).strip()

    # manual fixes for names that often fail
    manual = {
        "Bolivia (Plurinational State of)": "BOL",
        "Congo": "COG",
        "Democratic Republic of the Congo": "COD",
        "Czechia": "CZE",
        "Iran (Islamic Republic of)": "IRN",
        "Republic of Korea": "KOR",
        "Democratic People's Republic of Korea": "PRK",
        "Lao People's Democratic Republic": "LAO",
        "Micronesia (Federated States of)": "FSM",
        "Moldova (Republic of)": "MDA",
        "Palestine": "PSE",
        "Russian Federation": "RUS",
        "State of Palestine": "PSE",
        "Syrian Arab Republic": "SYR",
        "Türkiye": "TUR",
        "United Kingdom of Great Britain and Northern Ireland": "GBR",
        "United Republic of Tanzania": "TZA",
        "United States of America": "USA",
        "Venezuela (Bolivarian Republic of)": "VEN",
        "Viet Nam": "VNM",
    }

    if name in manual:
        return manual[name]

    try:
        country = pycountry.countries.lookup(name)
        return country.alpha_3
    except LookupError:
        return pd.NA

def load_gdp_per_capita_2018(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find GDP-per-capita file at {path}.")
    wb = pd.read_csv(path, skiprows=4)
    cols_needed = ["Country Name", "Country Code", "2018"]
    ensure_columns(wb, cols_needed, "GDP per capita data")
    gdp = wb[cols_needed].copy()
    gdp["2018"] = pd.to_numeric(gdp["2018"], errors="coerce")
    gdp = gdp.rename(columns={"Country Name": "country_name_gdp", "Country Code": "iso3", "2018": "gdp_per_capita_2018"})
    gdp["iso3"] = gdp["iso3"].astype(str).str.strip().str.upper()
    return gdp[["country_name_gdp", "iso3", "gdp_per_capita_2018"]].drop_duplicates(subset="iso3")


def load_migration_stock_2018(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find migration stock file at {path}.")

    df = pd.read_csv(path)

    cols_needed = [
        "Country",
        "International migrant stock as a percentage of the total population (both sexes)"
    ]
    ensure_columns(df, cols_needed, "migration stock data")

    stock = df[cols_needed].copy()

    stock = stock.rename(columns={
        "Country": "country_name_migration_stock",
        "International migrant stock as a percentage of the total population (both sexes)": "migration_stock_2018"
    })

    stock["country_name_migration_stock"] = (
        stock["country_name_migration_stock"]
        .astype(str)
        .str.strip()
    )

    stock["migration_stock_2018"] = pd.to_numeric(
        stock["migration_stock_2018"], errors="coerce"
    )

    stock["iso3"] = stock["country_name_migration_stock"].apply(country_to_iso3)

    return stock[
        ["country_name_migration_stock", "iso3", "migration_stock_2018"]
    ].drop_duplicates(subset="iso3")

def load_hdi_2020(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find HDI file at {path}.")
    df = pd.read_csv(path)
    cols_needed = ["country_name_hdi", "iso3", "hdi_2020"]
    ensure_columns(df, cols_needed, "HDI data")
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["hdi_2020"] = pd.to_numeric(df["hdi_2020"], errors="coerce")
    return df[["country_name_hdi", "iso3", "hdi_2020"]].drop_duplicates(subset="iso3")

def load_niche_data(path, niche_type="gdp_per_capita_2018"):
    if niche_type == "gdp_per_capita_2018":
        return load_gdp_per_capita_2018(path)
    elif niche_type == "migration_stock_2018":
        return load_migration_stock_2018(path)
    elif niche_type == "hdi_2020":
        return load_hdi_2020(path)
    else:
        raise ValueError(f"Unsupported niche type: {niche_type}")




def build_master_country_table(country_geo, pop, niche_df=None, niche_col="gdp_per_capita_2018"):
    master = country_geo.copy()

    for c in ["code", "iso3"]:
        master[c] = master[c].astype(str).str.strip().str.upper()

    pop_ = pop.copy()
    pop_["iso3"] = pop_["iso3"].astype(str).str.strip().str.upper()
    master = master.merge(pop_[["iso3", "population"]], on="iso3", how="left")

    if niche_df is not None:
        niche = niche_df.copy()
        niche["iso3"] = niche["iso3"].astype(str).str.strip().str.upper()

        if niche_col not in niche.columns:
            raise ValueError(f"{niche_col} not found in niche_df columns")

        master = master.merge(niche[["iso3", niche_col]], on="iso3", how="left")

    fallback = {
        "XKX": {"population": 1800000, niche_col: np.nan},
        "TWN": {"population": 23500000, niche_col: np.nan},
    }

    for iso3, vals in fallback.items():
        mask = master["iso3"] == iso3
        master.loc[mask & master["population"].isna(), "population"] = vals["population"]

        if niche_col in master.columns:
            master.loc[mask & master[niche_col].isna(), niche_col] = vals[niche_col]

    return master.drop_duplicates(subset="code").copy()




def add_niche(country_df, niche_col="gdp_per_capita_2018", method="log_minmax"):
    out = country_df.copy()

    if niche_col not in out.columns:
        raise ValueError(f"{niche_col} not found in country_df columns")

    v = pd.to_numeric(out[niche_col], errors="coerce")

    if method == "log_minmax":
        x = np.log1p(v)
        out["niche_raw"] = x
        rng = x.max() - x.min()
        out["niche"] = (x - x.min()) / rng if pd.notna(rng) and rng != 0 else np.nan

    elif method == "zscore_log":
        x = np.log1p(v)
        out["niche_raw"] = x
        std = x.std()
        out["niche"] = (x - x.mean()) / std if pd.notna(std) and std != 0 else np.nan

    elif method == "minmax":
        out["niche_raw"] = v
        rng = v.max() - v.min()
        out["niche"] = (v - v.min()) / rng if pd.notna(rng) and rng != 0 else np.nan

    elif method == "zscore":
        out["niche_raw"] = v
        std = v.std()
        out["niche"] = (v - v.mean()) / std if pd.notna(std) and std != 0 else np.nan

    elif method == "rank":
        out["niche_raw"] = v
        out["niche"] = v.rank(method="average", pct=True)

    else:
        raise ValueError("unknown method")

    return out

# def plot_population_vs_niche(country_df, pop_col="population", niche_col="niche", label_col="country_name"):
#     df = country_df.copy()

#     df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")
#     df[niche_col] = pd.to_numeric(df[niche_col], errors="coerce")
#     df = df.dropna(subset=[pop_col, niche_col])

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(df[pop_col], df[niche_col], alpha=0.7)

#     ax.set_xscale("log")
#     ax.set_xlabel("Population (log scale)")
#     ax.set_ylabel("Niche")
#     ax.set_title("Population vs niche")

#     plt.tight_layout()
#     plt.show()



