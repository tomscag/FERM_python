from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .model import FERM, RM, build_distance_matrix, normalize_country_code_to_iso3
from .preprocessing import split_flows_by_period
from .utils import (
    ensure_columns,
    load_country_geometries_global,
    load_gdp_per_capita_2018,
    load_migration_data,
    load_population_data,
)


ScopeKind = Literal["world", "within_continent", "corridor"]
FeatureKind = Literal["gdp", "sci", "gdp_sci"]
DistanceSource = Literal["gravity", "population_center"]
SelectionMetric = Literal[
    "median_improvement_abs_log",
    "share_better_abs_log",
    "r2_log_gain_vs_rm",
    "description_length_gain_bits",
]


@dataclass(frozen=True)
class GeographicScope:
    """Geographic universe used by one country-level experiment."""

    kind: ScopeKind = "world"
    origin_regions: tuple[str, ...] = ()
    destination_regions: tuple[str, ...] = ()

    @classmethod
    def world(cls) -> "GeographicScope":
        return cls(kind="world")

    @classmethod
    def within_continent(cls, continent: str) -> "GeographicScope":
        return cls(
            kind="within_continent",
            origin_regions=(continent,),
            destination_regions=(continent,),
        )

    @classmethod
    def corridor(
        cls,
        origin_regions: Iterable[str],
        destination_regions: Iterable[str],
    ) -> "GeographicScope":
        return cls(
            kind="corridor",
            origin_regions=tuple(origin_regions),
            destination_regions=tuple(destination_regions),
        )

    @property
    def label(self) -> str:
        if self.kind == "world":
            return "world"
        if self.kind == "within_continent":
            return f"within_{self.origin_regions[0].lower().replace(' ', '_')}"
        origins = "_".join(r.lower().replace(" ", "_") for r in self.origin_regions)
        destinations = "_".join(
            r.lower().replace(" ", "_") for r in self.destination_regions
        )
        return f"{origins}_to_{destinations}"


@dataclass(frozen=True)
class FeatureSpec:
    """Feature/Sigma specification to evaluate."""

    name: str
    kind: FeatureKind
    gdp_normalization: str = "zscore_log"
    sci_normalization: str = "minmax_signed"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a reproducible country-level RM/FERM experiment."""

    data_dir: Path = Path("data")
    gravity_path: Path = Path("Gravity_csv_V202211/Gravity_V202211_bilateral_nonbinary.csv")
    country_metadata_path: Path | None = None
    population_centers_path: Path = Path("country-center_of_populations.csv")
    output_dir: Path = Path("outputs/simulation_artifacts/country_experiment")
    scope: GeographicScope = field(default_factory=GeographicScope.world)
    feature_specs: tuple[FeatureSpec, ...] = (
        FeatureSpec(name="gdp_zscore_log", kind="gdp"),
        FeatureSpec(name="sci_minmax_signed", kind="sci"),
        FeatureSpec(name="gdp_plus_sci_additive", kind="gdp_sci"),
    )
    calibration_period: str = "validation_2019_h1"
    test_period: str = "test_2019_h2"
    sigma_grid: tuple[float, ...] = (
        0.05,
        0.1,
        0.2,
        0.35,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
    )
    refine_sigma_grid: bool = True
    seeds: tuple[int, ...] = (11, 29, 47)
    num_particles: int = 2000
    selection_metric: SelectionMetric = "median_improvement_abs_log"
    distance_source: DistanceSource = "gravity"
    gravity_year: int = 2021
    observed_routes_only: bool = True

    @property
    def flow_path(self) -> Path:
        return self.data_dir / "migrations/international_migration_flow.csv"

    @property
    def population_path(self) -> Path:
        return self.data_dir / "population/API_SP.POP.TOTL_DS2_en_csv_v2_61.csv"

    @property
    def gdp_path(self) -> Path:
        return self.data_dir / "features/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_46.csv"

    @property
    def gravity_countries_path(self) -> Path:
        return self.gravity_path.parent / "Countries_V202211.csv"

    @property
    def resolved_country_metadata_path(self) -> Path:
        if self.country_metadata_path is not None:
            return self.country_metadata_path
        return self.data_dir / "features/country_metadata_naturalearth_50m.csv"


def load_gravity_country_metadata(path: Path) -> pd.DataFrame:
    """Load offline country names/codes from the Gravity country table."""

    countries = pd.read_csv(path)
    ensure_columns(countries, ["iso3", "iso2", "country"], "Gravity countries")

    out = countries[["iso3", "iso2", "country"]].copy()
    out["iso3"] = out["iso3"].astype(str).str.strip().str.upper()
    out["code"] = out["iso2"].astype(str).str.strip().str.upper()
    out.loc[
        out["iso2"].isna() | out["code"].isin(["", "NAN", "NONE"]),
        "code",
    ] = out["iso3"]
    out["country_name"] = out["country"].astype(str).str.strip()
    out["continent"] = pd.NA
    out["lat"] = np.nan
    out["lon"] = np.nan
    out["coord_source"] = "gravity_countries_no_coordinates"
    manual = pd.DataFrame(
        [
            {
                "code": "XK",
                "iso3": "XKX",
                "country_name": "Kosovo",
                "continent": pd.NA,
                "lat": np.nan,
                "lon": np.nan,
                "coord_source": "manual_no_coordinates",
            },
            {
                "code": "TW",
                "iso3": "TWN",
                "country_name": "Taiwan",
                "continent": pd.NA,
                "lat": np.nan,
                "lon": np.nan,
                "coord_source": "manual_no_coordinates",
            },
        ]
    )
    out = pd.concat(
        [
            out[
                [
                    "code",
                    "iso3",
                    "country_name",
                    "continent",
                    "lat",
                    "lon",
                    "coord_source",
                ]
            ],
            manual,
        ],
        ignore_index=True,
    )
    return out[
        ["code", "iso3", "country_name", "continent", "lat", "lon", "coord_source"]
    ].drop_duplicates(subset="iso3", keep="last")


def load_country_metadata(
    cache_path: Path,
    gravity_countries_path: Path,
) -> pd.DataFrame:
    """
    Load country metadata from a local cache, with Natural Earth as optional fallback.

    The experiment runner uses Gravity distances, so lat/lon are not required
    for RM/FERM. Continents are required only for regional/corridor scopes.
    """

    if cache_path.exists():
        metadata = pd.read_csv(cache_path)
    else:
        try:
            _, country_geo = load_country_geometries_global()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            country_geo.to_csv(cache_path, index=False)
            metadata = country_geo
        except Exception:
            metadata = load_gravity_country_metadata(gravity_countries_path)

    for col in ["code", "iso3"]:
        metadata[col] = metadata[col].astype(str).str.strip().str.upper()

    required = ["code", "iso3", "country_name", "continent", "lat", "lon"]
    for col in required:
        if col not in metadata.columns:
            metadata[col] = pd.NA

    metadata.loc[metadata["code"].isin(["", "NAN", "NONE"]), "code"] = metadata["iso3"]
    return metadata[required + [c for c in metadata.columns if c not in required]].drop_duplicates(
        subset="iso3"
    )


def build_experiment_country_table(
    country_metadata: pd.DataFrame,
    populations: pd.DataFrame,
    gdp: pd.DataFrame,
) -> pd.DataFrame:
    """Merge metadata, population, and GDP into the country table used by experiments."""

    country = country_metadata.copy()
    populations = populations.copy()
    gdp = gdp.copy()

    for frame in [country, populations, gdp]:
        frame["iso3"] = frame["iso3"].astype(str).str.strip().str.upper()

    country = country.merge(
        populations[["iso3", "population"]],
        on="iso3",
        how="left",
    )
    country = country.merge(
        gdp[["iso3", "gdp_per_capita_2018"]],
        on="iso3",
        how="left",
    )

    fallback_population = {"XKX": 1800000, "TWN": 23500000}
    for iso3, population in fallback_population.items():
        mask = country["iso3"] == iso3
        country.loc[mask & country["population"].isna(), "population"] = population

    return country.drop_duplicates(subset="iso3")


def prepare_experiment_nodes(
    country_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    require_coordinates: bool = False,
) -> pd.DataFrame:
    """Extract model nodes from experiment metadata."""

    used_codes = sorted(set(flows_df["country_from"]).union(set(flows_df["country_to"])))
    nodes = country_df[country_df["iso3"].isin(used_codes)].drop_duplicates("iso3").copy()

    required = ["code", "iso3", "country_name", "population"]
    if require_coordinates:
        required.extend(["lat", "lon"])
    ensure_columns(nodes, required, "experiment nodes")

    nodes["code"] = nodes["code"].fillna(nodes["iso3"]).astype(str).str.strip().str.upper()
    nodes = nodes.dropna(subset=["population"]).copy()

    missing_codes = sorted(set(used_codes) - set(nodes["iso3"]))
    if missing_codes:
        print(f"Warning: missing metadata for these countries: {missing_codes}")

    return nodes


def filter_nodes_and_flows_by_feature_coverage(
    nodes: pd.DataFrame,
    scoped_flows: pd.DataFrame,
    calibration_flows: pd.DataFrame,
    test_flows: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Drop countries that lack node features required by the configured models."""

    dropped: dict[str, list[str]] = {}
    keep_codes = set(nodes["iso3"])

    needs_gdp = any(spec.kind in {"gdp", "gdp_sci"} for spec in config.feature_specs)
    if needs_gdp:
        gdp = load_gdp_per_capita_2018(config.gdp_path)
        gdp["iso3"] = gdp["iso3"].astype(str).str.strip().str.upper()
        valid_gdp_codes = set(
            gdp.loc[gdp["gdp_per_capita_2018"].notna(), "iso3"].astype(str)
        )
        missing_gdp = sorted(keep_codes - valid_gdp_codes)
        if missing_gdp:
            dropped["missing_gdp_per_capita_2018"] = missing_gdp
            keep_codes &= valid_gdp_codes

    nodes = nodes[nodes["iso3"].isin(keep_codes)].copy()

    def filter_flows(flows: pd.DataFrame) -> pd.DataFrame:
        return flows[
            flows["country_from"].isin(keep_codes)
            & flows["country_to"].isin(keep_codes)
        ].copy()

    return (
        nodes,
        filter_flows(scoped_flows),
        filter_flows(calibration_flows),
        filter_flows(test_flows),
        dropped,
    )


def fit_vector_normalization(values: pd.Series, method: str) -> dict[str, object]:
    """Fit node-feature normalization parameters on a reference sample."""
    x = pd.to_numeric(values, errors="coerce").astype(float)

    if method == "zscore_log":
        y = np.log1p(x)
        std = y.std(ddof=0)
        return {
            "method": method,
            "center": float(y.mean()),
            "scale": float(std) if std and np.isfinite(std) else 1.0,
        }

    if method == "log_minmax":
        y = np.log1p(x)
        rng = y.max() - y.min()
        return {
            "method": method,
            "min": float(y.min()),
            "range": float(rng) if rng and np.isfinite(rng) else 1.0,
        }

    if method == "log_rank":
        y = np.log1p(x)
        return {
            "method": method,
            "sorted_values": np.sort(y.dropna().to_numpy(dtype=float)).tolist(),
        }

    if method == "clipped_zscore_log":
        params = fit_vector_normalization(x, "zscore_log")
        params["method"] = method
        return params

    raise ValueError(f"Unsupported vector normalization: {method}")


def apply_vector_normalization(
    values: pd.Series,
    method: str,
    params: dict[str, object] | None = None,
) -> pd.Series:
    """Apply node-feature normalization, optionally using pre-fit parameters."""

    if params is None:
        params = fit_vector_normalization(values, method)
    if params.get("method") != method:
        raise ValueError(
            f"Normalization parameters were fit for {params.get('method')!r}, "
            f"not {method!r}."
        )

    x = pd.to_numeric(values, errors="coerce").astype(float)
    if method == "zscore_log":
        y = np.log1p(x)
        return (y - float(params["center"])) / float(params["scale"])

    if method == "log_minmax":
        y = np.log1p(x)
        return (y - float(params["min"])) / float(params["range"])

    if method == "log_rank":
        y = np.log1p(x)
        sorted_values = np.asarray(params["sorted_values"], dtype=float)
        if sorted_values.size == 0:
            return y * 0.0
        ranks = np.searchsorted(sorted_values, y.to_numpy(dtype=float), side="right")
        out = pd.Series(ranks / sorted_values.size, index=values.index, dtype=float)
        out[y.isna()] = np.nan
        return 2.0 * out - 1.0

    if method == "clipped_zscore_log":
        z_params = dict(params)
        z_params["method"] = "zscore_log"
        return apply_vector_normalization(values, "zscore_log", z_params).clip(-3.0, 3.0)

    raise ValueError(f"Unsupported vector normalization: {method}")


def normalize_vector(values: pd.Series, method: str) -> pd.Series:
    """Normalize a node-level feature into a signed, dimensionless series."""

    return apply_vector_normalization(values, method)


def normalize_matrix(values: pd.DataFrame, method: str) -> pd.DataFrame:
    """Normalize a corridor-level feature matrix."""

    return apply_matrix_normalization(values, method)


def fit_matrix_normalization(values: pd.DataFrame, method: str) -> dict[str, object]:
    """Fit corridor-feature normalization parameters on a reference matrix."""

    x = values.astype(float).copy()
    offdiag = ~np.eye(x.shape[0], dtype=bool)

    if method == "minmax_signed":
        arr = x.to_numpy(dtype=float)
        finite = arr[offdiag & np.isfinite(arr)]
        if finite.size == 0:
            return {"method": method, "min": 0.0, "range": 1.0}
        lo = finite.min()
        hi = finite.max()
        return {
            "method": method,
            "min": float(lo),
            "range": float(hi - lo) if hi != lo else 1.0,
        }

    if method == "global_rank_signed":
        arr = x.to_numpy(dtype=float)
        finite = arr[offdiag & np.isfinite(arr)]
        return {
            "method": method,
            "sorted_values": np.sort(finite.astype(float)).tolist(),
        }

    if method == "origin_log_zscore":
        y = np.log1p(x.clip(lower=0.0).fillna(0.0))
        y_values = y.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(y_values, np.nan)
        y = pd.DataFrame(y_values, index=y.index, columns=y.columns)
        center = y.mean(axis=1)
        scale = y.std(axis=1, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return {
            "method": method,
            "center_by_origin": center.to_dict(),
            "scale_by_origin": scale.to_dict(),
        }

    raise ValueError(f"Unsupported matrix normalization: {method}")


def apply_matrix_normalization(
    values: pd.DataFrame,
    method: str,
    params: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Apply corridor-feature normalization, optionally using pre-fit parameters."""

    x = values.astype(float).copy()
    if params is None:
        params = fit_matrix_normalization(x, method)
    if params.get("method") != method:
        raise ValueError(
            f"Matrix normalization parameters were fit for {params.get('method')!r}, "
            f"not {method!r}."
        )

    if method == "minmax_signed":
        out = 2.0 * (x - float(params["min"])) / float(params["range"]) - 1.0
        return out.fillna(-1.0)

    if method == "global_rank_signed":
        sorted_values = np.asarray(params["sorted_values"], dtype=float)
        if sorted_values.size == 0:
            return x.fillna(0.0)
        arr = x.to_numpy(dtype=float)
        ranks = np.searchsorted(sorted_values, arr, side="right") / sorted_values.size
        out = pd.DataFrame(2.0 * ranks - 1.0, index=x.index, columns=x.columns)
        return out.where(np.isfinite(arr), -1.0)

    if method == "origin_log_zscore":
        y = np.log1p(x.clip(lower=0.0).fillna(0.0))
        center = pd.Series(params["center_by_origin"]).reindex(y.index).astype(float)
        scale = pd.Series(params["scale_by_origin"]).reindex(y.index).astype(float)
        missing = center.isna() | scale.isna()
        if missing.any():
            fallback_center = y.mean(axis=1)
            fallback_scale = y.std(axis=1, ddof=0).replace(0.0, 1.0).fillna(1.0)
            center = center.fillna(fallback_center)
            scale = scale.fillna(fallback_scale)
        return y.sub(center, axis=0).div(scale, axis=0).fillna(0.0)

    raise ValueError(f"Unsupported matrix normalization: {method}")


def build_destination_sigma(
    nodes: pd.DataFrame,
    feature_by_iso3: pd.Series,
    normalization: str = "zscore_log",
    normalization_params: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    Build traditional/node FERM Sigma:
    Sigma_ii = mu_i and Sigma_ij = mu_j.
    """

    codes = nodes["iso3"].astype(str).str.upper().tolist()
    mu = apply_vector_normalization(
        feature_by_iso3.reindex(codes),
        normalization,
        params=normalization_params,
    )
    if mu.isna().any():
        missing = mu[mu.isna()].index.tolist()
        raise ValueError(f"Missing node feature values for: {missing}")

    sigma = pd.DataFrame(
        np.tile(mu.to_numpy()[None, :], (len(codes), 1)),
        index=codes,
        columns=codes,
    )
    for code in codes:
        sigma.loc[code, code] = mu.loc[code]
    return sigma


def build_relational_sigma(
    nodes: pd.DataFrame,
    corridor_matrix: pd.DataFrame,
    normalization: str = "minmax_signed",
    normalization_params: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    Build relational/corridor FERM Sigma:
    Sigma_ii = 0 and Sigma_ij = delta_ij for i != j.
    """

    codes = nodes["iso3"].astype(str).str.upper().tolist()
    raw = corridor_matrix.reindex(index=codes, columns=codes).astype(float)
    normalized = apply_matrix_normalization(
        raw,
        normalization,
        params=normalization_params,
    ).fillna(0.0)
    for code in codes:
        normalized.loc[code, code] = 0.0
    return normalized


def build_combined_sigma(
    destination_sigma: pd.DataFrame,
    relational_sigma: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build additive GDP + relational Sigma:
    Sigma_ii = mu_i and Sigma_ij = mu_j + delta_ij for i != j.
    """

    combined = destination_sigma.add(relational_sigma, fill_value=0.0)
    for code in combined.index:
        combined.loc[code, code] = destination_sigma.loc[code, code]
    return combined


def load_gravity_matrix(
    gravity_path: Path,
    value_col: str,
    year: int = 2021,
) -> pd.DataFrame:
    """Load one bilateral Gravity variable as an ISO3 origin-destination matrix."""

    cols = ["year", "iso3_o", "iso3_d", value_col]
    gravity = pd.read_csv(gravity_path, usecols=cols)
    ensure_columns(gravity, cols, "Gravity data")

    gravity = gravity[gravity["year"] == year].copy()
    if gravity.empty:
        raise ValueError(f"Gravity data has no rows for year {year}.")

    gravity["iso3_o"] = gravity["iso3_o"].astype(str).str.strip().str.upper()
    gravity["iso3_d"] = gravity["iso3_d"].astype(str).str.strip().str.upper()
    gravity[value_col] = pd.to_numeric(gravity[value_col], errors="coerce")

    return gravity.pivot_table(
        index="iso3_o",
        columns="iso3_d",
        values=value_col,
        aggfunc="mean",
    )


def load_gravity_distance_matrix(
    gravity_path: Path,
    year: int = 2021,
    value_col: str = "dist",
) -> pd.DataFrame:
    """Load Gravity distances as an ISO3 origin-destination matrix."""

    return load_gravity_matrix(gravity_path, value_col=value_col, year=year)


def load_population_center_distance_matrix(
    population_centers_path: Path,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Build distances between country centers of population.

    The input CSV is expected to contain ISO3 country codes in `alpha3` and
    center-of-population coordinates in `latitude`/`longitude`.
    """

    path = Path(population_centers_path)
    if not path.exists() and not path.is_absolute():
        local_path = Path(__file__).resolve().parents[2] / path
        if local_path.exists():
            path = local_path

    centers = pd.read_csv(path)
    ensure_columns(
        centers,
        ["alpha3", "latitude", "longitude"],
        "country center-of-population data",
    )

    centers = centers.rename(
        columns={"alpha3": "iso3", "latitude": "lat", "longitude": "lon"}
    )
    centers["iso3"] = centers["iso3"].map(normalize_country_code_to_iso3)
    centers = centers.dropna(subset=["iso3", "lat", "lon"]).copy()
    centers = centers.drop_duplicates("iso3")

    codes = nodes["iso3"].astype(str).str.strip().str.upper().tolist()
    center_nodes = centers[centers["iso3"].isin(codes)][["iso3", "lat", "lon"]].copy()
    center_nodes["code"] = center_nodes["iso3"]

    distance_matrix = build_distance_matrix(center_nodes)
    return distance_matrix.reindex(index=codes, columns=codes)


def filter_flows_for_scope(
    flows: pd.DataFrame,
    country_table: pd.DataFrame,
    scope: GeographicScope,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter migration flows and country metadata for a world/region/corridor scope."""

    flows_out = flows.copy()
    countries = country_table.copy()
    countries["iso3"] = countries["iso3"].astype(str).str.strip().str.upper()

    if scope.kind == "world":
        used_codes = sorted(
            set(flows_out["country_from"].dropna()).union(flows_out["country_to"].dropna())
        )
        countries = countries[countries["iso3"].isin(used_codes)].copy()
        return flows_out, countries

    if "continent" not in countries.columns or countries["continent"].isna().all():
        raise ValueError(
            "Regional scopes require country metadata with a `continent` column. "
            "Create or provide `country_metadata_path`, or run once with access "
            "to Natural Earth so the cache can be written."
        )

    continent_by_iso3 = countries.set_index("iso3")["continent"].to_dict()
    flows_out["origin_continent"] = flows_out["country_from"].map(continent_by_iso3)
    flows_out["destination_continent"] = flows_out["country_to"].map(continent_by_iso3)

    if scope.kind == "within_continent":
        region = scope.origin_regions[0]
        flows_out = flows_out[
            (flows_out["origin_continent"] == region)
            & (flows_out["destination_continent"] == region)
        ].copy()
    elif scope.kind == "corridor":
        flows_out = flows_out[
            flows_out["origin_continent"].isin(scope.origin_regions)
            & flows_out["destination_continent"].isin(scope.destination_regions)
        ].copy()
    else:
        raise ValueError(f"Unsupported scope kind: {scope.kind}")

    used_codes = sorted(
        set(flows_out["country_from"].dropna()).union(flows_out["country_to"].dropna())
    )
    countries = countries[countries["iso3"].isin(used_codes)].copy()

    return (
        flows_out.drop(columns=["origin_continent", "destination_continent"]),
        countries,
    )


def describe_coverage(
    nodes: pd.DataFrame,
    flows: pd.DataFrame,
    sigmas: dict[str, pd.DataFrame],
    distance_matrix: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize country/route/data availability for metadata and logs."""

    codes = nodes["iso3"].tolist()
    possible_routes = max(len(codes) * (len(codes) - 1), 0)
    observed_routes = flows[["country_from", "country_to"]].drop_duplicates().shape[0]
    out: dict[str, float | int] = {
        "n_countries": len(codes),
        "possible_routes": possible_routes,
        "observed_routes": int(observed_routes),
        "observed_migrants": float(flows["num_migrants"].sum()),
        "distance_coverage": float(
            distance_matrix.reindex(index=codes, columns=codes).notna().mean().mean()
        ),
    }
    for name, sigma in sigmas.items():
        offdiag = sigma.to_numpy(dtype=float).copy()
        np.fill_diagonal(offdiag, np.nan)
        out[f"{name}_sigma_offdiag_p01"] = float(np.nanpercentile(offdiag, 1))
        out[f"{name}_sigma_offdiag_p50"] = float(np.nanpercentile(offdiag, 50))
        out[f"{name}_sigma_offdiag_p99"] = float(np.nanpercentile(offdiag, 99))
    return out


def complete_matrix_codes(matrix: pd.DataFrame) -> list[str]:
    """
    Return a code subset whose square matrix has no missing values.

    Gravity can miss small territories. A single missing country creates NaNs in
    both a row and a column, so dropping every row with any NaN would often drop
    the whole scope. This removes the worst-covered code iteratively instead.
    """

    reduced = matrix.copy()
    while len(reduced) > 0 and reduced.isna().any().any():
        row_missing = reduced.isna().sum(axis=1)
        col_missing = reduced.isna().sum(axis=0)
        missing_score = row_missing.add(col_missing, fill_value=0)
        drop_code = str(missing_score.sort_values(ascending=False).index[0])
        reduced = reduced.drop(index=drop_code, columns=drop_code, errors="ignore")
    return [str(code) for code in reduced.index]


def comparison_keyed(
    comparison: pd.DataFrame,
    observed_col: str = "num_migrants",
) -> pd.DataFrame:
    """Return one row per OD pair with standard route-level diagnostics."""

    out = comparison.copy()
    out["country_from"] = out["country_from"].map(normalize_country_code_to_iso3)
    out["country_to"] = out["country_to"].map(normalize_country_code_to_iso3)
    out[observed_col] = pd.to_numeric(out[observed_col], errors="coerce").fillna(0.0)
    return out


def route_metrics_vs_rm(
    rm_comparison: pd.DataFrame,
    model_comparison: pd.DataFrame,
    observed_routes_only: bool = True,
) -> dict[str, float]:
    """Compute route-level FERM-vs-RM metrics on matched OD rows."""

    rm = comparison_keyed(rm_comparison)
    model = comparison_keyed(model_comparison)

    keep = ["country_from", "country_to", "num_migrants", "abs_log_ratio", "log_observed"]
    merged = rm[keep + ["log_predicted"]].rename(
        columns={
            "abs_log_ratio": "abs_log_ratio_rm",
            "log_predicted": "log_predicted_rm",
        }
    ).merge(
        model[
            [
                "country_from",
                "country_to",
                "abs_log_ratio",
                "log_predicted",
            ]
        ].rename(
            columns={
                "abs_log_ratio": "abs_log_ratio_model",
                "log_predicted": "log_predicted_model",
            }
        ),
        on=["country_from", "country_to"],
        how="inner",
    )

    if observed_routes_only:
        merged = merged[merged["num_migrants"] > 0].copy()

    if merged.empty:
        return {
            "n_eval_routes": 0.0,
            "median_improvement_abs_log": np.nan,
            "share_better_abs_log": np.nan,
            "pearson_log_model": np.nan,
            "pearson_log_rm": np.nan,
            "r2_log_gain_vs_rm": np.nan,
        }

    rm_abs = merged["abs_log_ratio_rm"]
    model_abs = merged["abs_log_ratio_model"]
    obs_log = merged["log_observed"]
    rm_pred = merged["log_predicted_rm"]
    model_pred = merged["log_predicted_model"]

    def safe_pearson(left: pd.Series, right: pd.Series) -> float:
        if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
            return np.nan
        return float(left.corr(right, method="pearson"))

    sse_rm = float(np.square(obs_log - rm_pred).sum())
    sse_model = float(np.square(obs_log - model_pred).sum())
    r2_gain = 1.0 - sse_model / sse_rm if sse_rm > 0 else np.nan

    return {
        "n_eval_routes": float(len(merged)),
        "median_improvement_abs_log": float(np.median(rm_abs - model_abs)),
        "share_better_abs_log": float((model_abs <= rm_abs).mean()),
        "pearson_log_model": safe_pearson(obs_log, model_pred),
        "pearson_log_rm": safe_pearson(obs_log, rm_pred),
        "r2_log_gain_vs_rm": float(r2_gain),
    }


def description_length_bits(
    flows: pd.DataFrame,
    probabilities: pd.DataFrame,
    eps: float = 1e-15,
) -> float:
    """Multinomial negative log likelihood in bits, conditional on origin totals."""

    flow_table = flows.copy()
    flow_table["country_from"] = flow_table["country_from"].map(normalize_country_code_to_iso3)
    flow_table["country_to"] = flow_table["country_to"].map(normalize_country_code_to_iso3)
    flow_table["num_migrants"] = pd.to_numeric(
        flow_table["num_migrants"], errors="coerce"
    ).fillna(0.0)

    total = 0.0
    for row in flow_table.itertuples(index=False):
        origin = row.country_from
        destination = row.country_to
        if origin not in probabilities.index or destination not in probabilities.columns:
            continue
        p = max(float(probabilities.loc[origin, destination]), eps)
        total -= float(row.num_migrants) * np.log2(p)
    return float(total)


def evaluate_against_rm(
    flows: pd.DataFrame,
    rm_result,
    model_result,
    observed_routes_only: bool = True,
) -> dict[str, float]:
    """Combine route metrics and description-length gain."""

    metrics = route_metrics_vs_rm(
        rm_result.comparison,
        model_result.comparison,
        observed_routes_only=observed_routes_only,
    )
    rm_dl = description_length_bits(flows, rm_result.probability_matrix)
    model_dl = description_length_bits(flows, model_result.probability_matrix)
    metrics["description_length_bits"] = model_dl
    metrics["description_length_rm_bits"] = rm_dl
    metrics["description_length_gain_bits"] = rm_dl - model_dl
    return metrics


def refined_sigma_grid(sigma_grid: Iterable[float], best_sigma: float) -> tuple[float, ...]:
    """Create a small local grid around the best coarse sigma."""

    grid = np.array(sorted(set(float(v) for v in sigma_grid if float(v) > 0.0)))
    if best_sigma not in grid:
        grid = np.array(sorted(set(grid.tolist() + [float(best_sigma)])))
    idx = int(np.where(grid == best_sigma)[0][0])
    lo = grid[max(idx - 1, 0)]
    hi = grid[min(idx + 1, len(grid) - 1)]
    if lo == hi:
        return (float(best_sigma),)
    refined = np.geomspace(lo, hi, num=7)
    return tuple(float(v) for v in sorted(set(np.round(refined, 8))))


def choose_best_sigma(
    calibration_summary: pd.DataFrame,
    metric: SelectionMetric,
) -> pd.DataFrame:
    """Pick the best sigma per feature from aggregated calibration rows."""

    ascending = metric == "description_length_bits"
    rows = []
    for feature_name, group in calibration_summary.groupby("feature_name"):
        selected = group.sort_values(metric, ascending=ascending).iloc[0].copy()
        selected["selected_metric"] = metric
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_seed_metrics(rows: list[dict[str, float | str | int]]) -> pd.DataFrame:
    """Aggregate repeated-seed metric rows by feature and sigma."""

    df = pd.DataFrame(rows)
    metric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "feature_name",
            "feature_kind",
            "sigma",
            "seed",
            "stage",
        }
    ]
    grouped = (
        df.groupby(["feature_name", "feature_kind", "sigma"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["feature_name", "sigma"])
    )
    return grouped


def build_experiment_sigmas(
    nodes: pd.DataFrame,
    config: ExperimentConfig,
    calibration_nodes: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all Sigma matrices requested by an experiment config."""

    gdp = load_gdp_per_capita_2018(config.gdp_path)
    gdp_by_iso3 = gdp.set_index("iso3")["gdp_per_capita_2018"]
    if calibration_nodes is None:
        calibration_nodes = nodes
    calibration_codes = calibration_nodes["iso3"].astype(str).str.upper().tolist()
    sci_matrix = load_gravity_matrix(
        config.gravity_path,
        value_col="scaled_sci_2021",
        year=config.gravity_year,
    )
    sigmas: dict[str, pd.DataFrame] = {}
    for spec in config.feature_specs:
        if spec.kind == "gdp":
            gdp_params = fit_vector_normalization(
                gdp_by_iso3.reindex(calibration_codes),
                spec.gdp_normalization,
            )
            sigmas[spec.name] = build_destination_sigma(
                nodes,
                gdp_by_iso3,
                normalization=spec.gdp_normalization,
                normalization_params=gdp_params,
            )
        elif spec.kind == "sci":
            sci_params = fit_matrix_normalization(
                sci_matrix.reindex(index=calibration_codes, columns=calibration_codes),
                spec.sci_normalization,
            )
            sigmas[spec.name] = build_relational_sigma(
                nodes,
                sci_matrix,
                normalization=spec.sci_normalization,
                normalization_params=sci_params,
            )
        elif spec.kind == "gdp_sci":
            gdp_params = fit_vector_normalization(
                gdp_by_iso3.reindex(calibration_codes),
                spec.gdp_normalization,
            )
            gdp_sigma = build_destination_sigma(
                nodes,
                gdp_by_iso3,
                normalization=spec.gdp_normalization,
                normalization_params=gdp_params,
            )
            sci_sigma = build_relational_sigma(
                nodes,
                sci_matrix,
                normalization=spec.sci_normalization,
                normalization_params=fit_matrix_normalization(
                    sci_matrix.reindex(index=calibration_codes, columns=calibration_codes),
                    spec.sci_normalization,
                ),
            )
            sigmas[spec.name] = build_combined_sigma(gdp_sigma, sci_sigma)
        else:
            raise ValueError(f"Unsupported feature kind: {spec.kind}")

    return sigmas


def prepare_experiment_data(config: ExperimentConfig) -> dict[str, object]:
    """Load and filter migration, country, feature, and distance data."""

    flows = load_migration_data(config.flow_path)
    populations = load_population_data(config.population_path)
    gdp = load_gdp_per_capita_2018(config.gdp_path)
    country_metadata = load_country_metadata(
        cache_path=config.resolved_country_metadata_path,
        gravity_countries_path=config.gravity_countries_path,
    )
    master = build_experiment_country_table(
        country_metadata=country_metadata,
        populations=populations,
        gdp=gdp,
    )
    scoped_flows, scoped_countries = filter_flows_for_scope(flows, master, config.scope)
    pair_lookup = split_flows_by_period(scoped_flows, scoped_countries)

    calibration_flows = pair_lookup[config.calibration_period].rename(
        columns={"total_migrants": "num_migrants"}
    )
    test_flows = pair_lookup[config.test_period].rename(
        columns={"total_migrants": "num_migrants"}
    )

    nodes = prepare_experiment_nodes(scoped_countries, scoped_flows)
    (
        nodes,
        scoped_flows,
        calibration_flows,
        test_flows,
        feature_dropped_codes,
    ) = filter_nodes_and_flows_by_feature_coverage(
        nodes=nodes,
        scoped_flows=scoped_flows,
        calibration_flows=calibration_flows,
        test_flows=test_flows,
        config=config,
    )

    if config.distance_source == "gravity":
        distance_matrix = load_gravity_distance_matrix(
            config.gravity_path,
            year=config.gravity_year,
        ).reindex(index=nodes["iso3"], columns=nodes["iso3"])
    elif config.distance_source == "population_center":
        distance_matrix = load_population_center_distance_matrix(
            config.population_centers_path,
            nodes=nodes,
        )
    else:
        raise ValueError(f"Unknown distance_source: {config.distance_source!r}")

    for code in distance_matrix.index:
        if code in distance_matrix.columns:
            distance_matrix.loc[code, code] = 0.0

    complete_codes = complete_matrix_codes(distance_matrix)
    if len(complete_codes) < len(distance_matrix):
        nodes = nodes[nodes["iso3"].isin(complete_codes)].copy()
        distance_matrix = distance_matrix.reindex(index=complete_codes, columns=complete_codes)
        scoped_flows = scoped_flows[
            scoped_flows["country_from"].isin(nodes["iso3"])
            & scoped_flows["country_to"].isin(nodes["iso3"])
        ].copy()
        calibration_flows = calibration_flows[
            calibration_flows["country_from"].isin(nodes["iso3"])
            & calibration_flows["country_to"].isin(nodes["iso3"])
        ].copy()
        test_flows = test_flows[
            test_flows["country_from"].isin(nodes["iso3"])
            & test_flows["country_to"].isin(nodes["iso3"])
        ].copy()

    calibration_codes = sorted(
        set(calibration_flows["country_from"].dropna()).union(
            calibration_flows["country_to"].dropna()
        )
    )
    calibration_nodes = nodes[nodes["iso3"].isin(calibration_codes)].copy()
    if calibration_nodes.empty:
        calibration_nodes = nodes

    sigmas = build_experiment_sigmas(nodes, config, calibration_nodes=calibration_nodes)

    return {
        "nodes": nodes,
        "calibration_nodes": calibration_nodes,
        "calibration_flows": calibration_flows,
        "test_flows": test_flows,
        "distance_matrix": distance_matrix,
        "sigmas": sigmas,
        "scoped_flows": scoped_flows,
        "feature_dropped_codes": feature_dropped_codes,
    }


def run_ferm_repeated(
    nodes: pd.DataFrame,
    flows: pd.DataFrame,
    sigma_matrix: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    sigmas: Iterable[float],
    seeds: Iterable[int],
    num_particles: int,
    rm_result,
    observed_routes_only: bool,
    feature_name: str,
    feature_kind: str,
    stage: str,
) -> list[dict[str, float | str | int]]:
    """Run one FERM specification over sigma and seed grids."""

    rows: list[dict[str, float | str | int]] = []
    for sigma in sigmas:
        for seed in seeds:
            model = FERM(
                nodes=nodes,
                flows=flows,
                features=sigma_matrix,
                distance_matrix=distance_matrix,
            )
            result = model.run(
                num_particles=num_particles,
                sigma=float(sigma),
                rng=np.random.default_rng(int(seed)),
            )
            metrics = evaluate_against_rm(
                flows,
                rm_result,
                result,
                observed_routes_only=observed_routes_only,
            )
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_kind": feature_kind,
                    "sigma": float(sigma),
                    "seed": int(seed),
                    "stage": stage,
                    **metrics,
                }
            )
    return rows


def run_experiment(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Run calibration and test evaluation, then save reproducible artifacts."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_experiment_data(config)
    nodes = data["nodes"]
    calibration_flows = data["calibration_flows"]
    test_flows = data["test_flows"]
    distance_matrix = data["distance_matrix"]
    sigmas = data["sigmas"]

    rm_calibration = RM(
        nodes=nodes,
        flows=calibration_flows,
        distance_matrix=distance_matrix,
    ).run()
    rm_test = RM(nodes=nodes, flows=test_flows, distance_matrix=distance_matrix).run()

    spec_by_name = {spec.name: spec for spec in config.feature_specs}
    calibration_rows: list[dict[str, float | str | int]] = []

    for feature_name, sigma_matrix in sigmas.items():
        spec = spec_by_name[feature_name]
        calibration_rows.extend(
            run_ferm_repeated(
                nodes=nodes,
                flows=calibration_flows,
                sigma_matrix=sigma_matrix,
                distance_matrix=distance_matrix,
                sigmas=config.sigma_grid,
                seeds=config.seeds,
                num_particles=config.num_particles,
                rm_result=rm_calibration,
                observed_routes_only=config.observed_routes_only,
                feature_name=feature_name,
                feature_kind=spec.kind,
                stage="coarse",
            )
        )

    calibration_summary = aggregate_seed_metrics(calibration_rows)

    if config.refine_sigma_grid:
        best_coarse = choose_best_sigma(calibration_summary, config.selection_metric)
        refined_rows: list[dict[str, float | str | int]] = []
        for row in best_coarse.itertuples(index=False):
            sigma_values = refined_sigma_grid(config.sigma_grid, float(row.sigma))
            spec = spec_by_name[row.feature_name]
            refined_rows.extend(
                run_ferm_repeated(
                    nodes=nodes,
                    flows=calibration_flows,
                    sigma_matrix=sigmas[row.feature_name],
                    distance_matrix=distance_matrix,
                    sigmas=sigma_values,
                    seeds=config.seeds,
                    num_particles=config.num_particles,
                    rm_result=rm_calibration,
                    observed_routes_only=config.observed_routes_only,
                    feature_name=row.feature_name,
                    feature_kind=spec.kind,
                    stage="refined",
                )
            )
        calibration_rows.extend(refined_rows)
        calibration_summary = aggregate_seed_metrics(calibration_rows)

    best_by_model = choose_best_sigma(calibration_summary, config.selection_metric)

    test_rows: list[dict[str, float | str | int]] = []
    for row in best_by_model.itertuples(index=False):
        spec = spec_by_name[row.feature_name]
        test_rows.extend(
            run_ferm_repeated(
                nodes=nodes,
                flows=test_flows,
                sigma_matrix=sigmas[row.feature_name],
                distance_matrix=distance_matrix,
                sigmas=(float(row.sigma),),
                seeds=config.seeds,
                num_particles=config.num_particles,
                rm_result=rm_test,
                observed_routes_only=config.observed_routes_only,
                feature_name=row.feature_name,
                feature_kind=spec.kind,
                stage="test",
            )
        )
    test_summary = aggregate_seed_metrics(test_rows)

    rm_calibration.comparison.to_csv(config.output_dir / "rm_calibration_comparison.csv", index=False)
    rm_test.comparison.to_csv(config.output_dir / "rm_test_comparison.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        config.output_dir / "calibration_routes_by_seed.csv", index=False
    )
    calibration_summary.to_csv(config.output_dir / "calibration_summary.csv", index=False)
    best_by_model.to_csv(config.output_dir / "best_by_model.csv", index=False)
    pd.DataFrame(test_rows).to_csv(config.output_dir / "test_rows_by_seed.csv", index=False)
    test_summary.to_csv(config.output_dir / "test_summary.csv", index=False)

    metadata = {
        "config": asdict(config),
        "scope_label": config.scope.label,
        "coverage": describe_coverage(nodes, data["scoped_flows"], sigmas, distance_matrix),
        "feature_dropped_codes": data["feature_dropped_codes"],
    }
    metadata["config"]["data_dir"] = str(config.data_dir)
    metadata["config"]["gravity_path"] = str(config.gravity_path)
    metadata["config"]["country_metadata_path"] = (
        str(config.country_metadata_path) if config.country_metadata_path is not None else None
    )
    metadata["config"]["output_dir"] = str(config.output_dir)
    with (config.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "calibration_summary": calibration_summary,
        "best_by_model": best_by_model,
        "test_summary": test_summary,
    }


def parse_scope(value: str) -> GeographicScope:
    """Parse CLI scope strings: world, within:Europe, corridor:Europe->Africa."""

    if value == "world":
        return GeographicScope.world()
    if value.startswith("within:"):
        return GeographicScope.within_continent(value.split(":", 1)[1])
    if value.startswith("corridor:"):
        raw = value.split(":", 1)[1]
        left, right = raw.split("->", 1)
        origins = tuple(part.strip() for part in left.split("+") if part.strip())
        destinations = tuple(part.strip() for part in right.split("+") if part.strip())
        return GeographicScope.corridor(origins, destinations)
    raise ValueError(
        "Scope must be 'world', 'within:<Continent>', or "
        "'corridor:<OriginContinent>[+...]-><DestinationContinent>[+...]'."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run country-level RM/FERM experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--gravity",
        type=Path,
        default=Path("Gravity_csv_V202211/Gravity_V202211_bilateral_nonbinary.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/simulation_artifacts/country_experiment"))
    parser.add_argument("--scope", default="world")
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--seeds", default="11,29,47")
    parser.add_argument("--sigma-grid", default="0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3,5")
    parser.add_argument(
        "--distance-source",
        choices=["gravity", "population_center"],
        default="gravity",
    )
    parser.add_argument(
        "--population-centers",
        type=Path,
        default=Path("country-center_of_populations.csv"),
    )
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args(argv)

    config = ExperimentConfig(
        data_dir=args.data_dir,
        gravity_path=args.gravity,
        population_centers_path=args.population_centers,
        output_dir=args.out,
        scope=parse_scope(args.scope),
        num_particles=args.particles,
        seeds=tuple(int(v) for v in args.seeds.split(",") if v.strip()),
        sigma_grid=tuple(float(v) for v in args.sigma_grid.split(",") if v.strip()),
        refine_sigma_grid=not args.no_refine,
        distance_source=args.distance_source,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
