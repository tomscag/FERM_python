from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from typing import Optional

import numpy as np
import pandas as pd
import pycountry
from scipy import stats
from tqdm import tqdm

from pathlib import Path 

from .utils import iso3_to_country


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6371.0
DEFAULT_EPS: float = 1.0


def normalize_country_code_to_iso3(code: object) -> object:
    """
    Normalize ISO2/ISO3-like country codes to ISO3 when possible.
    """
    if pd.isna(code):
        return code

    value = str(code).strip().upper()

    if len(value) == 3:
        return value

    if len(value) == 2:
        try:
            return pycountry.countries.lookup(value).alpha_3
        except LookupError:
            return value

    try:
        return pycountry.countries.lookup(value).alpha_3
    except LookupError:
        return value


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def validate_nodes(
    nodes: pd.DataFrame,
    required_columns: set[str],
    node_col:str = "iso3",
) -> None:
    """
    Validate the node table used by the mobility models.

    Parameters
    ----------
    nodes : pd.DataFrame
        Input node table.
    required_columns : set[str]
        Columns that MUST be present.

    Raises
    ------
    ValueError
        If required columns are missing, node codes are duplicated,
        or the input table is empty.
    """
    if nodes.empty:
        raise ValueError("`nodes` must not be empty.")

    missing = required_columns - set(nodes.columns)
    if missing:
        raise ValueError(f"`nodes` is missing required columns: {sorted(missing)}")

    if nodes[node_col].isna().any():
        raise ValueError("`nodes['code']` must not contain missing values.")

    if not nodes[node_col].is_unique:
        duplicated = nodes.loc[nodes["code"].duplicated(), "code"].tolist()
        raise ValueError(f"Node codes must be unique. Duplicates: {duplicated}")


# ---------------------------------------------------------------------------
# Geometry and sampling utilities
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two coordinates.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point, in degrees.
    lat2, lon2 : float
        Latitude and longitude of the second point, in degrees.

    Returns
    -------
    float
        Great-circle distance in kilometers.
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = (
        sin(dphi / 2.0) ** 2
        + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1.0 - a))


def build_distance_matrix(nodes: pd.DataFrame, node_col:str="iso3") -> pd.DataFrame:
    """
    Build the pairwise great-circle distance matrix between nodes.

    Parameters
    ----------
    nodes : pd.DataFrame
        Node table containing at least `code`, `lat`, and `lon`.

    Returns
    -------
    pd.DataFrame
        Square distance matrix indexed and columned by node code.
    """

    codes = nodes[node_col].to_numpy()
    coords = nodes.set_index(node_col)[["lat", "lon"]].astype(float)

    lat = np.radians(coords["lat"].to_numpy())
    lon = np.radians(coords["lon"].to_numpy())

    # These are used to build the N^2 distance pairs in vectorized form
    lat1 = lat[:, None]
    lat2 = lat[None, :]
    lon1 = lon[:, None]
    lon2 = lon[None, :]

    dphi = lat2 - lat1
    dlambda = lon2 - lon1

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2.0) ** 2
    )
    dist = 2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    np.fill_diagonal(dist, 0.0)

    return pd.DataFrame(dist, index=codes, columns=codes)


def gaussian_max_sample_vec(
    mu: float,
    sigma: float,
    n: int,
    size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample the maximum of `n` i.i.d. Gaussian random variables.

    If X_1, ..., X_n ~ N(mu, sigma^2), this function returns `size`
    samples from max(X_1, ..., X_n) using inverse-transform sampling.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution. Must be non-negative.
    n : int
        Number of Gaussian draws whose maximum is considered. Values < 1
        are clipped to 1.
    size : int
        Number of samples to generate.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape `(size,)` containing samples of the maximum.

    Raises
    ------
    ValueError
        If `sigma < 0` or `size < 0`.
    """
    if sigma < 0:
        raise ValueError("`sigma` must be non-negative.")
    if size < 0:
        raise ValueError("`size` must be non-negative.")

    n = max(int(n), 1)

    if sigma == 0:
        return np.full(size, float(mu), dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(size=size)
    q = np.exp(np.log(u) / n)  # equivalent to u ** (1 / n)
    return mu + sigma * stats.norm.ppf(q)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadiationRunResult:
    """
    Container for the output of the Models.
    """
    distance_matrix: pd.DataFrame    
    probability_matrix: pd.DataFrame
    predictions: pd.DataFrame
    intervening_population_matrix: Optional[pd.DataFrame] = None

    def write_predictions(self, filename: str) -> None:
        if Path(filename).exists():
            raise Exception("File already exists")
        else:
            self.predictions.to_csv(filename, index=False)


# ---------------------------------------------------------------------------
# Feature-Enriched Radiation Model
# ---------------------------------------------------------------------------

class FERM:
    """
    Feature-Enriched Radiation Model (FERM).

    This implementation estimates row-wise destination choice probabilities
    through Monte Carlo sampling. It follows the matrix formulation in which
    an attractiveness matrix Sigma centers both sides of the latent Gaussian
    sampling process:

    - Sigma[i, i] centers the threshold distribution for origin i.
    - Sigma[i, j] centers the offer distribution for corridor i -> j.

    For each origin i, a fixed number of particles is generated, each endowed
    with an origin-specific absorption threshold. Destinations are then scanned
    in order of increasing distance, and a particle is absorbed by the first
    destination whose sampled attractiveness exceeds the particle threshold.

    Notes
    -----
    The returned matrix is row-normalized and should be interpreted as an
    estimated origin-destination probability matrix rather than absolute flows.
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        flows: pd.DataFrame,
        features: pd.DataFrame | np.ndarray,
        node_col: str = "iso3",
    ) -> None:
        """
        Parameters
        ----------
        nodes : pd.DataFrame
            Node table containing at least:
            - `code` : unique node identifier
            - `population` : node population
            - `lat` : latitude in degrees
            - `lon` : longitude in degrees
        flows : pd.DataFrame
            Observed flows
        features : pd.DataFrame 
            Square attractiveness matrix Sigma. Rows are origins and columns
            are destinations. The index/columns must match `nodes['iso3']` if
            a DataFrame is provided. 
        """
        # validate_nodes(nodes, {"code", "iso3", "population", "lat", "lon"})
        self.nodes: pd.DataFrame = nodes.copy()
        self.flows = flows        
        self.features = self._prepare_attractiveness_matrix(features, node_col=node_col)
        self.distance_matrix = build_distance_matrix(self.nodes, node_col=node_col)   
        self.node_col = node_col

    def _prepare_attractiveness_matrix(
        self,
        features: pd.DataFrame,
        node_col: str = "iso3",
    ) -> pd.DataFrame:
        """
        Validate and align the Sigma attractiveness matrix.
        """
        codes = self.nodes[node_col].tolist()        
        missing_rows = set(codes) - set(features.index)
        missing_cols = set(codes) - set(features.columns)
        if missing_rows or missing_cols:
            raise ValueError(
                "`features` must contain all node ISO3 codes as both rows "
                f"and columns. Missing rows: {sorted(missing_rows)}; "
                f"missing columns: {sorted(missing_cols)}"
            )
        Sigma = features.loc[codes, codes].astype(float).copy()

        if Sigma.isna().any().any():
            raise ValueError("`features` contains missing values.")
        
        return Sigma

    def run(
        self,
        num_particles: int = 300,
        sigma: float = 0.15,
        origin_col: str = "country_from",
        dest_col: str = "country_to",
        flow_col: str = "num_migrants",
        pred_col: str = "predicted_migrants",
        verbose: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> RadiationRunResult:
        """
        Estimate the FERM origin-destination probability matrix.

        Parameters
        ----------
        num_particles : int, default=300
            Number of Monte Carlo particles sampled per origin.
        sigma : float, default=0.15
            Shared standard deviation of the Gaussian sampling kernel used for
            both thresholds and offers.
        verbose : bool, default=False
            If True, print the current origin code while processing.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        pd.DataFrame
            Origin-destination probability matrix.
        """
        if num_particles <= 0:
            raise ValueError("`num_particles` must be strictly positive.")

        if sigma < 0:
            raise ValueError("`sigma` must be non-negative.")

        if rng is None:
            rng = np.random.default_rng()

        nodes = self.nodes.set_index(self.node_col)
        populations = nodes["population"].round().clip(lower=1).astype(int)
        Sigma = self.features
        D = self.distance_matrix

        probabilities = pd.DataFrame(0.0, index=nodes.index, columns=nodes.index)
        
        for i in tqdm(range(len(nodes.index))):
            origin = nodes.index[i]
        #for origin in nodes.index:
            if verbose: print(f"Computing node {origin}")

            origin_population = populations[origin]
            threshold_center = Sigma.loc[origin, origin]

            origin_thresholds = gaussian_max_sample_vec(
                mu=threshold_center,
                sigma=sigma,
                n=origin_population,
                size=num_particles,
                rng=rng,
            )

            destinations = [d for d in D.loc[origin].sort_values().index if d != origin]

            assigned = np.zeros(num_particles, dtype=bool)
            counts = pd.Series(0.0, index=nodes.index)

            for destination in destinations:
                if assigned.all():
                    break

                destination_population = populations[destination]
                offer_center = Sigma.loc[origin, destination]

                remaining = (~assigned).sum()

                destination_attractiveness = gaussian_max_sample_vec(
                    mu=offer_center,
                    sigma=sigma,
                    n=destination_population,
                    size=remaining,
                    rng=rng,
                )

                winners_local = (
                    destination_attractiveness > origin_thresholds[~assigned]
                )

                if np.any(winners_local):
                    winners_global = np.where(~assigned)[0][winners_local]
                    counts[destination] += len(winners_global)
                    assigned[winners_global] = True

            total_assigned = counts.sum()
            if total_assigned > 0:
                probabilities.loc[origin] = counts / total_assigned

        predictions = predicted_flows_from_probabilities(
            self.flows, 
            probabilities, 
            nodes=nodes,
            origin_col=origin_col,
            dest_col=dest_col,
            flow_col=flow_col,
            pred_col=pred_col,
            eps = DEFAULT_EPS,
            )

        return RadiationRunResult(
            distance_matrix=D,
            probability_matrix=probabilities,
            predictions=predictions,
        )



# ---------------------------------------------------------------------------
# Classical Radiation Model
# ---------------------------------------------------------------------------

class RM:
    """
    Classical Radiation Model (RM).

    The model predicts destination choice probabilities from node populations
    and intervening opportunities. For origin i and destination j, the
    unnormalized probability is proportional to:

        m_i * n_j / [(m_i + s_ij) * (m_i + n_j + s_ij)]

    where:
    - m_i is the origin population,
    - n_j is the destination population,
    - s_ij is the total population within radius d_ij from origin i,
      excluding i and j.
    """

    def __init__(self,
                 nodes: pd.DataFrame,
                 flows: pd.DataFrame,
                 eps: float = DEFAULT_EPS,
                 node_col: str = "iso3",
                 distance_matrix: Optional[pd.DataFrame | np.ndarray] = None) -> None:
        """
        Parameters
        ----------
        nodes : pd.DataFrame
            Node table containing at least:
            - `code`
            - `population`
            - `lat`
            - `lon`
        flows : pd.DataFrame
            Observed flows
        eps : float, default=1.0
            Small constant used in logarithmic error diagnostics.
        distance_matrix : pd.DataFrame or np.ndarray, optional
            Square distance matrix used to order destinations from each origin.
            Rows are origins and columns are destinations. If omitted, distances
            are computed from `nodes['lat']` and `nodes['lon']`.
        """
        validate_nodes(nodes, {node_col, "population", "lat", "lon"}, node_col=node_col)
        self.node_col: str = node_col
        self.nodes: pd.DataFrame = nodes.copy()
        self.flows: pd.DataFrame = flows.copy()
        self.eps: float = float(eps)
        self.distance_matrix = build_distance_matrix(self.nodes,node_col=node_col)


    @staticmethod
    def build_intervening_population_matrix(
        nodes: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        node_col:str="iso3",
        verbose:bool=True,
    ) -> pd.DataFrame:
        """
        Compute the intervening-population matrix S.

        For each pair (i, j), S[i, j] is the total population of all nodes k
        such that d(i, k) < d(i, j), excluding i and j.

        Parameters
        ----------
        nodes : pd.DataFrame
            Node table with columns `code` and `population`.
        distance_matrix : pd.DataFrame
            Pairwise distance matrix indexed by node code.

        Returns
        -------
        pd.DataFrame
            Intervening-population matrix.
        """
        validate_nodes(nodes, {node_col, "population"}, node_col=node_col)

        codes = nodes[node_col].tolist()
        populations = nodes.set_index(node_col)["population"].astype(float)
        S = pd.DataFrame(0.0, index=codes, columns=codes)
        #for origin in codes:
        for i in tqdm(range(len(codes))):
            #if verbose: print(f"Computing node {origin}")
            origin = codes[i]
            d_origin = distance_matrix.loc[origin]

            for destination in codes:
                if origin == destination:
                    continue

                dij = d_origin[destination]
                mask = d_origin < dij
                mask.loc[origin] = False
                mask.loc[destination] = False

                S.loc[origin, destination] = populations.loc[mask.index[mask]].sum()

        return S

    def radiation_probabilities(
        self,
        nodes: pd.DataFrame,
        intervening_population_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the row-wise destination probabilities of the radiation model.

        Parameters
        ----------
        nodes : pd.DataFrame
            Node table with columns `iso3` codes and `population`.
        intervening_population_matrix : pd.DataFrame
            Matrix S of intervening populations.

        Returns
        -------
        pd.DataFrame
            Probability matrix.
        """
        validate_nodes(nodes, {self.node_col, "population"}, node_col=self.node_col)

        populations = nodes.set_index(self.node_col)["population"].astype(float)
        codes = nodes[self.node_col].tolist()

        probabilities = pd.DataFrame(0.0, index=codes, columns=codes)

        for origin in codes:
            m_i = populations[origin]
            s_i = intervening_population_matrix.loc[origin]

            numerator = m_i * populations
            denominator = (m_i + s_i) * (m_i + populations + s_i)

            p_i = numerator / denominator.replace(0.0, np.nan)
            p_i.loc[origin] = 0.0
            p_i = p_i.fillna(0.0)

            row_sum = p_i.sum()
            if row_sum > 0:
                p_i = p_i / row_sum

            probabilities.loc[origin] = p_i

        return probabilities

    def run(
        self,
        renormalize: bool = True,
        origin_col: str = "country_from",
        dest_col: str = "country_to",
        flow_col: str = "num_migrants",
        pred_col: str = "predicted_migrants",
    ) -> RadiationRunResult:
        """
        Run the classical radiation model.

        Parameters
        ----------
        renormalize : bool, default=True
            If True, enforce row-wise normalization of the probability matrix.
        origin_col, dest_col, flow_col, pred_col : str
            Column names used when `flows` is provided.

        Returns
        -------
        RadiationRunResult
            Container with distance matrix, intervening-population matrix,
            probability matrix, and optional predictions table.
        """
        D = self.distance_matrix
        print("Building intervening-population")
        S = self.build_intervening_population_matrix(self.nodes, D, node_col=self.node_col)
        print("Building probabilities")
        P = self.radiation_probabilities(self.nodes, S)

        if renormalize:
            P = P.div(P.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

        predictions = None
        if self.flows is not None:
            predictions = predicted_flows_from_probabilities(
                flows=self.flows,
                P=P,
                nodes=self.nodes,
                origin_col=origin_col,
                dest_col=dest_col,
                flow_col=flow_col,
                pred_col=pred_col,
                eps=self.eps,
            )

        return RadiationRunResult(
            distance_matrix=D,
            probability_matrix=P,
            predictions=predictions,
        )


# ---------------------------------------------------------------------------
# Prediction and diagnostics
# ---------------------------------------------------------------------------

def add_error_columns(
    predictions: pd.DataFrame,
    observed_col: str = "num_migrants",
    predicted_col: str = "predicted_migrants",
    eps: float = DEFAULT_EPS,
) -> pd.DataFrame:
    """
    Add residual and logarithmic error diagnostics to a predictions table.

    Parameters
    ----------
    predictions : pd.DataFrame
        Table containing observed and predicted flow columns.
    observed_col : str, default='num_migrants'
        Name of the observed-flow column.
    predicted_col : str, default='predicted_migrants'
        Name of the predicted-flow column.
    eps : float, default=1.0
        Small additive constant used to stabilize logarithms.

    Returns
    -------
    pd.DataFrame
        Copy of the input table with additional error columns.
    """
    out = predictions.copy()

    residual = out[predicted_col] - out[observed_col]

    out["residual"] = residual
    out["signed_error"] = residual
    out["abs_error"] = np.abs(residual)
    out["log_observed"] = np.log10(out[observed_col] + eps)
    out["log_predicted"] = np.log10(out[predicted_col] + eps)
    out["log_ratio"] = np.log10(
        (out[predicted_col] + eps) / (out[observed_col] + eps)
    )
    out["abs_log_ratio"] = np.abs(out["log_ratio"])

    return out


def predicted_flows_from_probabilities(
    flows: pd.DataFrame,
    P: pd.DataFrame,
    nodes: pd.DataFrame,
    origin_col: str = "country_from",
    dest_col: str = "country_to",
    flow_col: str = "num_migrants",
    pred_col: str = "predicted_migrants",
    eps: float = DEFAULT_EPS,
    add_error: bool = False,
) -> pd.DataFrame:
    """
    Convert a probability matrix into predicted flows by matching
    observed total outflows per origin.

    Parameters
    ----------
    flows : pd.DataFrame
        Observed origin-destination flow table.
    P : pd.DataFrame
        Origin-destination probability matrix.
    nodes : pd.DataFrame, optional
        Node metadata used to map codes to country names.
    origin_col, dest_col, flow_col, pred_col : str
        Column names in the input/output tables.
    eps : float, default=1.0
        Small constant used in logarithmic error diagnostics.

    Returns
    -------
    pd.DataFrame
        Comparison table containing observed and predicted flows and
        error diagnostics.
    """
    flows = flows.groupby([origin_col, dest_col], as_index=False)[flow_col].sum()
    outflow = flows.groupby(origin_col)[flow_col].sum()

    if {"iso3"}.issubset(nodes.columns):
        outflow.index = outflow.index.map(normalize_country_code_to_iso3)
    
    total_outflow = outflow.reindex(P.index).fillna(0.0).to_numpy()

    predicted_matrix = total_outflow[:, None] * P.to_numpy()
    predicted = pd.DataFrame(predicted_matrix, index=P.index, columns=P.columns)
    predicted.index.name = origin_col
    predicted.columns.name = dest_col

    predicted = predicted.stack().reset_index(name=pred_col)
    predicted = predicted[predicted[origin_col] != predicted[dest_col]].copy()

    predictions = flows.merge(predicted, on=[origin_col, dest_col], how="outer")

    if {"iso3"}.issubset(nodes.columns):
        predictions["country_from_name"] = predictions[origin_col].map(iso3_to_country)
        predictions["country_to_name"] = predictions[dest_col].map(iso3_to_country)

    predictions[flow_col] = predictions[flow_col].fillna(0.0)
    predictions[pred_col] = predictions[pred_col].fillna(0.0)   
    predictions[pred_col] = predictions[pred_col].round()

    if add_error:
        predictions = add_error_columns(
            predictions=predictions,
            observed_col=flow_col,
            predicted_col=pred_col,
            eps=eps,
        )

    return predictions


def run_parallel(*args, **kwargs):
    """
    Compatibility wrapper for the legacy raster FERM API.

    The implementation lives in `ferm.cluster_runner` and returns a row-normalized
    conditional destination-probability matrix.
    """
    from ferm.cluster_runner import run_parallel as _run_parallel

    return _run_parallel(*args, **kwargs)
