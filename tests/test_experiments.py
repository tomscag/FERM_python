import numpy as np
import pandas as pd

from ferm.experiments import (
    GeographicScope,
    build_combined_sigma,
    build_destination_sigma,
    build_relational_sigma,
    complete_matrix_codes,
    description_length_bits,
    ExperimentConfig,
    FeatureSpec,
    fit_matrix_normalization,
    fit_vector_normalization,
    filter_flows_for_scope,
    filter_nodes_and_flows_by_feature_coverage,
    load_gravity_country_metadata,
    parse_scope,
    route_metrics_vs_rm,
)


def test_destination_sigma_uses_destination_off_diagonal_and_origin_diagonal():
    nodes = pd.DataFrame({"iso3": ["AAA", "BBB", "CCC"]})
    feature = pd.Series({"AAA": 10.0, "BBB": 20.0, "CCC": 30.0})

    sigma = build_destination_sigma(nodes, feature, normalization="zscore_log")

    assert list(sigma.index) == ["AAA", "BBB", "CCC"]
    assert np.isclose(sigma.loc["AAA", "BBB"], sigma.loc["BBB", "BBB"])
    assert np.isclose(sigma.loc["CCC", "AAA"], sigma.loc["AAA", "AAA"])


def test_destination_sigma_can_use_prefit_calibration_normalization():
    nodes_cal = pd.DataFrame({"iso3": ["AAA", "BBB"]})
    nodes_test = pd.DataFrame({"iso3": ["AAA", "BBB", "CCC"]})
    feature = pd.Series({"AAA": 10.0, "BBB": 20.0, "CCC": 1_000.0})

    params = fit_vector_normalization(
        feature.reindex(nodes_cal["iso3"]),
        "zscore_log",
    )
    sigma = build_destination_sigma(
        nodes_test,
        feature,
        normalization="zscore_log",
        normalization_params=params,
    )

    expected_ccc = (
        np.log1p(feature.loc["CCC"]) - params["center"]
    ) / params["scale"]
    assert np.isclose(sigma.loc["AAA", "CCC"], expected_ccc)
    assert sigma.loc["AAA", "CCC"] > 1.0


def test_relational_sigma_has_neutral_diagonal():
    nodes = pd.DataFrame({"iso3": ["AAA", "BBB"]})
    raw = pd.DataFrame(
        [[10.0, 2.0], [7.0, 4.0]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )

    sigma = build_relational_sigma(nodes, raw, normalization="minmax_signed")

    assert sigma.loc["AAA", "AAA"] == 0.0
    assert sigma.loc["BBB", "BBB"] == 0.0
    assert sigma.loc["AAA", "BBB"] != sigma.loc["BBB", "AAA"]


def test_relational_sigma_can_use_prefit_offdiagonal_calibration_normalization():
    nodes_cal = pd.DataFrame({"iso3": ["AAA", "BBB"]})
    nodes_test = pd.DataFrame({"iso3": ["AAA", "BBB", "CCC"]})
    raw = pd.DataFrame(
        [
            [np.nan, 2.0, 100.0],
            [4.0, np.nan, 200.0],
            [300.0, 400.0, np.nan],
        ],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )

    params = fit_matrix_normalization(
        raw.reindex(index=nodes_cal["iso3"], columns=nodes_cal["iso3"]),
        "minmax_signed",
    )
    sigma = build_relational_sigma(
        nodes_test,
        raw,
        normalization="minmax_signed",
        normalization_params=params,
    )

    assert params["min"] == 2.0
    assert params["range"] == 2.0
    assert sigma.loc["AAA", "BBB"] == -1.0
    assert sigma.loc["BBB", "AAA"] == 1.0
    assert sigma.loc["CCC", "BBB"] > 1.0


def test_combined_sigma_keeps_destination_diagonal():
    nodes = pd.DataFrame({"iso3": ["AAA", "BBB"]})
    destination = pd.DataFrame(
        [[1.0, 2.0], [1.0, 2.0]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )
    relational = pd.DataFrame(
        [[0.0, 0.5], [-0.5, 0.0]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )

    sigma = build_combined_sigma(destination, relational)

    assert sigma.loc["AAA", "AAA"] == destination.loc["AAA", "AAA"]
    assert sigma.loc["BBB", "BBB"] == destination.loc["BBB", "BBB"]
    assert sigma.loc["AAA", "BBB"] == 2.5


def test_filter_flows_for_corridor_scope():
    countries = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "CCC"],
            "continent": ["Europe", "Africa", "Asia"],
        }
    )
    flows = pd.DataFrame(
        {
            "country_from": ["AAA", "BBB", "CCC", "AAA"],
            "country_to": ["BBB", "AAA", "AAA", "CCC"],
            "num_migrants": [10, 20, 30, 40],
        }
    )

    scoped_flows, scoped_countries = filter_flows_for_scope(
        flows,
        countries,
        GeographicScope.corridor(("Europe",), ("Africa",)),
    )

    assert scoped_flows[["country_from", "country_to"]].values.tolist() == [["AAA", "BBB"]]
    assert set(scoped_countries["iso3"]) == {"AAA", "BBB"}


def test_filter_world_scope_does_not_require_continent():
    countries = pd.DataFrame({"iso3": ["AAA", "BBB"], "code": ["AA", "BB"]})
    flows = pd.DataFrame(
        {
            "country_from": ["AAA"],
            "country_to": ["BBB"],
            "num_migrants": [10],
        }
    )

    scoped_flows, scoped_countries = filter_flows_for_scope(
        flows,
        countries,
        GeographicScope.world(),
    )

    assert len(scoped_flows) == 1
    assert set(scoped_countries["iso3"]) == {"AAA", "BBB"}


def test_load_gravity_country_metadata_fills_model_columns(tmp_path):
    path = tmp_path / "countries.csv"
    pd.DataFrame(
        {
            "iso3": ["aaa", "bbb"],
            "iso2": ["aa", None],
            "country": ["Country A", "Country B"],
        }
    ).to_csv(path, index=False)

    metadata = load_gravity_country_metadata(path)

    assert metadata.loc[metadata["iso3"] == "AAA", "code"].iloc[0] == "AA"
    assert metadata.loc[metadata["iso3"] == "BBB", "code"].iloc[0] == "BBB"
    assert {"continent", "lat", "lon"}.issubset(metadata.columns)
    assert "XKX" in set(metadata["iso3"])


def test_filter_nodes_and_flows_by_required_gdp_coverage(tmp_path):
    data_dir = tmp_path / "data"
    feature_dir = data_dir / "features"
    feature_dir.mkdir(parents=True)
    gdp_path = feature_dir / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_46.csv"
    gdp_path.write_text(
        "header\nheader\nheader\nheader\n"
        "Country Name,Country Code,2018\n"
        "Country A,AAA,1000.0\n"
        "Country B,BBB,\n",
        encoding="utf-8",
    )

    nodes = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "code": ["AA", "BB"],
            "population": [100.0, 100.0],
        }
    )
    flows = pd.DataFrame(
        {
            "country_from": ["AAA", "BBB"],
            "country_to": ["BBB", "AAA"],
            "num_migrants": [1.0, 2.0],
        }
    )
    config = ExperimentConfig(
        data_dir=data_dir,
        feature_specs=(FeatureSpec(name="gdp", kind="gdp"),),
    )

    filtered_nodes, scoped, calibration, test, dropped = (
        filter_nodes_and_flows_by_feature_coverage(
            nodes,
            flows,
            flows,
            flows,
            config,
        )
    )

    assert filtered_nodes["iso3"].tolist() == ["AAA"]
    assert scoped.empty
    assert calibration.empty
    assert test.empty
    assert dropped == {"missing_gdp_per_capita_2018": ["BBB"]}


def test_route_metrics_use_observed_routes_only():
    rm = pd.DataFrame(
        {
            "country_from": ["AAA", "AAA"],
            "country_to": ["BBB", "CCC"],
            "num_migrants": [10.0, 0.0],
            "abs_log_ratio": [0.5, 0.0],
            "log_observed": [1.0, 0.0],
            "log_predicted": [0.5, 0.0],
        }
    )
    model = pd.DataFrame(
        {
            "country_from": ["AAA", "AAA"],
            "country_to": ["BBB", "CCC"],
            "num_migrants": [10.0, 0.0],
            "abs_log_ratio": [0.2, 99.0],
            "log_observed": [1.0, 0.0],
            "log_predicted": [0.8, 99.0],
        }
    )

    metrics = route_metrics_vs_rm(rm, model, observed_routes_only=True)

    assert metrics["n_eval_routes"] == 1.0
    assert metrics["median_improvement_abs_log"] == 0.3


def test_description_length_bits_reads_probabilities_for_observed_routes():
    flows = pd.DataFrame(
        {
            "country_from": ["AAA", "AAA"],
            "country_to": ["BBB", "CCC"],
            "num_migrants": [2.0, 1.0],
        }
    )
    probabilities = pd.DataFrame(
        [[0.0, 0.25, 0.75]],
        index=["AAA"],
        columns=["AAA", "BBB", "CCC"],
    )

    bits = description_length_bits(flows, probabilities)

    assert np.isclose(bits, -(2.0 * np.log2(0.25) + np.log2(0.75)))


def test_parse_scope_labels():
    assert parse_scope("world").label == "world"
    assert parse_scope("within:Europe").label == "within_europe"
    assert parse_scope("corridor:Europe->Africa").label == "europe_to_africa"


def test_complete_matrix_codes_drops_sparse_problem_code():
    matrix = pd.DataFrame(
        [
            [0.0, 1.0, np.nan],
            [1.0, 0.0, np.nan],
            [np.nan, np.nan, 0.0],
        ],
        index=["AAA", "BBB", "BAD"],
        columns=["AAA", "BBB", "BAD"],
    )

    assert complete_matrix_codes(matrix) == ["AAA", "BBB"]
