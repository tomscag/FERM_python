import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import scipy.sparse as sp
from pathlib import Path
from ferm.model import FERM, RM, run_parallel
import pytest

@pytest.fixture
def fake_data(tmp_path):
    # Coordinate axes
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])
    
    # Save coords
    np.save(tmp_path / "x_test.npy", x)
    np.save(tmp_path / "y_test.npy", y)

    # Save niche
    niche = np.ones((4, 4))
    np.save(tmp_path / "niche_test.npy", niche)

    # Create population GeoTIFF
    pop_data = xr.DataArray(
        data=np.ones((1, 4, 4), dtype=np.uint8),
        dims=("band", "y", "x"),
        coords={"band": [1], "x": x, "y": y},
    )
    pop_data.rio.write_crs("EPSG:4326", inplace=True)
    pop_path = tmp_path / "test_pop.tif"
    pop_data.rio.to_raster(pop_path)

    return {
        "niche": str(tmp_path / "niche_test.npy"),
        "x": str(tmp_path / "x_test.npy"),
        "y": str(tmp_path / "y_test.npy"),
        "pop": str(pop_path),
    }


def test_run_parallel_structure(fake_data):
    P = run_parallel(
        path_niche_array=fake_data["niche"],
        path_x=fake_data["x"],
        path_y=fake_data["y"],
        path_pop=fake_data["pop"],
        nb_particules=10,
        sigma=1.0,
        n_processes=1
    )

    assert isinstance(P, sp.csr_matrix)
    assert P.shape[0] == P.shape[1]
    assert P.shape[0] > 0


def test_country_models_accept_external_distance_matrix_without_coordinates():
    nodes = pd.DataFrame(
        {
            "code": ["AA", "BB", "CC"],
            "iso3": ["AAA", "BBB", "CCC"],
            "country_name": ["A", "B", "C"],
            "population": [1000, 2000, 1500],
        }
    )
    flows = pd.DataFrame(
        {
            "country_from": ["AAA", "AAA", "BBB"],
            "country_to": ["BBB", "CCC", "CCC"],
            "num_migrants": [10.0, 5.0, 3.0],
        }
    )
    distance = pd.DataFrame(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    features = pd.DataFrame(
        0.0,
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )

    rm_result = RM(nodes=nodes, flows=flows, distance_matrix=distance).run()
    ferm_result = FERM(
        nodes=nodes,
        flows=flows,
        features=features,
        distance_matrix=distance,
    ).run(num_particles=20, sigma=1.0, rng=np.random.default_rng(1))

    assert rm_result.probability_matrix.shape == (3, 3)
    assert ferm_result.probability_matrix.shape == (3, 3)
    assert not set(rm_result.comparison["country_to_name"]).intersection(
        {"AAA", "BBB", "CCC"}
    )
    assert not set(ferm_result.comparison["country_to_name"]).intersection(
        {"AAA", "BBB", "CCC"}
    )
