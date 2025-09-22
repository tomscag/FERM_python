import numpy as np
import xarray as xr
import rioxarray
import scipy.sparse as sp
from pathlib import Path
from ferm.model import run_parallel
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
