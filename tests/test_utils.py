import numpy as np
import pytest
from ferm.utils import parse_lat_lon, precise_the_mask


def test_parse_lat_lon_shape():
    x_pop = np.array([10, 20, 30])
    y_pop = np.array([40, 50, 60])
    array_niche = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    mask = np.where(array_niche != 0)
    mask_x, mask_y, points = parse_lat_lon(mask, x_pop, y_pop)

    assert len(mask_x) == len(mask_y) == len(points)
    assert all(len(p) == 2 for p in points)


def test_precise_the_mask_bounds():
    x_pop = np.array([-50, -20, 10, 50])   # index 0 and 3 are out
    y_pop = np.array([0, 10, 30, 60])      # index 0 and 3 are out
    array_niche = np.ones((len(x_pop), len(y_pop)))

    masked = precise_the_mask(
        xmin=-40, xmax=30, ymin=5,
        x_pop=x_pop, y_pop=y_pop,
        array_niche=array_niche.copy()
    )

    expected_zero_rows = [0, 3]  # -50, 50
    expected_zero_cols = [0, 3]  # 0, 60

    # Only these must be zeroed
    for r in expected_zero_rows:
        assert np.allclose(masked[r, :], 0, atol=1e-8)
    for c in expected_zero_cols:
        assert np.allclose(masked[:, c], 0, atol=1e-8)

    # Double-check that inner values stay
    assert masked[1, 1] == 1
    assert masked[1, 2] == 1
