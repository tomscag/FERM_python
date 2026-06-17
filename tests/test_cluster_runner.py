import numpy as np
import scipy.sparse as sp

from ferm.cluster_runner import normalize_assigned_rows


def test_normalize_assigned_rows_conditions_on_assigned_particles():
    counts = sp.csr_matrix(
        [
            [0, 300, 200, 100],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )

    P = normalize_assigned_rows(counts)

    assert np.allclose(P.toarray()[0], [0.0, 0.5, 1.0 / 3.0, 1.0 / 6.0])
    assert np.isclose(P.toarray()[0].sum(), 1.0)
    assert np.allclose(P.toarray()[1], [0.0, 0.0, 0.0, 0.0])
