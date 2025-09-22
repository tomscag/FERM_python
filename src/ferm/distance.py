import numpy as np
from geopy.distance import geodesic
from geokernels.distance import geodist
from tqdm import tqdm

def wrap_geodist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute geodesic distance in kilometers between two points using geokernels.

    Parameters
    ----------
    a : np.ndarray
        Coordinates of the first point (latitude, longitude).
    b : np.ndarray
        Coordinates of the second point (latitude, longitude).

    Returns
    -------
    float
        Distance in kilometers.
    """
    return geodist(a, b, metric='km')

def distance_matrix(mask: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Compute a symmetric geodesic distance matrix between masked coordinates.

    Parameters
    ----------
    mask : tuple of np.ndarray
        Tuple containing arrays of x and y indices (latitude and longitude).

    Returns
    -------
    np.ndarray
        Symmetric matrix of pairwise geodesic distances (in kilometers).
    """
    mask_x, mask_y = mask
    n = len(mask_x)
    dist_mat = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing geodesic distances"):
        for j in range(i, n):
            x1, y1 = mask_x[i], mask_y[i]
            x2, y2 = mask_x[j], mask_y[j]
            dist = geodesic((x1, y1), (x2, y2)).km
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist  # symmetry

    return dist_mat
