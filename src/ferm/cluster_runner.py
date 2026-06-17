import numpy as np
import scipy.sparse as sp
import rioxarray
import os
from multiprocessing import Pool
from ferm.sampling import gaussian_distribution_max
from ferm.utils import parse_lat_lon
from ferm.distance import wrap_geodist

# Globals for multiprocessing
array_niche = None
x_pop = None
y_pop = None
mask = None
mask_x = None
mask_y = None
points = None

def initializer():
    """Initialize global variables for multiprocessing pool workers."""
    global array_niche, x_pop, y_pop, mask, mask_x, mask_y, points
    pass

def normalize_assigned_rows(counts: sp.spmatrix) -> sp.csr_matrix:
    """
    Convert assigned-destination counts into conditional destination probabilities.

    Rows with no assigned particles remain zero. Nonzero rows sum to one, so the
    resulting matrix is P(destination | particle leaves origin).
    """
    counts = counts.tocsr()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    inverse_row_sums = np.zeros_like(row_sums, dtype=float)
    inverse_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    return sp.diags(inverse_row_sums) @ counts

def FERM_multiprocessing(i: int, path_pop: str, nb_particules: int, sigma: float) -> sp.lil_matrix:
    """
    Variant of FERM row computation with adjustments for cluster execution.

    Uses +1e-6 offset to prevent zero mean values and relies on sorted distance traversal.

    Parameters
    ----------
    i : int
        Index of origin point.
    path_pop : str
        Path to population raster.
    nb_particules : int
        Number of particles per origin.
    sigma : float
        Standard deviation of the Gaussian absorption.

    Returns
    -------
    sp.lil_matrix
        Sparse matrix with a single row of mobility transitions.
    """
    global array_niche, mask, mask_x, points
    df_pop = rioxarray.open_rasterio(path_pop)
    p1 = np.array(points[i])
    x_current = mask[0][i]
    y_current = mask[1][i]
    pop_i = int(df_pop.sel(x=p1[1], y=p1[0]).values[0])

    row = sp.lil_matrix((1, len(mask_x)))
    if pop_i < 1:
        return row

    distances = np.array([wrap_geodist(p1, points[j]) for j in range(len(points))])
    index_sort = np.argsort(distances)[1:]

    for _ in range(nb_particules):
        mu = array_niche[x_current][y_current] + 1e-6
        absorption_i = gaussian_distribution_max(sigma, mu, pop_i)

        for index in index_sort:
            p_dest = np.array(points[index])
            pop_j = int(df_pop.sel(x=p_dest[1], y=p_dest[0]).values[0])

            if pop_j < 1:
                continue

            x_dest = mask[0][index]
            y_dest = mask[1][index]
            mu_j = array_niche[x_dest][y_dest] + 1e-6
            absorbance_j = gaussian_distribution_max(sigma, mu_j, pop_j)

            if absorbance_j > absorption_i:
                row[0, index] += 1
                break

    return row

def run_parallel(path_niche_array: str, path_x: str, path_y: str, path_pop: str,
                 nb_particules: int = 500, sigma: float = 1.0,
                 n_processes: int | None = None,
                 chunksize: int = 1) -> sp.csr_matrix:
    """
    Launch FERM simulation and return conditional destination probabilities.

    Parameters
    ----------
    path_niche_array : str
        Path to .npy file of niche array.
    path_x : str
        Path to .npy file of longitude coordinates.
    path_y : str
        Path to .npy file of latitude coordinates.
    path_pop : str
        Path to .tif file of population raster.
    nb_particules : int, optional
        Number of particles per origin. Default is 500.
    sigma : float, optional
        Standard deviation. Default is 1.0.
    n_processes : int, optional
        Number of parallel workers. Default uses all available CPUs.
    chunksize : int, optional
        Chunk size for multiprocessing. Default is 1.
    """
    global array_niche, x_pop, y_pop, mask, mask_x, mask_y, points

    array_niche = np.load(path_niche_array)
    x_pop = np.load(path_x)
    y_pop = np.load(path_y)
    mask = np.where(array_niche != 0)
    mask_x, mask_y, points = parse_lat_lon(mask, x_pop, y_pop)

    args = [(i, path_pop, nb_particules, sigma) for i in range(len(mask_x))]
    P_final = sp.lil_matrix((len(mask_x), len(mask_x)))

    if n_processes is None:
        n_processes = os.cpu_count()

    if n_processes == 1:
        results = [FERM_multiprocessing(*arg) for arg in args]
    else:
        with Pool(processes=n_processes, initializer=initializer) as pool:
            print("Running with", n_processes, "cores and chunksize=", chunksize)
            results = pool.starmap(FERM_multiprocessing, args, chunksize=chunksize)

    for i, row in enumerate(results):
        P_final[i] = row

    return normalize_assigned_rows(P_final)


def run_cluster(path_niche_array: str, path_x: str, path_y: str, path_pop: str,
                nb_particules: int = 500, sigma: float = 1.0,
                save_path: str = "mobility_sigma=1_chunksize=1.npz",
                chunksize: int = 1) -> None:
    """
    Launch FERM simulation using cluster-style multiprocessing with fine granularity.

    Parameters
    ----------
    path_niche_array : str
        Path to .npy file of niche array.
    path_x : str
        Path to .npy file of longitude coordinates.
    path_y : str
        Path to .npy file of latitude coordinates.
    path_pop : str
        Path to .tif file of population raster.
    nb_particules : int, optional
        Number of particles per origin. Default is 500.
    sigma : float, optional
        Standard deviation. Default is 1.0.
    save_path : str, optional
        File name to save sparse mobility matrix. Default is 'mobility_sigma=1_chunksize=1.npz'.
    chunksize : int, optional
        Chunk size for multiprocessing. Default is 1.
    """
    P_final = run_parallel(
        path_niche_array=path_niche_array,
        path_x=path_x,
        path_y=path_y,
        path_pop=path_pop,
        nb_particules=nb_particules,
        sigma=sigma,
        n_processes=os.cpu_count(),
        chunksize=chunksize,
    )
    sp.save_npz(save_path, P_final)
