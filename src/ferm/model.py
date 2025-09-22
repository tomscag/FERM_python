import numpy as np
import scipy.sparse as sp
import rioxarray
from multiprocessing import Pool
from ferm.sampling import gaussian_distribution_max
from ferm.utils import parse_lat_lon, precise_the_mask
from ferm.distance import wrap_geodist

# Globals (to be set inside `initializer`)
array_niche = None
x_pop = None
y_pop = None
mask = None
mask_x = None
mask_y = None
points = None

def FERM(path_niche_array: str, path_x: str, path_y: str, path_pop: str,
         nb_particules: int = 100, sigma: float = 1.0,
         save_path: str = "pop=test_sparse_mobility_mat.npz") -> None:
    """
    Run the FERM simulation and save the resulting mobility matrix.

    Parameters
    ----------
    path_niche_array : str
        Path to the .npy file containing niche suitability array.
    path_x : str
        Path to the .npy file containing longitude coordinates.
    path_y : str
        Path to the .npy file containing latitude coordinates.
    path_pop : str
        Path to the raster file containing population data.
    nb_particules : int, optional
        Number of walkers per origin location. Default is 100.
    sigma : float, optional
        Standard deviation used for Gaussian absorption. Default is 1.0.
    save_path : str, optional
        Path to save the resulting sparse matrix. Default is "pop=test_sparse_mobility_mat.npz".
    """
    global mask_x, mask_y, points

    df_pop = rioxarray.open_rasterio(path_pop)
    array_niche_local = np.load(path_niche_array)
    x_pop_local = np.load(path_x)
    y_pop_local = np.load(path_y)

    array_niche_local = precise_the_mask(-40, 47, 4, x_pop_local, y_pop_local, array_niche_local)
    mask_local = np.where(array_niche_local != 0)
    mask_x, mask_y, points = parse_lat_lon(mask_local, x_pop_local, y_pop_local)

    P_final = sp.lil_matrix((len(mask_x), len(mask_x)))

    for i in range(len(mask_x)):
        p1 = np.array(points[i])
        x_current = mask_local[0][i]
        y_current = mask_local[1][i]
        pop_i = int(df_pop.sel(x=p1[1], y=p1[0]).values[0])

        if pop_i < 1:
            continue

        distances = np.array([wrap_geodist(p1, points[j]) for j in range(len(points))])
        index_sort = np.argsort(distances)[1:]  # Exclude self

        for _ in range(nb_particules):
            mu = array_niche_local[x_current][y_current] + 1e-6
            absorption_i = gaussian_distribution_max(sigma, mu, pop_i)

            for index in index_sort:
                p_dest = np.array(points[index])
                pop_j = int(df_pop.sel(x=p_dest[1], y=p_dest[0]).values[0])

                if pop_j < 1:
                    continue

                x_dest = mask_local[0][index]
                y_dest = mask_local[1][index]
                mu_j = array_niche_local[x_dest][y_dest] + 1e-6
                absorbance_j = gaussian_distribution_max(sigma, mu_j, pop_j)

                if absorbance_j > absorption_i:
                    P_final[i, index] += 1
                    break

    P_final = P_final.tocsr()
    P_final = P_final / nb_particules
    sp.save_npz(save_path, P_final)

def initializer():
    """Initialize global variables for multiprocessing pool workers."""
    global array_niche, x_pop, y_pop, mask, mask_x, mask_y, points
    pass

def FERM_multiprocessing(i: int, path_pop: str,
                          nb_particules: int, sigma: float) -> sp.lil_matrix:
    """
    Multiprocessing helper for computing a single row of the FERM matrix.

    Parameters
    ----------
    i : int
        Index of origin point.
    path_pop : str
        Path to the population raster.
    nb_particules : int
        Number of walkers per origin.
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
                 nb_particules: int = 100, sigma: float = 1.0,
                 n_processes: int = 12) -> sp.csr_matrix:
    """
    Launch parallel FERM computation.

    Parameters
    ----------
    path_niche_array : str
        Path to .npy niche data.
    path_x : str
        Path to longitude .npy file.
    path_y : str
        Path to latitude .npy file.
    path_pop : str
        Path to .tif population raster.
    nb_particules : int, optional
        Number of particles per origin.
    sigma : float, optional
        Standard deviation.
    n_processes : int, optional
        Number of parallel workers.

    Returns
    -------
    sp.csr_matrix
        Sparse mobility matrix.
    """
    global array_niche, x_pop, y_pop, mask, mask_x, mask_y, points

    array_niche = np.load(path_niche_array)
    x_pop = np.load(path_x)
    y_pop = np.load(path_y)
    mask = np.where(array_niche != 0)
    mask_x, mask_y, points = parse_lat_lon(mask, x_pop, y_pop)

    args = [(i, path_pop, nb_particules, sigma) for i in range(len(mask_x))]
    P_final = sp.lil_matrix((len(mask_x), len(mask_x)))

    with Pool(processes=n_processes, initializer=initializer) as pool:
        results = pool.starmap(FERM_multiprocessing, args)

    for i, row in enumerate(results):
        P_final[i] = row

    return P_final.tocsr() / nb_particules
