import numpy as np

def parse_lat_lon(mask: tuple[np.ndarray, np.ndarray], x_pop: np.ndarray, y_pop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse latitude and longitude coordinates from a binary mask and population coordinates.

    Parameters
    ----------
    mask : tuple of np.ndarray
        Tuple of arrays indicating valid (x, y) indices where the mask is non-zero.
    x_pop : np.ndarray
        Array of longitude values.
    y_pop : np.ndarray
        Array of latitude values.

    Returns
    -------
    tuple of np.ndarray
        - mask_x: x-indices
        - mask_y: y-indices
        - points: array of shape (N, 2) with (latitude, longitude) pairs
    """
    mask_x, mask_y = mask
    coord_x = x_pop[mask_x]
    coord_y = y_pop[mask_y]
    points = np.array([(lat, lon) for lon, lat in zip(coord_x, coord_y)])
    return mask_x, mask_y, points

def precise_the_mask(xmin: float, xmax: float, ymin: float,
                     x_pop: np.ndarray, y_pop: np.ndarray,
                     array_niche: np.ndarray) -> np.ndarray:
    """
    Filter the niche array by applying spatial bounds on latitude and longitude.

    Parameters
    ----------
    xmin : float
        Minimum longitude bound.
    xmax : float
        Maximum longitude bound.
    ymin : float
        Minimum latitude bound.
    x_pop : np.ndarray
        Longitude values.
    y_pop : np.ndarray
        Latitude values.
    array_niche : np.ndarray
        2D niche suitability array.

    Returns
    -------
    np.ndarray
        Niche array with values outside bounds set to zero.
    """
    x_cut = np.where((x_pop >= xmin) & (x_pop <= xmax), x_pop, 0)
    y_cut = np.where(y_pop >= ymin, y_pop, 0)

    for x in np.where(x_cut == 0)[0]:
        array_niche[x, :] = 0

    for y in np.where(y_cut == 0)[0]:
        array_niche[:, y] = 0

    return array_niche
