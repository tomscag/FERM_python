# config.py

"""
Default configuration parameters for the FERM model.
"""

# Default number of particles per origin
NB_PARTICULES: int = 100

# Standard deviation of the Gaussian used in absorption sampling
SIGMA: float = 1.0

# Default bounding box for filtering niche
XMIN: float = -40
XMAX: float = 47
YMIN: float = 4

# Default number of parallel processes (can be set to os.cpu_count())
N_PROCESSES: int = 12

# Default file paths (to be overridden by CLI or script)
DEFAULT_PATHS = {
    "niche_array": "array_of_niche_to_pop.npy",
    "x_coords": "x_pop.npy",
    "y_coords": "y_pop.npy",
    "pop_raster": "pop_test.tif",
    "output_matrix": "mobility_matrix.npz"
}

