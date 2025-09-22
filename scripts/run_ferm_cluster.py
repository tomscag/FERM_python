# scripts/run_ferm_cluster.py

import argparse
from ferm.cluster_runner import run_cluster
from ferm.config import DEFAULT_PATHS, NB_PARTICULES, SIGMA


def main():
    parser = argparse.ArgumentParser(description="Run FERM cluster-mode simulation (multiprocessing)")
    parser.add_argument("--niche", default=DEFAULT_PATHS["niche_array"], help="Path to niche array (.npy)")
    parser.add_argument("--x", default=DEFAULT_PATHS["x_coords"], help="Path to longitude array (.npy)")
    parser.add_argument("--y", default=DEFAULT_PATHS["y_coords"], help="Path to latitude array (.npy)")
    parser.add_argument("--pop", default=DEFAULT_PATHS["pop_raster"], help="Path to population raster (.tif)")
    parser.add_argument("--out", default="mobility_sigma=1_chunksize=1.npz", help="Output sparse matrix path")
    parser.add_argument("--sigma", type=float, default=SIGMA, help="Standard deviation for sampling")
    parser.add_argument("--particles", type=int, default=NB_PARTICULES, help="Number of walkers per origin")
    parser.add_argument("--chunksize", type=int, default=1, help="Chunksize for multiprocessing")

    args = parser.parse_args()

    run_cluster(
        path_niche_array=args.niche,
        path_x=args.x,
        path_y=args.y,
        path_pop=args.pop,
        nb_particules=args.particles,
        sigma=args.sigma,
        save_path=args.out,
        chunksize=args.chunksize
    )


if __name__ == "__main__":
    main()

