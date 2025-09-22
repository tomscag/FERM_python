# FERM: Feature-Enriched Radiation Model 

The Features-Enriched Radiation Model (FERM) is a flexible mathematical model that can be applied when, besides the population, other exogenous information can be used to model the attractiveness of geographical locations. This information is encoded in the features of the locations and acts as a cause of large-scale human movements. Our model is based on the same physical process of the original Radiation Model, but the stochastic process is generalized in order for the features to be included as external drivers. The features can change the mobility patterns by changing the bilateral flows, while the global influence of the surrounding locations is still governed by an emission/absorption process.

FERM uses geospatial raster inputs and Gaussian-based adaptive sampling to simulate migration transitions as sparse matrices.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/CoMuNeLab/FERM_python
cd FERM_python
pip install -e .
```

Requires Python >= 3.10.

## Dependencies

* numpy
* scipy
* rioxarray
* rasterio
* geopy
* geokernels
* ARSpy
* tqdm

## sage

### CLI

#### Standard (Single Process)

```bash
python scripts/run_ferm.py \
  --niche data/array_of_niche_to_pop.npy \
  --x data/x_pop.npy \
  --y data/y_pop.npy \
  --pop data/pop_test.tif \
  --out mobility_matrix.npz \
  --particles 100 --sigma 1.0
```

#### Cluster Mode (Multiprocessing)

```bash
python scripts/run_ferm_cluster.py \
  --niche data/array_of_niche_to_pop.npy \
  --x data/x_pop.npy \
  --y data/y_pop.npy \
  --pop data/pop_cluster.tif \
  --out mobility_sigma=1_chunksize=1.npz \
  --particles 500 --sigma 1.0 --chunksize 1
```

### Python API

```python
from FERM.model import FERM
from FERM.cluster_runner import run_cluster

FERM(path_niche_array, path_x, path_y, path_pop)
# or
run_cluster(path_niche_array, path_x, path_y, path_pop)
```

## Project Structure

```
src/FERM/
├── sampling.py         # ARS-based Gaussian sampling
├── distance.py         # Geospatial distance utilities
├── utils.py            # Mask parsing and filtering
├── model.py            # FERM simulation logic
├── cluster_runner.py   # Multiprocessing version
├── config.py           # Global constants
```

## Tests

To run the test suite:

```bash
pytest tests/ -v
```

FERM includes tests for:
* Sampling distributions
* Mask filtering
* Output matrix structure

## License

MIT License © CoMuNeLab

