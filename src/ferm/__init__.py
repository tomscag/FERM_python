# __init__.py

"""
FERM (Feature-Enriched Radiation Model): A model for simulating
migration behavior using ecological niches and population data.

This package provides:
- Adaptive Gaussian max sampling via ARS
- Geospatial distance utilities
- Mask filtering and coordinate parsing
- Mobility matrix computation via single-core and multi-core routines
"""

__version__ = "0.1.0"

from . import sampling
from . import distance
from . import utils
from . import model
from . import config
