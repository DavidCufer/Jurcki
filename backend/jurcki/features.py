"""Feature stack on the model grid: static layers now, dynamic (weather) later.

``load_static_stack`` reads the rasters in data/interim/static once and
``sample_static`` / ``sample_rowcol`` pull their values at points or cells.
Everything is on the shared grid (see config.py), so row/col indexing works
across layers.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, grid

STATIC_DIR = config.INTERIM / "static"

# Layers defined for every cell of the grid (also outside Slovenia).
STATIC_CORE = ["elev", "slope", "northness", "eastness", "tpi500", "tpi2000", "twi",
               "tri", "treecover", "grassland", "cropland", "builtup", "forest_edge_dist"]
# Layers from Slovenian forest-type mapping; NaN where it is not available
# (most of the grid outside Slovenia, and non-forest cells inside).
STATIC_FOREST_TYPE = ["conifer_frac", "ndvi_summer", "ndvi_leafoff"]
STATIC_NAMES = STATIC_CORE + STATIC_FOREST_TYPE


@lru_cache(maxsize=4)
def _load(names: tuple) -> np.ndarray:
    return np.stack([grid.read_tif(STATIC_DIR / f"{n}.tif").astype("float32") for n in names])


def load_static_stack(names=None):
    """(names, array of shape (n_layers, rows, cols)); cached, do not modify in place."""
    names = tuple(names or STATIC_NAMES)
    return list(names), _load(names)


def sample_rowcol(rows, cols, names=None) -> pd.DataFrame:
    """Static values at grid cells; cells outside the grid give NaN."""
    names, stack = load_static_stack(names)
    rows, cols = np.asarray(rows), np.asarray(cols)
    n_r, n_c = stack.shape[1:]
    ok = (rows >= 0) & (rows < n_r) & (cols >= 0) & (cols < n_c)
    out = np.full((len(rows), len(names)), np.nan, dtype="float32")
    out[ok] = stack[:, rows[ok], cols[ok]].T
    return pd.DataFrame(out, columns=names)


def sample_static(lon, lat, names=None) -> pd.DataFrame:
    """Static values at WGS84 points (NaN outside the grid)."""
    rows, cols = grid.lonlat_to_rowcol(lon, lat)
    return sample_rowcol(rows, cols, names)


def stack_to_frame(names=None, mask=None) -> pd.DataFrame:
    """Long table (one row per cell, or per True cell of ``mask``) of the static layers."""
    names, stack = load_static_stack(names)
    flat = stack.reshape(len(names), -1).T
    if mask is not None:
        flat = flat[np.asarray(mask).ravel()]
    return pd.DataFrame(flat, columns=names)


# --- dynamic features -----------------------------------------------------
# Weather-derived features (lagged rain, 20-day temperature, ...) go here:
# a function returning an array on the same grid, per day, so it can be
# stacked with the static layers above. Not implemented yet.
