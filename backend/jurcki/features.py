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
# Weather-derived features at a point (from the ERA5 daily history) built on
# ``model.prior.weather_features``. Everything is time-first, so the same code
# serves one point (1-D) and a stack of point-years (2-D: time x series).
WEATHER_VARS = ["tmean", "tmin", "tmax", "precip", "swvl1", "stl1", "snow_depth"]
SNOW_MIN_M = 0.02   # snow depth (m) that counts as snow on the ground
SNOW_DAYS = 7       # window for the snow flag
DYNAMIC_NAMES = ["t20", "t7", "p26", "p14", "p7", "pulse", "tmax10", "tmin7",
                 "swvl1_7", "swvl1_14", "stl1_20", "snow", "trigger"]


@lru_cache(maxsize=1)
def _history():
    """ERA5 daily history of the study area (all years), loaded once."""
    from .sources import weather
    return weather.open_history()


def era5_orography(lon, lat) -> float:
    """ERA5 surface height (m) bilinearly interpolated at a point."""
    return float(_history()["orography"].interp(lat=lat, lon=lon))


def point_elevation(lon, lat):
    """Elevation (m) and its source: the DEM where inside the grid, else ERA5 orography."""
    z = float(sample_static([lon], [lat], ["elev"])["elev"].iloc[0])
    if np.isfinite(z):
        return z, "dem"
    return era5_orography(lon, lat), "era5"


def point_weather(lon, lat, elev, start, end) -> pd.DataFrame:
    """Daily weather at a point, ``start``..``end`` (inclusive), indexed by date.

    Bilinear interpolation of the ERA5 0.25 deg history; ``tmean``, ``tmin``
    and ``tmax`` are shifted by the lapse rate from the interpolated ERA5
    orography to ``elev`` (None -> no shift). Days missing at the source are
    interpolated across; days after the end of the history are dropped.
    The ERA5 height is in ``df.attrs["orography"]``.
    """
    from .sources import weather
    ds = _history()
    if not (ds.lon.min() <= lon <= ds.lon.max() and ds.lat.min() <= lat <= ds.lat.max()):
        raise ValueError(f"point ({lon}, {lat}) is outside the ERA5 window")
    pt = ds[WEATHER_VARS + ["orography"]].sel(time=slice(start, end)).interp(lat=lat, lon=lon)
    orog = float(pt["orography"])
    df = pt[WEATHER_VARS].to_dataframe().drop(columns=["lat", "lon"], errors="ignore")
    df = df.astype("float64").interpolate(method="time", limit_area="inside")
    df = df.dropna(how="all").ffill().bfill()
    if elev is not None:
        df[["tmean", "tmin", "tmax"]] += weather.LAPSE_RATE * (elev - orog)
    df.attrs["orography"] = orog
    return df


def dynamic_feature_arrays(w, p=None) -> dict:
    """Weather arrays (time first) -> dict of lagged features of the same shape.

    ``w``: mapping with tmean, tmin, tmax, precip, swvl1, stl1, snow_depth.
    Besides the ``prior.weather_features`` (t20, p26, p7, pulse, tmax10,
    tmin7) it adds a 7-day mean temperature, 14-day rain, soil moisture
    means over 7 and 14 days, the 20-day mean soil temperature, a snow flag
    (snow on the ground within the last week) and the trigger score itself.
    """
    from .model import prior
    p = p or prior.PriorParams()
    w = {k: np.asarray(v, dtype="float64") for k, v in w.items()}
    f = prior.weather_features(w["tmean"], w["tmin"], w["tmax"], w["precip"], p)
    f["t7"] = prior._rolling_mean(w["tmean"], 7)
    f["p14"] = prior._rolling_sum(w["precip"], 14)
    f["swvl1_7"] = prior._rolling_mean(w["swvl1"], 7)
    f["swvl1_14"] = prior._rolling_mean(w["swvl1"], 14)
    f["stl1_20"] = prior._rolling_mean(w["stl1"], 20)
    f["snow"] = (prior._rolling_max(w["snow_depth"], SNOW_DAYS) > SNOW_MIN_M).astype(float)
    f["trigger"] = prior.trigger_score(f, p)
    return f


def dynamic_features(df: pd.DataFrame, params=None) -> pd.DataFrame:
    """``point_weather`` output -> daily DataFrame of dynamic features (see ``DYNAMIC_NAMES``).

    Windows are trailing, so the first weeks of ``df`` are a warm-up: start
    the weather at least ~60 days before the first day that is used.
    """
    f = dynamic_feature_arrays({k: df[k].to_numpy() for k in WEATHER_VARS}, params)
    return pd.DataFrame({k: f[k] for k in DYNAMIC_NAMES}, index=df.index)
