"""Model grid helpers: transform, shape, reprojection, cell centres, land mask.

All layers live on one grid (see config.py). Helpers take an optional ``res``
so finer working grids (e.g. 25 m) share the same origin and extent.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
from pyproj import Transformer
from shapely.geometry import shape

from . import config
from .sources._http import download

NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          "master/geojson/ne_10m_admin_0_countries.geojson")
BOUNDARY_PATH = config.RAW / "slovenia_ne10m.geojson"
MASK_PATH = config.INTERIM / "mask.tif"


def grid_transform(res: float = config.GRID_RES) -> Affine:
    """Affine transform of the grid (north-up, origin at the top-left corner)."""
    return from_origin(*config.GRID_ORIGIN, res, res)


def grid_shape(res: float = config.GRID_RES) -> tuple[int, int]:
    """(rows, cols) of the grid."""
    w, h = config.GRID_EXTENT
    return int(round(h / res)), int(round(w / res))


def grid_bounds() -> tuple[float, float, float, float]:
    """(left, bottom, right, top) in EPSG:3794."""
    x0, y0 = config.GRID_ORIGIN
    w, h = config.GRID_EXTENT
    return x0, y0 - h, x0 + w, y0


def grid_profile(res: float = config.GRID_RES, **kw) -> dict:
    """rasterio profile for a single-band float32 GeoTIFF on the grid."""
    rows, cols = grid_shape(res)
    prof = dict(driver="GTiff", height=rows, width=cols, count=1, dtype="float32",
                crs=config.CRS, transform=grid_transform(res), nodata=np.nan,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    prof.update(kw)
    return prof


def cell_xy(res: float = config.GRID_RES) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre x, y (EPSG:3794) as 2-D arrays of grid shape."""
    rows, cols = grid_shape(res)
    x0, y0 = config.GRID_ORIGIN
    x = x0 + (np.arange(cols) + 0.5) * res
    y = y0 - (np.arange(rows) + 0.5) * res
    return np.meshgrid(x, y)


def cell_lonlat(res: float = config.GRID_RES) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre lon, lat (EPSG:4326) as 2-D arrays of grid shape."""
    x, y = cell_xy(res)
    t = Transformer.from_crs(config.CRS, config.CRS_WGS84, always_xy=True)
    lon, lat = t.transform(x, y)
    return lon, lat


def lonlat_to_rowcol(lon, lat, res: float = config.GRID_RES):
    """Grid (row, col) of WGS84 points; may be outside the grid."""
    t = Transformer.from_crs(config.CRS_WGS84, config.CRS, always_xy=True)
    x, y = t.transform(lon, lat)
    x0, y0 = config.GRID_ORIGIN
    return (np.floor((y0 - np.asarray(y)) / res).astype(int),
            np.floor((np.asarray(x) - x0) / res).astype(int))


def reproject_to_grid(src, src_transform=None, src_crs=None, res: float = config.GRID_RES,
                      resampling=Resampling.bilinear, src_nodata=None,
                      dtype="float32", dst_nodata=np.nan) -> np.ndarray:
    """Reproject a 2-D array (with src_transform/src_crs) or a raster path onto the grid."""
    dst = np.full(grid_shape(res), dst_nodata, dtype=dtype)
    kw = dict(destination=dst, dst_transform=grid_transform(res), dst_crs=config.CRS,
              resampling=resampling, dst_nodata=dst_nodata)
    if isinstance(src, (str, Path)):
        with rasterio.open(src) as ds:
            nd = src_nodata if src_nodata is not None else ds.nodata
            reproject(rasterio.band(ds, 1), src_nodata=nd, **kw)
    else:
        reproject(np.asarray(src), src_transform=src_transform, src_crs=src_crs,
                  src_nodata=src_nodata, **kw)
    return dst


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k blocks (NaN cells ignored); shape must be divisible by k."""
    r, c = a.shape
    b = a.reshape(r // k, k, c // k, k)
    if np.isnan(a).any():
        with np.errstate(all="ignore"):
            return np.nanmean(b, axis=(1, 3)).astype("float32")
    return b.mean(axis=(1, 3), dtype="float32")


def write_tif(path, arr: np.ndarray, res: float = config.GRID_RES, **kw) -> Path:
    """Write a 2-D array on the grid as a float32 (or given dtype) GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = grid_profile(res, dtype=str(arr.dtype), **kw)
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr, 1)
    return path


def read_tif(path) -> np.ndarray:
    """Read band 1 of a GeoTIFF."""
    with rasterio.open(path) as ds:
        return ds.read(1)


def slovenia_boundary() -> dict:
    """Slovenia polygon (GeoJSON geometry, EPSG:4326) from Natural Earth, cached in data/raw."""
    if not BOUNDARY_PATH.exists():
        tmp = config.RAW / "ne_10m_admin_0_countries.geojson"
        if not download(NE_URL, tmp, timeout=120):
            raise RuntimeError(f"not found: {NE_URL}")
        with open(tmp) as f:
            feats = [x for x in json.load(f)["features"]
                     if x["properties"].get("ADM0_A3") == "SVN"]
        with open(BOUNDARY_PATH, "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        tmp.unlink()
    with open(BOUNDARY_PATH) as f:
        return json.load(f)["features"][0]["geometry"]


def land_mask(geometry: dict | None = None, res: float = config.GRID_RES) -> np.ndarray:
    """Boolean grid, True where the cell centre lies inside Slovenia (or ``geometry``, WGS84)."""
    geom = gpd.GeoSeries([shape(geometry or slovenia_boundary())],
                         crs=config.CRS_WGS84).to_crs(config.CRS).iloc[0]
    return features.rasterize([(geom, 1)], out_shape=grid_shape(res),
                              transform=grid_transform(res), fill=0, dtype="uint8").astype(bool)


def load_mask() -> np.ndarray:
    """Slovenia land mask from data/interim/mask.tif (built on first use)."""
    if not MASK_PATH.exists():
        build_mask()
    return read_tif(MASK_PATH).astype(bool)


def build_mask() -> Path:
    """Rasterise the Slovenia boundary onto the grid and save mask.tif (uint8, 1 = land)."""
    return write_tif(MASK_PATH, land_mask().astype("uint8"), nodata=None)


if __name__ == "__main__":
    p = build_mask()
    print(p, int(load_mask().sum()), "land cells")
