"""ESA WorldCover 2021 v200 (10 m) -> land-cover layers on the model grid.

Tiles are 3x3 degree COGs on the public AWS bucket, named by their SW corner
(multiples of 3 degrees). Class fractions are computed by area-averaging class
indicators from 10 m to a 50 m grid, then block-meaning to the 250 m grid.

Layers written to data/interim/static/ (float32, model grid):
  treecover         fraction of cell that is tree cover (class 10), 0-1
  grassland         fraction grassland (class 30), 0-1
  cropland          fraction cropland (class 40), 0-1
  builtup           fraction built-up (class 50), 0-1
  forest_edge_dist  signed distance to the forest edge in metres, mean over the
                    cell and clipped to +-5 km: positive inside forest (distance
                    to the nearest non-forest), negative outside (minus distance
                    to the nearest forest). Forest = 50 m pixels with >= 50 %
                    tree cover.
"""
import argparse
import math

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling
from scipy import ndimage

from .. import config, grid
from ._http import download

URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WC_DIR = config.RAW / "worldcover"
STATIC = config.INTERIM / "static"
WORK_RES = 50   # m, intermediate grid for fractions and edge distance
MAX_DIST = 5000  # m, edge distance is clipped to +-this
CLASSES = {"treecover": 10, "grassland": 30, "cropland": 40, "builtup": 50}
LAYERS = (*CLASSES, "forest_edge_dist")


def tile_name(lat: int, lon: int) -> str:
    """File name of the 3x3 degree tile whose SW corner is (lat, lon)."""
    return f"ESA_WorldCover_10m_2021_v200_{'N' if lat >= 0 else 'S'}{abs(lat):02d}" \
           f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_Map.tif"


def lonlat_bounds() -> tuple[float, float, float, float]:
    """WGS84 bounds covering both BBOX_WGS84 and the model grid."""
    lon, lat = grid.cell_lonlat(config.GRID_RES * 4)
    b = config.BBOX_WGS84
    return (min(b[0], lon.min()), min(b[1], lat.min()),
            max(b[2], lon.max()), max(b[3], lat.max()))


def needed_tiles() -> list[tuple[int, int]]:
    """(lat, lon) SW corners (multiples of 3) of tiles covering lonlat_bounds()."""
    minx, miny, maxx, maxy = lonlat_bounds()
    return [(la, lo) for la in range(3 * math.floor(miny / 3), math.ceil(maxy), 3)
            for lo in range(3 * math.floor(minx / 3), math.ceil(maxx), 3)]


def download_tiles() -> list:
    """Download missing tiles into data/raw/worldcover/."""
    paths = []
    for lat, lon in needed_tiles():
        name = tile_name(lat, lon)
        if download(f"{URL}/{name}", WC_DIR / name, timeout=120):
            paths.append(WC_DIR / name)
        else:
            print("no tile:", name)
    return paths


def fractions(paths) -> dict[str, np.ndarray]:
    """Class fractions (0-1) on the 50 m grid, area-averaged from the 10 m map."""
    minx, miny, maxx, maxy = lonlat_bounds()
    srcs = [rasterio.open(p) for p in paths]
    lc, tr = merge(srcs, bounds=(minx - .01, miny - .01, maxx + .01, maxy + .01), nodata=0)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    out = {}
    for name, code in CLASSES.items():
        f = grid.reproject_to_grid((lc[0] == code).view("uint8"), tr, crs, res=WORK_RES,
                                   resampling=Resampling.average, dst_nodata=0)
        out[name] = np.where(f < 1e-3, 0, np.clip(f, 0, 1)).astype("float32")  # drop warp noise
    return out


def forest_edge_distance(tree50: np.ndarray) -> np.ndarray:
    """Signed distance (m) to the forest edge on the 50 m grid: + inside, - outside."""
    forest = tree50 >= 0.5
    half = WORK_RES / 2
    inside = ndimage.distance_transform_edt(forest) * WORK_RES - half
    outside = ndimage.distance_transform_edt(~forest) * WORK_RES - half
    return np.clip(np.where(forest, inside, -outside), -MAX_DIST, MAX_DIST).astype("float32")


def derive(force: bool = False) -> None:
    """Compute all land-cover layers and write them to data/interim/static/."""
    if not force and all((STATIC / f"{n}.tif").exists() for n in LAYERS):
        print("land-cover layers exist, skipping")
        return
    k = config.GRID_RES // WORK_RES
    frac = fractions(download_tiles())
    out = {n: grid.block_mean(f, k) for n, f in frac.items()}
    out["forest_edge_dist"] = grid.block_mean(forest_edge_distance(frac["treecover"]), k)
    for name, arr in out.items():
        grid.write_tif(STATIC / f"{name}.tif", arr.astype("float32"))
        print(f"{name}: min {np.nanmin(arr):.3g} max {np.nanmax(arr):.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="recompute layers even if present")
    derive(ap.parse_args().force)


if __name__ == "__main__":
    main()
