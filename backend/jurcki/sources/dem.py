"""Copernicus GLO-30 DEM -> terrain layers on the model grid.

Tiles are downloaded from the public AWS bucket, mosaicked, warped to
EPSG:3794 at a 25 m working resolution (native is ~30 m; 25 m makes each
250 m cell exactly 10 x 10 working pixels) and then aggregated by block mean.

Layers written to data/interim/static/ (float32, model grid):
  elev       mean elevation (m)
  slope      mean slope (deg, Horn)
  northness  mean of cos(aspect); +1 = north-facing slope, 0 = flat or mixed
  eastness   mean of sin(aspect); +1 = east-facing slope
  tpi500     elevation minus mean in a 1 km square window (+-500 m), m
  tpi2000    elevation minus mean in a 4 km square window (+-2 km), m
  twi        topographic wetness index ln(a / tan(slope)) from D8 flow
             accumulation on a 125 m DEM (depressions filled by priority-flood)
  tri        Riley terrain ruggedness index at 25 m (m), block-mean
"""
import argparse
import heapq
import math

import numpy as np
import rasterio
from rasterio.merge import merge
from scipy import ndimage

from .. import config, grid
from ._http import download

URL = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_DIR = config.RAW / "dem"
MOSAIC = DEM_DIR / "dem_mosaic.tif"
STATIC = config.INTERIM / "static"
WORK_RES = 25     # m, working resolution for slope/TPI/TRI
TWI_RES = 125     # m, resolution for flow accumulation
LAYERS = ("elev", "slope", "northness", "eastness", "tpi500", "tpi2000", "twi", "tri")


def tile_name(lat: int, lon: int) -> str:
    """Tile id for the 1x1 degree tile whose SW corner is (lat, lon)."""
    return f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_" \
           f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"


def needed_tiles() -> list[tuple[int, int]]:
    """(lat, lon) SW corners of tiles covering both BBOX_WGS84 and the model grid."""
    lon, lat = grid.cell_lonlat(config.GRID_RES * 4)
    minx = min(config.BBOX_WGS84[0], lon.min())
    maxx = max(config.BBOX_WGS84[2], lon.max())
    miny = min(config.BBOX_WGS84[1], lat.min())
    maxy = max(config.BBOX_WGS84[3], lat.max())
    return [(la, lo) for la in range(math.floor(miny), math.floor(maxy) + 1)
            for lo in range(math.floor(minx), math.floor(maxx) + 1)]


def download_tiles() -> list:
    """Download missing tiles into data/raw/dem/ (tiles over pure sea may not exist)."""
    paths = []
    for lat, lon in needed_tiles():
        name = tile_name(lat, lon)
        dest = DEM_DIR / f"{name}.tif"
        if download(f"{URL}/{name}/{name}.tif", dest):
            paths.append(dest)
        else:
            print("no tile (sea?):", name)
    return paths


def make_mosaic(paths) -> None:
    """Mosaic the tiles into one compressed GeoTIFF (skipped if it exists)."""
    if MOSAIC.exists():
        return
    srcs = [rasterio.open(p) for p in paths]
    arr, tr = merge(srcs, nodata=0)
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=1,
                dtype=arr.dtype, crs=srcs[0].crs, transform=tr, nodata=0,
                compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    for s in srcs:
        s.close()
    with rasterio.open(MOSAIC, "w", **prof) as ds:
        ds.write(arr)


def horn_gradient(z: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981) gradients (east, north), m per m. Edges are replicated."""
    p = np.pad(z, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    dzdy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * res)
    return dzdx, dzdy


def riley_tri(z: np.ndarray) -> np.ndarray:
    """Riley terrain ruggedness: sqrt of summed squared differences to the 8 neighbours."""
    p = np.pad(z, 1, mode="edge")
    c = p[1:-1, 1:-1]
    acc = np.zeros_like(z)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if (dy, dx) != (1, 1):
                acc += (p[dy:dy + z.shape[0], dx:dx + z.shape[1]] - c) ** 2
    return np.sqrt(acc)


def tpi(z: np.ndarray, radius_m: float, res: float) -> np.ndarray:
    """Elevation minus the mean in a square window of half-width radius_m."""
    size = 2 * int(round(radius_m / res)) + 1
    return z - ndimage.uniform_filter(z, size=size, mode="nearest")


def flow_accumulation(z: np.ndarray) -> np.ndarray:
    """D8 flow accumulation (cell counts) with depressions filled by priority-flood.

    Floods inwards from the grid border (all border cells are outlets); each cell
    drains to the neighbour that first reached it, i.e. its lowest neighbour on the
    filled surface. Accumulation then runs in reverse flood order.
    """
    rows, cols = z.shape
    w = cols + 2
    zl = np.pad(z.astype("float64"), 1).ravel().tolist()
    closed = np.zeros((rows + 2, w), dtype=bool)
    closed[0, :] = closed[-1, :] = closed[:, 0] = closed[:, -1] = True
    idx = np.arange((rows + 2) * w).reshape(rows + 2, w)
    seeds = np.unique(np.concatenate([idx[1, 1:-1], idx[-2, 1:-1], idx[1:-1, 1], idx[1:-1, -2]]))
    closed = bytearray(closed.ravel().tobytes())
    heap = [(zl[i], i) for i in seeds.tolist()]
    for _, i in heap:
        closed[i] = 1
    heapq.heapify(heap)
    recv = {}
    order = []
    offs = (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
    eps = 1e-5
    while heap:
        h, i = heapq.heappop(heap)
        order.append(i)
        for o in offs:
            j = i + o
            if closed[j]:
                continue
            closed[j] = 1
            recv[j] = i
            zj = zl[j]
            heapq.heappush(heap, (zj if zj > h else h + eps, j))
    acc = [1] * len(zl)
    for i in reversed(order):
        r = recv.get(i)
        if r is not None:
            acc[r] += acc[i]
    return np.array(acc, dtype="float32").reshape(rows + 2, w)[1:-1, 1:-1]


def twi_layer(z25: np.ndarray) -> np.ndarray:
    """TWI on the 125 m DEM, aggregated to the model grid."""
    k = TWI_RES // WORK_RES
    z = grid.block_mean(z25, k)
    dzdx, dzdy = horn_gradient(z, TWI_RES)
    tanb = np.maximum(np.hypot(dzdx, dzdy), 0.01)   # clamp flats
    sca = flow_accumulation(z) * TWI_RES            # specific catchment area, m
    twi = np.log(sca / tanb).astype("float32")
    return grid.block_mean(twi, config.GRID_RES // TWI_RES)


def derive(force: bool = False) -> None:
    """Compute all terrain layers and write them to data/interim/static/."""
    if not force and all((STATIC / f"{n}.tif").exists() for n in LAYERS):
        print("terrain layers exist, skipping")
        return
    k = config.GRID_RES // WORK_RES
    z = grid.reproject_to_grid(MOSAIC, res=WORK_RES)
    z = np.nan_to_num(z, nan=0.0)          # tiles missing over sea
    out = {"elev": grid.block_mean(z, k)}

    dzdx, dzdy = horn_gradient(z, WORK_RES)
    mag = np.hypot(dzdx, dzdy)
    out["slope"] = grid.block_mean(np.degrees(np.arctan(mag)), k)
    safe = np.where(mag > 0, mag, 1)
    out["northness"] = grid.block_mean(-dzdy / safe * (mag > 0), k)
    out["eastness"] = grid.block_mean(-dzdx / safe * (mag > 0), k)
    del dzdx, dzdy, mag, safe

    out["tpi500"] = grid.block_mean(tpi(z, 500, WORK_RES), k)
    out["tpi2000"] = grid.block_mean(tpi(z, 2000, WORK_RES), k)
    out["tri"] = grid.block_mean(riley_tri(z), k)
    out["twi"] = twi_layer(z)
    for name, arr in out.items():
        grid.write_tif(STATIC / f"{name}.tif", arr.astype("float32"))
        print(f"{name}: min {np.nanmin(arr):.3g} max {np.nanmax(arr):.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="recompute layers even if present")
    args = ap.parse_args()
    make_mosaic(download_tiles())
    derive(args.force)


if __name__ == "__main__":
    main()
