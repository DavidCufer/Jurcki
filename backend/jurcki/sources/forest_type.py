"""Forest type (conifer vs broadleaf) from Sentinel-2 leaf-on / leaf-off NDVI.

Deciduous broadleaf trees are leafless in winter, conifers stay green. Sentinel-2
L2A COGs (public AWS bucket ``sentinel-cogs``, MGRS zone 33T) are read through
their overviews at 20 m, windowed to Slovenia. Per tile we pick the scenes that
give the best clear-sky (and, in winter, snow-free) coverage of forest, using a
cheap 160 m read of the SCL band, and build two median NDVI composites:

  leaf-on   June-August                         (SCL vegetation / bare soil only)
  leaf-off  15 Nov - 5 Apr, sun elevation >= 20 deg  (same, so snow is excluded)

Cloud, cloud shadow, cirrus and snow are masked and the mask is dilated by 2 px.
Tree pixels are WorldCover class 10 that are not clearly non-green in summer
(leaf-on NDVI >= TREE_NDVI). A tree pixel is conifer when its leaf-off NDVI is
above CONIFER_NDVI = 0.60. Calibration: over ~58 M tree pixels the leaf-off NDVI
is a broadleaf mode (mean 0.47, sd 0.08: leafless canopy plus green understory)
with a long conifer shoulder (mean 0.68, sd 0.10) rather than a clean valley.
A two-component Gaussian mixture puts the equal-probability point at 0.60 (CLI
prints it; histogram in data/processed/forest_type_hist.csv) and gives a national
conifer share of ~30 %. Otsu (0.55) would give 43 %, too many. Deciduous larch
counts as broadleaf, and 20 m pixels in mixed stands fall to either side, so a
cell value is a share of pixels, not a species mix.

Layers written to data/interim/static/ (float32, model grid; NaN where a cell
has < MIN_TREE_AREA tree area with a valid leaf-off observation):
  conifer_frac   share of tree pixels that are conifer, 0-1
  ndvi_summer    mean leaf-on NDVI over tree pixels
  ndvi_leafoff   mean leaf-off NDVI over tree pixels
  forest_type_valid  share of the cell's tree pixels with a leaf-off observation, 0-1

Raw window reads are cached in data/raw/s2/<tile>/, composites in
data/interim/s2/, so reruns are cheap.
"""
import argparse
import json
import math
import os
import time
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy import ndimage
from sklearn.mixture import GaussianMixture

from .. import config, grid
from . import landcover

for _k, _v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
               "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2"}.items():
    os.environ.setdefault(_k, _v)

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
PREFIX = "sentinel-s2-l2a-cogs/33/T"
UTM = "EPSG:32633"
TILE_M, PX = 109_800, 20              # tile edge and working pixel size (m)
COLS = {"U": 300_000, "V": 400_000, "W": 500_000, "X": 600_000}   # tile west edge
ROWS = {"L": 5_100_000, "M": 5_200_000}                            # tile north edge
YEARS = range(2021, 2025)
SUMMER = ((6, 1), (8, 31))            # (month, day) window, inclusive
WINTER = ((11, 15), (4, 5))           # wraps over the new year
N_SCENES = {"summer": 3, "winter": 6}
WORKERS = 6                           # parallel HTTP reads
MAX_CLOUD = 60                        # % tile cloud cover for candidates
MIN_SUN = 20                          # deg, avoids deep topographic shadow
GOOD_SCL = (4, 5)                     # vegetation, not vegetated
BAD_SCL = (3, 8, 9, 10, 11)           # shadow, cloud, cirrus, snow
TREE_NDVI = 0.6                       # min leaf-on NDVI of a tree pixel
CONIFER_NDVI = 0.60                   # min leaf-off NDVI of a conifer pixel
MIN_TREE_AREA = 0.03                  # min valid tree area (fraction of cell)
S2_RAW = config.RAW / "s2"
S2_INTERIM = config.INTERIM / "s2"
S2_CACHE = config.CACHE / "s2"
STATIC = config.INTERIM / "static"
LAYERS = ("conifer_frac", "ndvi_summer", "ndvi_leafoff", "forest_type_valid")
HIST_CSV = config.PROCESSED / "forest_type_hist.csv"


# --- tiles and windows -------------------------------------------------------------

def tile_windows(min_cells: int = 500, buf: float = 500) -> dict[str, tuple[int, int, int, int]]:
    """Per MGRS tile: (col0, row0, ncols, nrows) in 20 m pixels around the Slovenia cells.

    Sizes are multiples of 8 so the 160 m selection read lines up with the window.
    """
    land = grid.load_mask()
    x, y = grid.cell_xy()
    X, Y = Transformer.from_crs(config.CRS, UTM, always_xy=True).transform(x[land], y[land])
    n = TILE_M // PX // 8 * 8
    out = {}
    for c, x0 in COLS.items():
        for r, y0 in ROWS.items():
            m = (X >= x0) & (X < x0 + TILE_M) & (Y <= y0) & (Y > y0 - TILE_M)
            if m.sum() < min_cells:
                continue
            c0 = max(0, int((X[m].min() - buf - x0) // (8 * PX))) * 8
            c1 = min(n, math.ceil((X[m].max() + buf - x0) / (8 * PX)) * 8)
            r0 = max(0, int((y0 - Y[m].max() - buf) // (8 * PX))) * 8
            r1 = min(n, math.ceil((y0 - Y[m].min() + buf) / (8 * PX)) * 8)
            out[c + r] = (c0, r0, c1 - c0, r1 - r0)
    return out


def window_transform(tile: str, win):
    """Affine transform of a tile window at 20 m."""
    c0, r0 = win[:2]
    return from_origin(COLS[tile[0]] + c0 * PX, ROWS[tile[1]] - r0 * PX, PX, PX)


def tree_mask(tile: str, win) -> np.ndarray:
    """WorldCover tree cover (class 10) on the 20 m window grid (bool), cached."""
    path = S2_RAW / tile / "tree.tif"
    if not path.exists():
        dst = np.zeros(win[3:1:-1], dtype="uint8")
        for i, p in enumerate(landcover.download_tiles()):
            with rasterio.open(p) as ds:
                reproject(rasterio.band(ds, 1), dst, dst_transform=window_transform(tile, win),
                          dst_crs=UTM, resampling=Resampling.mode, init_dest_nodata=i == 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", height=dst.shape[0], width=dst.shape[1], count=1,
                           dtype="uint8", crs=UTM, transform=window_transform(tile, win),
                           compress="deflate", tiled=True) as ds:
            ds.write(dst, 1)
    with rasterio.open(path) as ds:
        return ds.read(1) == 10


# --- scene catalogue ---------------------------------------------------------------

def in_season(d: date, season: str) -> bool:
    """True if d falls in the summer or winter composite window."""
    (m0, d0), (m1, d1) = SUMMER if season == "summer" else WINTER
    k, k0, k1 = (d.month, d.day), (m0, d0), (m1, d1)
    return k0 <= k <= k1 if k0 <= k1 else (k >= k0 or k <= k1)


def season_months(season: str) -> list[int]:
    return [6, 7, 8] if season == "summer" else [11, 12, 1, 2, 3, 4]


def scene_date(sid: str) -> date:
    """Acquisition date from a scene id such as S2A_33TUL_20230702_0_L2A."""
    s = sid.split("_")[2]
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def _get(url: str, **kw) -> requests.Response:
    """GET with retries (the proxy sometimes drops connections under load)."""
    for k in range(5):
        try:
            r = requests.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == 4:
                raise
            time.sleep(2 * (k + 1))


def _json_cache(name: str):
    path = S2_CACHE / name
    return path, (json.loads(path.read_text()) if path.exists() else {})


def _save(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def list_scenes(tile: str, season: str) -> list[str]:
    """Scene ids of a tile in the season's months over YEARS (S3 listing, cached)."""
    path, cache = _json_cache(f"list_{tile}.json")
    todo = [(y, m) for y in YEARS for m in season_months(season) if f"{y}-{m}" not in cache]

    def fetch(ym):
        pre = f"{PREFIX}/{tile}/{ym[0]}/{ym[1]}/"
        r = _get(BUCKET, params={"list-type": "2", "prefix": pre, "delimiter": "/"})
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        return [e.text.rstrip("/").split("/")[-1]
                for e in ET.fromstring(r.content).findall("s:CommonPrefixes/s:Prefix", ns)]

    with ThreadPoolExecutor(WORKERS) as ex:
        for ym, ids in zip(todo, ex.map(fetch, todo)):
            cache[f"{ym[0]}-{ym[1]}"] = ids
    if todo:
        _save(path, cache)
    ids = sorted(i for y in YEARS for m in season_months(season) for i in cache[f"{y}-{m}"]
                 if in_season(scene_date(i), season))
    return list({scene_date(i): i for i in reversed(ids)}.values())[::-1]   # one scene per day


def scene_url(tile: str, sid: str, name: str) -> str:
    d = scene_date(sid)
    return f"{BUCKET}/{PREFIX}/{tile}/{d.year}/{d.month}/{sid}/{name}"


def scene_meta(tile: str, ids: list[str]) -> dict[str, dict]:
    """Cloud %, sun elevation and nodata % per scene, from the STAC item JSON (cached)."""
    path, cache = _json_cache(f"meta_{tile}.json")
    todo = [i for i in ids if i not in cache]

    def fetch(sid):
        j = _get(scene_url(tile, sid, f"{sid}.json")).json()
        p = j["properties"]
        return {"cloud": p.get("eo:cloud_cover", 100), "sun": p.get("view:sun_elevation", 0),
                "nodata": p.get("s2:nodata_pixel_percentage", 0)}

    with ThreadPoolExecutor(WORKERS) as ex:
        for sid, m in zip(todo, ex.map(fetch, todo)):
            cache[sid] = m
    if todo:
        _save(path, cache)
    return {i: cache[i] for i in ids}


def coarse_valid(tile: str, ids: list[str], win) -> dict[str, np.ndarray]:
    """Usable-pixel masks (SCL vegetation / bare soil) at 160 m over the window (cached)."""
    path = S2_CACHE / f"coarse_{tile}.npz"
    have = dict(np.load(path)) if path.exists() else {}
    todo = [i for i in ids if i not in have]
    c0, r0, w, h = win

    def fetch(sid):
        with rasterio.open(scene_url(tile, sid, "SCL.tif")) as ds:
            scl = ds.read(1, window=Window(c0, r0, w, h), out_shape=(h // 8, w // 8),
                          resampling=Resampling.nearest)
        return np.isin(scl, GOOD_SCL)

    with ThreadPoolExecutor(WORKERS) as ex:
        for sid, v in zip(todo, ex.map(fetch, todo)):
            have[sid] = v
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **have)
    return {i: have[i] for i in ids}


def pick_scenes(valid: dict[str, np.ndarray], weight: np.ndarray, n_max: int,
                depth: int = 3, min_gain: float = 0.005) -> list[str]:
    """Greedy choice of scenes maximising weighted valid coverage up to ``depth`` observations."""
    count = np.zeros(weight.shape)
    total = weight.sum()
    chosen = []
    for _ in range(n_max):
        gain = {s: (weight * v * (count < depth)).sum() for s, v in valid.items() if s not in chosen}
        if not gain:
            break
        best = max(gain, key=gain.get)
        if gain[best] < min_gain * total:
            break
        chosen.append(best)
        count += valid[best]
    return chosen


def select(tile: str, season: str, win, tree: np.ndarray) -> list[str]:
    """Pick the scenes for a tile and season (low cloud, enough sun, best forest coverage)."""
    ids = list_scenes(tile, season)
    meta = scene_meta(tile, ids)
    cand = [i for i in ids if meta[i]["cloud"] <= MAX_CLOUD
            and (season == "summer" or meta[i]["sun"] >= MIN_SUN)]
    weight = grid.block_mean(tree[:win[3], :win[2]].astype("float32"), 8)
    chosen = pick_scenes(coarse_valid(tile, cand, win), weight, N_SCENES[season])
    print(f"{tile} {season}: {len(ids)} scenes, {len(cand)} candidates, picked {chosen}")
    return chosen


# --- scene reads and composites ----------------------------------------------------

def fetch_scene(tile: str, sid: str, win) -> "os.PathLike":
    """Read B04, B08 and SCL (20 m, window) and cache them as a 3-band uint16 GeoTIFF."""
    path = S2_RAW / tile / f"{sid}.tif"
    if path.exists():
        return path
    c0, r0, w, h = win
    bands = []
    for name, k in (("B04", 2), ("B08", 2), ("SCL", 1)):   # k: window scale (10 m bands -> 20 m overview)
        with rasterio.open(scene_url(tile, sid, f"{name}.tif")) as ds:
            bands.append(ds.read(1, window=Window(c0 * k, r0 * k, w * k, h * k), out_shape=(h, w),
                                 resampling=Resampling.nearest))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with rasterio.open(tmp, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint16",
                       crs=UTM, transform=window_transform(tile, win), compress="deflate",
                       predictor=2, tiled=True, blockxsize=256, blockysize=256) as ds:
        ds.write(np.stack(bands).astype("uint16"))
    tmp.rename(path)
    return path


def ndvi_masked(red: np.ndarray, nir: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """NDVI from L2A digital numbers, NaN where unusable.

    Usable = SCL vegetation / bare soil and not within 2 px of cloud, shadow, cirrus or snow.
    (The sentinel-cogs bands hold plain reflectance x 1e4 even where the STAC item
    advertises a -0.1 offset, so no offset is applied.)
    """
    r, n = red.astype("float32"), nir.astype("float32")
    with np.errstate(all="ignore"):
        v = (n - r) / (n + r)
    bad = ndimage.binary_dilation(np.isin(scl, BAD_SCL), iterations=2)
    ok = np.isin(scl, GOOD_SCL) & ~bad & (red > 0) & (nir > 0)
    return np.where(ok, v, np.nan).astype("float32")


def composite(paths) -> tuple[np.ndarray, np.ndarray]:
    """Median NDVI and number of valid observations over the cached scene files."""
    stack = []
    for p in paths:
        with rasterio.open(p) as ds:
            red, nir, scl = ds.read()
        stack.append(ndvi_masked(red, nir, scl))
    if not stack:
        return None, None
    stack = np.stack(stack)
    n = np.isfinite(stack).sum(0).astype("uint8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices
        med = np.nanmedian(stack, axis=0).astype("float32")
    return med, n


def tile_composites(tile: str, win, tree: np.ndarray) -> dict[str, np.ndarray]:
    """Leaf-on and leaf-off median NDVI plus observation counts for a tile window (cached)."""
    path = S2_INTERIM / f"{tile}.npz"
    if path.exists():
        return dict(np.load(path))
    out = {}
    for key, season in (("on", "summer"), ("off", "winter")):
        ids = select(tile, season, win, tree)
        with ThreadPoolExecutor(3) as ex:
            paths = list(ex.map(lambda s: fetch_scene(tile, s, win), ids))
        out[f"ndvi_{key}"], out[f"n_{key}"] = composite(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


# --- classification and aggregation ------------------------------------------------

def classify(ndvi_on, ndvi_off, wc_tree, tree_min: float = TREE_NDVI, conifer_min: float = CONIFER_NDVI):
    """Pixel classes from composites: (tree, valid, conifer) boolean arrays.

    tree     WorldCover tree pixel, dropped if its leaf-on NDVI is below ``tree_min``
    valid    tree pixel with a leaf-off observation
    conifer  valid pixel whose leaf-off NDVI is above ``conifer_min``
    """
    with np.errstate(invalid="ignore"):
        tree = np.asarray(wc_tree, bool) & ~(ndvi_on < tree_min)
        valid = tree & np.isfinite(ndvi_off)
        conifer = valid & (ndvi_off > conifer_min)
    return tree, valid, conifer


def pixel_sums(comp: dict, wc_tree: np.ndarray, **kw) -> dict[str, np.ndarray]:
    """Per-pixel indicators/values whose cell averages are the area sums used by cell_layers."""
    on, off = comp["ndvi_on"], comp["ndvi_off"]
    tree, valid, conifer = classify(on, off, wc_tree, **kw)
    on_ok = tree & np.isfinite(on)
    f = lambda a: a.astype("float32")
    return {"tree": f(tree), "valid": f(valid), "conifer": f(conifer), "on_ok": f(on_ok),
            "off_sum": np.where(valid, off, 0).astype("float32"),
            "on_sum": np.where(on_ok, on, 0).astype("float32")}


def cell_layers(s: dict[str, np.ndarray], min_area: float = MIN_TREE_AREA) -> dict[str, np.ndarray]:
    """Output layers from cell-averaged pixel sums (see pixel_sums)."""
    with np.errstate(all="ignore"):
        ok = s["valid"] >= min_area
        out = {"conifer_frac": np.where(ok, s["conifer"] / s["valid"], np.nan),
               "ndvi_leafoff": np.where(ok, s["off_sum"] / s["valid"], np.nan),
               "ndvi_summer": np.where(s["on_ok"] >= min_area, s["on_sum"] / s["on_ok"], np.nan),
               "forest_type_valid": np.where(s["tree"] >= min_area, s["valid"] / s["tree"], np.nan)}
    return {k: np.clip(v, -1, 1).astype("float32") if k.startswith("ndvi") else v.astype("float32")
            for k, v in out.items()}


def to_grid(arr: np.ndarray, tile: str, win) -> np.ndarray:
    """Cell-average a 20 m window array onto the model grid (0 outside the window)."""
    return grid.reproject_to_grid(arr, window_transform(tile, win), UTM,
                                  resampling=Resampling.average, dst_nodata=0)


def histogram(vals: np.ndarray, lo: float = -0.2, hi: float = 1.0, step: float = 0.02):
    """Histogram of leaf-off NDVI over tree pixels: (bin centres, counts)."""
    edges = np.arange(lo, hi + step / 2, step)
    return (edges[:-1] + edges[1:]) / 2, np.histogram(np.clip(vals, lo, hi - 1e-6), edges)[0]


def gmm_split(vals: np.ndarray) -> tuple[float, float]:
    """Two-Gaussian fit of leaf-off NDVI: (equal-probability point, share of the upper mode)."""
    vals = np.asarray(vals, "float64")[:, None]
    g = GaussianMixture(2, random_state=0).fit(vals)
    hi = int(np.argmax(g.means_))
    xs = np.linspace(g.means_.min(), g.means_.max(), 1001)[:, None]
    p = g.predict_proba(xs)[:, hi]
    return float(xs[np.argmin(np.abs(p - 0.5)), 0]), float(g.weights_[hi])


def derive(force: bool = False, conifer_min: float = CONIFER_NDVI) -> None:
    """Build all forest-type layers and write them to data/interim/static/."""
    if not force and all((STATIC / f"{n}.tif").exists() for n in LAYERS):
        print("forest-type layers exist, skipping")
        return
    sums, hist, sample = {}, 0, []
    rng = np.random.default_rng(0)
    for tile, win in tile_windows().items():
        wc = tree_mask(tile, win)
        comp = tile_composites(tile, win, wc)
        _, valid, _ = classify(comp["ndvi_on"], comp["ndvi_off"], wc)
        c, hist_t = histogram(comp["ndvi_off"][valid])
        hist = hist + hist_t
        sample.append(rng.choice(comp["ndvi_off"][valid], 50_000))
        for key, a in pixel_sums(comp, wc, conifer_min=conifer_min).items():
            sums[key] = sums.get(key, 0) + to_grid(a, tile, win)
        print(f"{tile}: window {win[2]}x{win[3]} px, tree px {int(wc.sum())}")
    thr, share = gmm_split(np.concatenate(sample))
    print(f"leaf-off NDVI of tree pixels: mixture split {thr:.2f} (upper mode {share:.0%}), "
          f"used threshold {conifer_min}")
    HIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(HIST_CSV, np.c_[c, hist], delimiter=",", fmt="%.2f,%d", header="ndvi_leafoff,count", comments="")
    for name, arr in cell_layers(sums).items():
        grid.write_tif(STATIC / f"{name}.tif", arr)
        print(f"{name}: min {np.nanmin(arr):.3g} max {np.nanmax(arr):.3g} mean {np.nanmean(arr):.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="recompute layers even if present")
    ap.add_argument("--conifer-ndvi", type=float, default=CONIFER_NDVI, help="leaf-off NDVI split")
    a = ap.parse_args()
    derive(a.force, a.conifer_ndvi)


if __name__ == "__main__":
    main()
