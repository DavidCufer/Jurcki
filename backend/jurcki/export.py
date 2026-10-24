"""Turn daily score grids into the app files (see "App data contract" in docs/ARCHITECTURE.md).

Per day:  ``score_YYYY-MM-DD.png``  (RGBA heat map, Web-Mercator rows, row 0 = maxLat)
          ``hotspots_YYYY-MM-DD.geojson`` (de-clustered top cells)
Once:     ``meta.json`` and ``reports.geojson``.

Demo output built from the real static layers (clearly flagged ``"demo": true``)::

    python -m jurcki.export --demo
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from . import config, grid

SCHEMA_VERSION = 1

# --- PNG geometry --------------------------------------------------------------------------
# Outer extent of every PNG (a little larger than Slovenia).  Columns are spaced evenly in
# longitude, rows evenly in Web-Mercator y with the same pixel size, so pixels are square on
# the map.  DLON = 0.00135 deg is ~105 m on the ground at 46 N (both ways, as pixels are
# square in Mercator), i.e. ~2400 x 1600 px for Slovenia.
PNG_LON = (13.36, 16.62)      # minLon, maxLon (maxLon is snapped to a whole pixel count)
PNG_LAT = (45.41, 46.89)      # minLat, maxLat (maxLat is snapped to a whole pixel count)
DLON = 0.00135                # degrees of longitude per pixel
R_EARTH = 6378137.0           # EPSG:3857 sphere radius

# --- colours -------------------------------------------------------------------------------
# ColorBrewer YlOrRd, sampled at six stops (monotone luminance, readable on green/topo maps).
LEGEND: list[list] = [
    [0.0, "#FFFFB2"], [0.2, "#FED976"], [0.4, "#FEB24C"],
    [0.6, "#FD8D3C"], [0.8, "#E31A1C"], [1.0, "#800026"],
]
MIN_SCORE = 0.04        # below this the PNG is fully transparent
ALPHA_FULL_AT = 0.40    # alpha ramps linearly from 0 at MIN_SCORE to ALPHA_MAX here
ALPHA_MAX = 0.92
LEVELS = (64, 48, 32, 24)   # score is quantised to this many levels (keeps the PNG small);
MAX_PNG_BYTES = 1_900_000   # the coarser level counts are tried until the file fits

# --- hotspots ------------------------------------------------------------------------------
HOTSPOT_MIN_SCORE = 0.15
HOTSPOT_MIN_SEP_M = 1500.0
HOTSPOT_MAX = 300

ELEV_PATH = config.INTERIM / "static" / "elev.tif"


# ===========================================================================================
# Web Mercator helpers (same math as MapLibre's image quad: x linear in lon, y = ln tan(...))
# ===========================================================================================
def merc_x(lon):
    return R_EARTH * np.radians(lon)


def merc_y(lat):
    return R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def merc_lon(x):
    return np.degrees(np.asarray(x) / R_EARTH)


def merc_lat(y):
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y) / R_EARTH)) - np.pi / 2)


@lru_cache(maxsize=1)
def png_geometry() -> dict:
    """Pixel grid of the PNGs: size, Mercator pixel size and the outer ``bounds`` (lon/lat)."""
    nx = int(math.ceil((PNG_LON[1] - PNG_LON[0]) / DLON))
    px = float(merc_x(PNG_LON[0] + DLON) - merc_x(PNG_LON[0]))   # Mercator metres per pixel
    x0 = float(merc_x(PNG_LON[0]))
    y_bottom = float(merc_y(PNG_LAT[0]))
    ny = int(math.ceil((float(merc_y(PNG_LAT[1])) - y_bottom) / px))
    y_top = y_bottom + ny * px
    bounds = [round(float(merc_lon(x0)), 7), round(float(merc_lat(y_bottom)), 7),
              round(float(merc_lon(x0 + nx * px)), 7), round(float(merc_lat(y_top)), 7)]
    return dict(nx=nx, ny=ny, px=px, x0=x0, y_top=y_top, bounds=bounds)


def lonlat_to_pixel(lon, lat, bounds=None, size=None):
    """Fractional (row, col) of lon/lat in a PNG, using Mercator math (pixel centres at +0.5).

    Pass the ``bounds`` of meta.json and the PNG ``size`` = (height, width) to place points
    in an arbitrary exported file; default is this module's geometry.
    """
    if bounds is None:
        g = png_geometry()
        bounds, size = g["bounds"], (g["ny"], g["nx"])
    h, w = size
    x0, x1 = merc_x(bounds[0]), merc_x(bounds[2])
    y1, y0 = merc_y(bounds[3]), merc_y(bounds[1])   # y1 = top (maxLat)
    col = (merc_x(np.asarray(lon, float)) - x0) / (x1 - x0) * w
    row = (y1 - merc_y(np.asarray(lat, float))) / (y1 - y0) * h
    return row, col


@lru_cache(maxsize=1)
def _sample_coords():
    """Fractional model-grid (row, col) of every PNG pixel centre, shape (2, ny, nx)."""
    g = png_geometry()
    lon = merc_lon(g["x0"] + (np.arange(g["nx"]) + 0.5) * g["px"])
    lat = merc_lat(g["y_top"] - (np.arange(g["ny"]) + 0.5) * g["px"])
    lon2, lat2 = np.meshgrid(lon, lat)
    t = Transformer.from_crs(config.CRS_WGS84, config.CRS, always_xy=True)
    x, y = t.transform(lon2, lat2)
    x0, y0 = config.GRID_ORIGIN
    res = config.GRID_RES
    col = (x - x0) / res - 0.5       # cell centres sit at integer index coordinates
    row = (y0 - y) / res - 0.5
    return np.stack([row, col]).astype("float32")


def resample_to_png(score: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Bilinear-resample a model-grid array to the PNG pixel grid (float32, NaN = no data).

    NaN cells and cells outside ``mask`` (True = inside Slovenia) do not contribute; pixels
    whose centre falls outside the mask (or the grid) are NaN.
    """
    score = np.asarray(score, dtype="float64")
    if score.shape != grid.grid_shape():
        raise ValueError(f"score has shape {score.shape}, expected {grid.grid_shape()}")
    valid = np.isfinite(score)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    coords = _sample_coords()
    num = map_coordinates(np.where(valid, score, 0.0), coords, order=1, mode="constant", cval=0.0)
    den = map_coordinates(valid.astype("float64"), coords, order=1, mode="constant", cval=0.0)
    out = np.full(num.shape, np.nan, dtype="float32")
    ok = den > 0.5
    out[ok] = (num[ok] / den[ok]).astype("float32")
    if mask is not None:   # keep the country outline crisp: decide by the (interpolated) mask
        inside = map_coordinates(np.asarray(mask, "float64"), coords, order=1,
                                 mode="constant", cval=0.0) >= 0.5
        out[~inside] = np.nan
    return out


# ===========================================================================================
# Colours
# ===========================================================================================
def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


@lru_cache(maxsize=None)
def colour_lut(n_levels: int = LEVELS[0]) -> np.ndarray:
    """(n_levels, 4) uint8 RGBA lookup: level i <-> score i / (n_levels - 1)."""
    v = np.array([s[0] for s in LEGEND])
    rgb = np.array([_hex_rgb(s[1]) for s in LEGEND], float)
    s = np.linspace(0, 1, n_levels)
    lut = np.zeros((n_levels, 4), np.uint8)
    for c in range(3):
        lut[:, c] = np.round(np.interp(s, v, rgb[:, c]))
    a = np.clip((s - MIN_SCORE) / (ALPHA_FULL_AT - MIN_SCORE), 0, 1) * ALPHA_MAX
    a[s < MIN_SCORE] = 0
    lut[:, 3] = np.round(a * 255)
    return lut


def colorize(pix: np.ndarray, n_levels: int = LEVELS[0]) -> np.ndarray:
    """Score image (float, NaN = none) -> RGBA uint8 using the legend + alpha ramp."""
    lut = colour_lut(n_levels)
    lvl = np.clip(np.round(np.nan_to_num(pix, nan=0.0) * (n_levels - 1)), 0, n_levels - 1).astype(np.intp)
    rgba = lut[lvl]
    none = ~np.isfinite(pix)
    rgba[none] = lut[0]            # transparent pixels keep the lowest legend colour (no dark fringes)
    rgba[none, 3] = 0
    return rgba


def write_score_png(score: np.ndarray, path, mask: np.ndarray | None = None) -> Path:
    """Resample ``score`` (model grid, 0..1) and write the RGBA heat map."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = resample_to_png(score, mask)
    for n in LEVELS:
        Image.fromarray(colorize(pix, n), "RGBA").save(path, format="PNG", optimize=True,
                                                       compress_level=9)
        if path.stat().st_size <= MAX_PNG_BYTES:
            break
    return path


# ===========================================================================================
# Hotspots
# ===========================================================================================
def pick_hotspots(score: np.ndarray, mask: np.ndarray | None = None,
                  min_score: float = HOTSPOT_MIN_SCORE, min_sep_m: float = HOTSPOT_MIN_SEP_M,
                  max_n: int = HOTSPOT_MAX) -> list[tuple[int, int]]:
    """(row, col) of the best cells, greedily de-clustered: no two closer than ``min_sep_m``."""
    s = np.asarray(score, float)
    ok = np.isfinite(s) & (s >= min_score)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    rows, cols = np.nonzero(ok)
    if rows.size == 0:
        return []
    # stable sort: descending score, ties broken by position
    order = np.argsort(-s[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]
    pts = np.column_stack([rows, cols]) * float(config.GRID_RES)
    tree = cKDTree(pts)
    suppressed = np.zeros(len(rows), bool)
    picked: list[tuple[int, int]] = []
    for i in range(len(rows)):
        if suppressed[i]:
            continue
        picked.append((int(rows[i]), int(cols[i])))
        if len(picked) >= max_n:
            break
        suppressed[tree.query_ball_point(pts[i], r=min_sep_m - 1e-6)] = True
    return picked


def _num(v, digits=3):
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


def hotspots_geojson(score, habitat, trigger, elev=None, mask=None, **kw) -> dict:
    """Hotspot FeatureCollection (sorted by descending score)."""
    cells = pick_hotspots(score, mask, **kw)
    feats = []
    if cells:
        rc = np.array(cells)
        lon, lat = grid.cell_lonlat()
        for r, c in rc:
            e = None if elev is None else elev[r, c]
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [round(float(lon[r, c]), 5), round(float(lat[r, c]), 5)]},
                "properties": {
                    "score": _num(score[r, c]),
                    "habitat": _num(habitat[r, c]),
                    "trigger": _num(trigger[r, c]),
                    "elev": None if e is None or not np.isfinite(e) else int(round(float(e))),
                },
            })
    return {"type": "FeatureCollection", "features": feats}


@lru_cache(maxsize=1)
def _load_elev() -> np.ndarray | None:
    if not ELEV_PATH.exists():
        warnings.warn(f"{ELEV_PATH} missing: hotspot elev will be null")
        return None
    return grid.read_tif(ELEV_PATH)


# ===========================================================================================
# Public API
# ===========================================================================================
def export_day(day, score, habitat, trigger, out_dir, mask=None, elev=None) -> dict:
    """Write ``score_<day>.png`` and ``hotspots_<day>.geojson`` into ``out_dir``.

    ``score``, ``habitat``, ``trigger``: 2-D float arrays (0..1, NaN allowed) on the model
    grid.  ``mask`` defaults to the Slovenia land mask, ``elev`` to data/interim/static/elev.tif.
    Returns {"png": path, "hotspots": path, "n_hotspots": int}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = _day_str(day)
    mask = grid.load_mask() if mask is None else mask
    elev = _load_elev() if elev is None else elev
    for name, a in (("score", score), ("habitat", habitat), ("trigger", trigger)):
        if np.shape(a) != grid.grid_shape():
            raise ValueError(f"{name} has shape {np.shape(a)}, expected {grid.grid_shape()}")
    png = write_score_png(score, out_dir / f"score_{day}.png", mask)
    gj = hotspots_geojson(np.asarray(score), np.asarray(habitat), np.asarray(trigger), elev, mask)
    hp = out_dir / f"hotspots_{day}.geojson"
    hp.write_text(json.dumps(gj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"png": png, "hotspots": hp, "n_hotspots": len(gj["features"])}


def write_meta(days, first_forecast_day, out_dir, extra: dict | None = None) -> Path:
    """Write meta.json (schema_version 1).  ``extra`` keys (e.g. {"demo": True}) are added."""
    days = sorted({_day_str(d) for d in days})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": days,
        "first_forecast_day": _day_str(first_forecast_day),
        "bounds": png_geometry()["bounds"],
        "legend": LEGEND,
    }
    if extra:
        meta.update(extra)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def write_reports(df, out_dir) -> Path:
    """Write reports.geojson from a DataFrame with lon, lat, date, source, precision_m, text, url."""
    feats = []
    for rec in df.to_dict("records"):
        lon, lat = rec.get("lon"), rec.get("lat")
        if lon is None or lat is None or not (np.isfinite(lon) and np.isfinite(lat)):
            continue

        def clean(v):
            if v is None:
                return None
            try:
                if v != v:      # NaN / NaT
                    return None
            except (TypeError, ValueError):
                pass
            return v

        d = clean(rec.get("date"))
        prec = clean(rec.get("precision_m"))
        text, url = clean(rec.get("text")), clean(rec.get("url"))
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(float(lon), 5), round(float(lat), 5)]},
            "properties": {
                "date": None if d is None else _day_str(d),
                "source": clean(rec.get("source")),
                "precision_m": None if prec is None else int(round(float(prec))),
                "text": None if text is None else str(text),
                "url": None if url is None else str(url),
            },
        })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "reports.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False,
                            separators=(",", ":")), encoding="utf-8")
    return p


def _day_str(d) -> str:
    if isinstance(d, str):
        return date.fromisoformat(d).isoformat()
    if hasattr(d, "date") and callable(d.date):     # datetime / pandas Timestamp
        d = d.date()
    return d.isoformat()


# ===========================================================================================
# Demo
# ===========================================================================================
def demo_trigger(shape, day_index: int, seed: int = 7) -> np.ndarray:
    """Smooth synthetic 'weather' field in 0.15..1 that drifts east with the day index (DEMO ONLY)."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype("float32")
    f = np.zeros(shape, "float32")
    for _ in range(14):
        cx = rng.uniform(0, cols) + 25 * day_index
        cy = rng.uniform(0, rows)
        s = rng.uniform(60, 220)          # cells (15-55 km)
        f += rng.uniform(0.4, 1.0) * np.exp(-(((xx - cx) / s) ** 2 + ((yy - cy) / (0.8 * s)) ** 2))
    f /= f.max()
    return np.clip(0.15 + 0.95 * f, 0, 1).astype("float32")


def build_demo(out_dir=None, n_days: int = 3, today: date | None = None) -> Path:
    """Demo output: prior.habitat_score (month of ``today``) x synthetic trigger.  NOT a forecast."""
    from .model import prior

    out_dir = Path(out_dir or config.OUTPUT / "demo")
    today = today or date.today()
    st = config.INTERIM / "static"
    elev, slope = grid.read_tif(st / "elev.tif"), grid.read_tif(st / "slope.tif")
    tree = grid.read_tif(st / "treecover.tif")
    edge = grid.read_tif(st / "forest_edge_dist.tif")
    north = grid.read_tif(st / "northness.tif")
    hab = prior.habitat_score(elev, slope, tree, edge, north, month=today.month).astype("float32")
    mask = grid.load_mask()
    days = [today + timedelta(days=i) for i in range(n_days)]
    for i, d in enumerate(days):
        trig = demo_trigger(hab.shape, i)
        res = export_day(d, hab * trig, hab, trig, out_dir, mask=mask, elev=elev)
        print(f"{d}: {res['png'].name} {res['png'].stat().st_size / 1e6:.2f} MB, "
              f"{res['n_hotspots']} hotspots")
    write_meta(days, days[1] if n_days > 1 else days[0], out_dir,
               extra={"demo": True, "note": "DEMO: real habitat prior x synthetic trigger, not a forecast"})
    try:
        import pandas as pd
        write_reports(pd.DataFrame([
            {"lon": 14.31, "lat": 46.05, "date": today - timedelta(days=4), "source": "forum",
             "precision_m": 5000, "text": "DEMO: ob Pohorju prvi jurčki po dežju.", "url": None},
            {"lon": 14.84, "lat": 46.12, "date": today - timedelta(days=3), "source": "gbif",
             "precision_m": 30, "text": "DEMO: Boletus edulis.", "url": None},
        ]), out_dir)
    except ImportError:   # pragma: no cover
        pass
    return out_dir


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true", help="write demo output to data/output/demo/")
    ap.add_argument("--out", default=None, help="output directory (demo)")
    ap.add_argument("--days", type=int, default=3)
    a = ap.parse_args(argv)
    if not a.demo:
        ap.error("only --demo is available from the command line")
    print("wrote", build_demo(a.out, a.days))


if __name__ == "__main__":
    main()
