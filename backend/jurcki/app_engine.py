"""Reference implementation of the on-phone engine (the spec in docs/APP_ENGINE.md).

This is a deliberately plain re-implementation of what the Android app does, written
from the spec and *not* reusing ``predict`` / ``model.prior``: explicit trailing-window
loops, an Open-Meteo JSON parser, gap filling, bilinear downscaling on the coarse
lattice and the score / hotspot / colour rules.  Tests compare it with the production
code paths (``predict`` + ``model.prior`` + ``export``) so that the spec is proven to
describe production, and the parity vectors for the Kotlin port come from production.

Everything is float64 numpy; the Kotlin port may use Double (or Float, within the
tolerances stated in the spec).
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

R_EARTH = 6378137.0
LAPSE_RATE = -6.5e-3          # K per metre
HAB_SCALE = 250.0             # habitat uint8 = round(habitat * 250)
HAB_NODATA = 255
ELEV_NODATA = -32768
TRIGGER_FEATURES = ("t20", "p26", "pulse", "tmax10", "tmin7")
TEMP_FEATURES = ("t20", "tmax10", "tmin7")      # lapse-corrected inputs of the trigger
VARS = ("tmean", "tmax", "tmin", "precip")
OM_DAILY = {"tmean": "temperature_2m_mean", "tmax": "temperature_2m_max",
            "tmin": "temperature_2m_min", "precip": "precipitation_sum"}


# ------------------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------------------
def merc_x(lon):
    return R_EARTH * np.radians(lon)


def merc_y(lat):
    return R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def merc_lon(x):
    return np.degrees(np.asarray(x) / R_EARTH)


def merc_lat(y):
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y) / R_EARTH)) - np.pi / 2)


def lonlat_to_pixel(lon, lat, geo):
    """Fractional (row, col) of lon/lat in the phone raster; pixel (r, c) covers
    [r, r+1) x [c, c+1), its centre is at (r + 0.5, c + 0.5).  ``geo`` is model.json ``raster``."""
    col = (merc_x(np.asarray(lon, float)) - geo["x0"]) / geo["px"]
    row = (geo["y_top"] - merc_y(np.asarray(lat, float))) / geo["px"]
    return row, col


def pixel_centres(geo):
    """(lon, lat) of every pixel centre: 1-D lon (nx,) and 1-D lat (ny,)."""
    lon = merc_lon(geo["x0"] + (np.arange(geo["nx"]) + 0.5) * geo["px"])
    lat = merc_lat(geo["y_top"] - (np.arange(geo["ny"]) + 0.5) * geo["px"])
    return lon, lat


# ------------------------------------------------------------------------------------------
# Weather: Open-Meteo JSON -> gap-free daily arrays
# ------------------------------------------------------------------------------------------
def parse_openmeteo(responses, n_points):
    """Open-Meteo JSON (list of locations, or one dict) -> (dates, {var: (T, n_points)}, elevation).

    Locations are matched to the requested points by position (the API keeps the request
    order).  ``dates`` is the contiguous list of ISO days from the earliest to the latest day
    returned by any location; days missing at a location are NaN.  ``null`` -> NaN.
    ``elevation`` is the ``elevation`` echoed by the API per point (the orography).
    """
    locs = responses if isinstance(responses, list) else [responses]
    if len(locs) != n_points:
        raise ValueError(f"{len(locs)} locations for {n_points} points")
    days = sorted({t for loc in locs for t in loc["daily"]["time"]})
    d0, d1 = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    dates = [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
    pos = {d: i for i, d in enumerate(dates)}
    out = {v: np.full((len(dates), n_points), np.nan) for v in VARS}
    elev = np.full(n_points, np.nan)
    for k, loc in enumerate(locs):
        idx = [pos[t] for t in loc["daily"]["time"]]
        for v in VARS:
            vals = loc["daily"][OM_DAILY[v]]
            out[v][idx, k] = [np.nan if x is None else float(x) for x in vals]
        if loc.get("elevation") is not None:
            elev[k] = float(loc["elevation"])
    return dates, out, elev


def fill_gaps(a):
    """Fill NaN in a (T, ...) array along time: linear inside gaps, nearest value outside
    (carry the last valid value forward, the first valid value backward).  Series without
    any valid value stay NaN."""
    a = np.array(a, dtype=float, copy=True)
    flat = a.reshape(a.shape[0], -1)
    t = np.arange(a.shape[0])
    for k in range(flat.shape[1]):
        ok = np.isfinite(flat[:, k])
        if ok.any() and not ok.all():
            flat[:, k] = np.interp(t, t[ok], flat[ok, k])
    return flat.reshape(a.shape)


def fill_missing_points(a, n_lat, n_lon):
    """Points (columns) that are NaN for the whole period take the series of the nearest
    valid point (Euclidean distance in lattice index space; ties: first in row-major order).
    ``a``: (T, n_lat * n_lon), lattice order lat-major."""
    bad = ~np.isfinite(a).any(axis=0)
    if not bad.any():
        return a
    if bad.all():
        raise ValueError("no valid weather at any point")
    ii, jj = np.divmod(np.arange(n_lat * n_lon), n_lon)
    good = np.flatnonzero(~bad)
    a = a.copy()
    for k in np.flatnonzero(bad):
        d = (ii[good] - ii[k]) ** 2 + (jj[good] - jj[k]) ** 2
        a[:, k] = a[:, good[int(np.argmin(d))]]
    return a


def daily_series(responses, n_lat, n_lon):
    """Open-Meteo JSON -> (dates, {var: (T, n_lat, n_lon)}, orography (n_lat, n_lon)).

    The full "daily aggregated series" step of the spec: parse, gap-fill every variable at
    every point, then replace points without any data.  Points are in lattice order
    (latitude ascending outer, longitude ascending inner)."""
    n = n_lat * n_lon
    dates, raw, elev = parse_openmeteo(responses, n)
    w = {}
    for v in VARS:
        a = fill_missing_points(fill_gaps(raw[v]), n_lat, n_lon)
        w[v] = a.reshape(len(dates), n_lat, n_lon)
    if np.isnan(elev).any():
        raise ValueError("a location has no elevation")
    return dates, w, elev.reshape(n_lat, n_lon)


# ------------------------------------------------------------------------------------------
# Coarse features (trailing windows, inclusive of the current day, partial at the start)
# ------------------------------------------------------------------------------------------
def roll_sum(x, w):
    return np.stack([x[max(0, t - w + 1):t + 1].sum(axis=0) for t in range(x.shape[0])])


def roll_mean(x, w):
    return np.stack([x[max(0, t - w + 1):t + 1].mean(axis=0) for t in range(x.shape[0])])


def roll_max(x, w):
    return np.stack([x[max(0, t - w + 1):t + 1].max(axis=0) for t in range(x.shape[0])])


def roll_min(x, w):
    return np.stack([x[max(0, t - w + 1):t + 1].min(axis=0) for t in range(x.shape[0])])


def coarse_features(w, p):
    """{tmean,tmax,tmin,precip: (T, ...)} + trigger params dict -> t20, p26, pulse, tmax10, tmin7."""
    lo, hi = p["pulse_lag"]
    pulse0 = (roll_sum(w["precip"], 3) >= p["pulse_min"]).astype(float)
    rm = roll_max(pulse0, hi - lo + 1)
    pulse = np.zeros_like(rm)
    if rm.shape[0] > lo:
        pulse[lo:] = rm[:rm.shape[0] - lo]
    return {"t20": roll_mean(w["tmean"], p["t_window"]),
            "p26": roll_sum(w["precip"], p["p_window"]),
            "pulse": pulse,
            "tmax10": roll_max(w["tmax"], p["heat_window"]),
            "tmin7": roll_min(w["tmin"], p["frost_window"])}


def trigger_score(f, p):
    """Fruiting trigger in 0..1 from the five features (any broadcastable shape)."""
    temp = np.exp(-0.5 * ((f["t20"] - p["t_opt"]) / p["t_sd"]) ** 2)
    rain = f["p26"] / (f["p26"] + p["p_half"])
    rain = 0.6 * rain + 0.4 * f["pulse"]
    heat = np.clip(1 - (f["tmax10"] - p["heat_tmax"]) / 5.0, 0.2, 1.0)
    frost = np.clip((f["tmin7"] - p["frost_tmin"]) / 3.0 + 1.0, 0.0, 1.0)
    return np.clip(temp * rain * heat * frost, 0.0, 1.0)


# ------------------------------------------------------------------------------------------
# Downscaling to the phone raster
# ------------------------------------------------------------------------------------------
def bilinear_weights(lat, lon, lat0, lon0, step, n_lat, n_lon):
    """Corner indices and fractions of points in the regular coarse lattice (edge values are
    extended outwards, no extrapolation).  ``lat`` / ``lon`` are arrays of equal shape."""
    fi = np.clip((np.asarray(lat, float) - lat0) / step, 0, n_lat - 1)
    fj = np.clip((np.asarray(lon, float) - lon0) / step, 0, n_lon - 1)
    i0 = np.minimum(np.floor(fi).astype(np.intp), n_lat - 2)
    j0 = np.minimum(np.floor(fj).astype(np.intp), n_lon - 2)
    return i0, j0, fi - i0, fj - j0


def bilinear(field, wts):
    i0, j0, di, dj = wts
    return (field[..., i0, j0] * (1 - di) * (1 - dj) + field[..., i0, j0 + 1] * ((1 - di) * dj)
            + field[..., i0 + 1, j0] * (di * (1 - dj)) + field[..., i0 + 1, j0 + 1] * (di * dj))


def downscale_features(feats_day, orog, elev, wts):
    """Coarse features of one day (each (n_lat, n_lon)) -> the five trigger features at the
    pixels: bilinear in lon/lat, temperature ones shifted by the lapse rate."""
    dz = LAPSE_RATE * (elev - bilinear(orog, wts))
    out = {}
    for n in TRIGGER_FEATURES:
        v = bilinear(feats_day[n], wts)
        out[n] = v + dz if n in TEMP_FEATURES else v
    return out


def month_of(day):
    return int(str(day)[5:7])


def season_factor(day, table):
    return float(table[str(month_of(day))])


def scale_score(habitat, season, trig, reference):
    return np.clip(habitat * season * trig / reference, 0.0, 1.0)


# ------------------------------------------------------------------------------------------
# Hotspots and colours
# ------------------------------------------------------------------------------------------
def pick_hotspots(rank, mask, min_score=0.15, min_sep_px=6.0, max_n=300):
    """Greedy de-clustered top cells of ``rank`` (2-D, NaN = none), as (row, col) list.

    Candidates: mask and rank >= min_score.  Order: descending rank, ties by row then column.
    Pick the best remaining, drop every candidate whose pixel distance is < min_sep_px."""
    ok = np.isfinite(rank) & (rank >= min_score) & mask.astype(bool)
    rows, cols = np.nonzero(ok)
    order = np.argsort(-rank[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]
    alive = np.ones(len(rows), bool)
    out = []
    lim = min_sep_px * min_sep_px
    for i in range(len(rows)):
        if not alive[i]:
            continue
        out.append((int(rows[i]), int(cols[i])))
        if len(out) >= max_n:
            break
        d2 = (rows - rows[i]).astype(float) ** 2 + (cols - cols[i]).astype(float) ** 2
        alive &= ~(d2 < lim)
    return out


def hotspot_rank(score, habitat, trig):
    """Ranking value: score + 0.001 * habitat * trigger (breaks ties among clipped scores)."""
    return score + 1e-3 * np.nan_to_num(habitat * trig)


def colour_lut(legend, n_levels=64, min_score=0.04, alpha_full_at=0.40, alpha_max=0.92):
    """(n_levels, 4) uint8 RGBA table: level i <-> score i / (n_levels - 1)."""
    v = np.array([s[0] for s in legend])
    rgb = np.array([[int(s[1][k:k + 2], 16) for k in (1, 3, 5)] for s in legend], float)
    s = np.linspace(0, 1, n_levels)
    lut = np.zeros((n_levels, 4), np.uint8)
    for c in range(3):
        lut[:, c] = np.round(np.interp(s, v, rgb[:, c]))
    a = np.clip((s - min_score) / (alpha_full_at - min_score), 0, 1) * alpha_max
    a[s < min_score] = 0
    lut[:, 3] = np.round(a * 255)
    return lut


def colourise(score, lut):
    """Score image (NaN = none) -> RGBA uint8 using the LUT (level = rint(score * (n - 1)))."""
    n = len(lut)
    lvl = np.clip(np.round(np.nan_to_num(score, nan=0.0) * (n - 1)), 0, n - 1).astype(np.intp)
    rgba = lut[lvl].copy()
    none = ~np.isfinite(score)
    rgba[none] = lut[0]
    rgba[none, 3] = 0
    return rgba


def decode_habitat(u8):
    """uint8 raster -> float habitat (NaN where nodata)."""
    h = u8.astype("float64") / HAB_SCALE
    h[u8 == HAB_NODATA] = np.nan
    return h


def decode_elev(i16):
    e = i16.astype("float64")
    e[i16 == ELEV_NODATA] = np.nan
    return e


def finite_or_none(x, digits=3):
    return None if x is None or not math.isfinite(x) else round(float(x), digits)
