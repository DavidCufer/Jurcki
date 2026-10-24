"""Export the on-phone "model package" for the Android app (no server).

The phone computes the maps itself: it downloads Open-Meteo weather for the 0.25 deg ERA5
lattice, turns it into the fruiting trigger and multiplies it with a static habitat raster.
Python stays the training / backtest lab and writes everything the Kotlin engine needs, plus
test vectors generated from the production code paths::

    python -m jurcki.app_package            # everything
    python -m jurcki.app_package --skip-stats --skip-validation   # quick: rasters + model.json

Outputs
-------
``android/app/src/main/assets/model/``
    ``habitat.u8.gz``  uint8, habitat 0..1 -> 0..250, 255 = nodata / outside Slovenia
    ``elev.i16.gz``    int16 little endian, metres, -32768 = nodata
    ``mask.u8.gz``     uint8, 1 = Slovenia land
    ``model.json``     the engine spec (parameters, weather request, formulas, geometry)
    ``stats.json``     numbers + texts of the app's "model / success" screen
    ``finds.json``     dated finds 2010+ with their backtest values
``android/app/src/test/resources/parity/``  JVM unit-test vectors (see ``build_vectors``)
``docs/APP_ENGINE.md`` is the human-readable spec (hand-written, kept in sync by the tests).

The rasters are on a Web-Mercator-aligned grid (north-up, row 0 = maxLat, columns even in
longitude, rows even in Mercator y), the same convention as the score PNGs of ``export``.
All rasters are binary, little endian, row-major, gzip-compressed (mtime 0, so the files are
reproducible).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import map_coordinates

from . import app_engine as eng
from . import config, export, features, grid, predict
from .model import prior, trigger
from .sources import weather

SCHEMA_VERSION = 1
ASSETS_DIR = config.ROOT / "android" / "app" / "src" / "main" / "assets" / "model"
VECTORS_DIR = config.ROOT / "android" / "app" / "src" / "test" / "resources" / "parity"

# --- raster geometry ----------------------------------------------------------------------
# Same outer extent as the PNGs.  Pixel size: 360 m in Web Mercator = 250 m on the ground at
# 46.1 N (the model grid is 250 m), i.e. 1009 x 662 pixels, the same count of cells as the
# 250 m EPSG:3794 grid.
EXTENT_LON = export.PNG_LON
EXTENT_LAT = export.PNG_LAT
PX_M = 360.0
MIN_SEP_PX = 6.0     # hotspot separation: 1.5 km / 250 m
COLOR_LEVELS = 64

# Open-Meteo request spec
OM_DAILY = ",".join(weather._OM_DAILY)
LOOKBACK_DAYS = predict.LOOKBACK
BATCH = 50


def phone_geometry() -> dict:
    """Pixel grid of the phone rasters (Mercator x0 / y_top / px are exact doubles)."""
    x0 = float(export.merc_x(EXTENT_LON[0]))
    nx = int(math.ceil((float(export.merc_x(EXTENT_LON[1])) - x0) / PX_M))
    y_bottom = float(export.merc_y(EXTENT_LAT[0]))
    ny = int(math.ceil((float(export.merc_y(EXTENT_LAT[1])) - y_bottom) / PX_M))
    y_top = y_bottom + ny * PX_M
    bounds = [float(export.merc_lon(x0)), float(export.merc_lat(y_bottom)),
              float(export.merc_lon(x0 + nx * PX_M)), float(export.merc_lat(y_top))]
    return {"crs": "EPSG:3857-aligned", "nx": nx, "ny": ny, "px": PX_M, "x0": x0, "y_top": y_top,
            "r_earth": export.R_EARTH, "bounds": [round(b, 7) for b in bounds],
            "ground_px_m_at_46N": round(PX_M * math.cos(math.radians(46.1)), 1)}


# ===========================================================================================
# Rasters
# ===========================================================================================
def _model_coords(geo: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fractional model-grid (row, col) of every phone pixel centre: cell centres at integers."""
    lon, lat = eng.pixel_centres(geo)
    lon2, lat2 = np.meshgrid(lon, lat)
    t = Transformer.from_crs(config.CRS_WGS84, config.CRS, always_xy=True)
    x, y = t.transform(lon2, lat2)
    x0, y0 = config.GRID_ORIGIN
    res = config.GRID_RES
    return (y0 - y) / res - 0.5, (x - x0) / res - 0.5


def resample_masked(src: np.ndarray, valid: np.ndarray, rc: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Bilinear resampling of ``src`` at fractional model-grid coordinates, using only cells
    where ``valid`` (renormalised weights); NaN where no valid cell contributes."""
    coords = np.stack(rc)
    s = np.where(valid, np.asarray(src, "float64"), 0.0)
    num = map_coordinates(s, coords, order=1, mode="constant", cval=0.0)
    den = map_coordinates(valid.astype("float64"), coords, order=1, mode="constant", cval=0.0)
    out = np.full(num.shape, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def build_rasters(geo: dict | None = None) -> dict:
    """Resample habitat / elevation / mask from the 250 m EPSG:3794 grid to the phone grid.

    * mask: nearest model cell (the cell that contains the pixel centre);
    * habitat, elev: bilinear between cell centres using only Slovenian land cells with data
      (weights renormalised), valid exactly where mask == 1; habitat is quantised to
      round(h * 250), elev to round(metres).
    """
    geo = geo or phone_geometry()
    rc = _model_coords(geo)
    mask_src = predict.load_mask()
    hab_src = np.nan_to_num(predict.load_habitat(), nan=0.0)
    elev_src = predict.load_elev()
    nr, nc = mask_src.shape
    ri = np.floor(rc[0] + 0.5).astype(int)
    ci = np.floor(rc[1] + 0.5).astype(int)
    inside = (ri >= 0) & (ri < nr) & (ci >= 0) & (ci < nc)
    mask = np.zeros(ri.shape, np.uint8)
    mask[inside] = mask_src[ri[inside], ci[inside]]
    land = mask.astype(bool)
    valid = mask_src & np.isfinite(hab_src) & np.isfinite(elev_src)
    hab = resample_masked(hab_src, valid, rc)
    elev = resample_masked(elev_src, valid, rc)
    land &= np.isfinite(hab) & np.isfinite(elev)
    mask = land.astype(np.uint8)
    hab_u8 = np.full(mask.shape, eng.HAB_NODATA, np.uint8)
    hab_u8[land] = np.clip(np.round(hab[land] * eng.HAB_SCALE), 0, eng.HAB_SCALE).astype(np.uint8)
    elev_i16 = np.full(mask.shape, eng.ELEV_NODATA, "<i2")
    elev_i16[land] = np.clip(np.round(elev[land]), -32767, 32767).astype("<i2")
    return {"habitat": hab_u8, "elev": elev_i16, "mask": mask}


RASTER_FILES = {"habitat": ("habitat.u8.gz", "<u1"), "elev": ("elev.i16.gz", "<i2"),
                "mask": ("mask.u8.gz", "<u1")}


def write_rasters(rasters: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, (fname, dt) in RASTER_FILES.items():
        raw = np.ascontiguousarray(rasters[name]).astype(dt).tobytes()
        (out_dir / fname).write_bytes(gzip.compress(raw, 9, mtime=0))
        sizes[fname] = (out_dir / fname).stat().st_size
    return sizes


def read_raster(path: Path, dtype: str, shape: tuple) -> np.ndarray:
    return np.frombuffer(gzip.decompress(Path(path).read_bytes()), dtype=dtype).reshape(shape)


def load_package_rasters(out_dir: Path, geo: dict) -> dict:
    shape = (geo["ny"], geo["nx"])
    return {n: read_raster(Path(out_dir) / f, dt, shape) for n, (f, dt) in RASTER_FILES.items()}


# ===========================================================================================
# Coarse weather lattice and Open-Meteo requests
# ===========================================================================================
def coarse_lattice() -> dict:
    """The 0.25 deg ERA5 points (lat asc outer, lon asc inner) + the mean-DEM elevation of
    the 0.25 deg box around each (the ``elevation`` passed to Open-Meteo, whole metres)."""
    pts = predict.era5_points()
    elev = [int(f"{e:.0f}") for e in predict.coarse_mean_elevation(pts)]
    lats = sorted({p[0] for p in pts})
    lons = sorted({p[1] for p in pts})
    return {"lat0": lats[0], "lon0": lons[0], "step": 0.25, "n_lat": len(lats), "n_lon": len(lons),
            "points": [[la, lo, z] for (la, lo), z in zip(pts, elev)]}


@contextmanager
def capture_requests(responder=None):
    """Replace ``weather._get_json``; records (url, params) and answers with ``responder``."""
    calls = []
    orig = weather._get_json

    def fake(url, params):
        calls.append((url, dict(params)))
        return responder(url, params)

    weather._get_json = fake
    try:
        yield calls
    finally:
        weather._get_json = orig


def forecast_requests(lattice: dict) -> list[dict]:
    """The forecast requests of the phone, built by the production client (one per batch)."""
    pts = [(p[0], p[1]) for p in lattice["points"]]
    elev = [p[2] for p in lattice["points"]]
    with capture_requests(lambda u, p: (_ for _ in ()).throw(_Stop())) as calls:
        for i in range(0, len(pts), BATCH):   # one call per batch: stop after the first request
            try:
                weather._om_fetch(weather.OPENMETEO_FORECAST, pts[i:i + BATCH],
                                  {"past_days": predict.OM_PAST_DAYS,
                                   "forecast_days": predict.OM_FORECAST_DAYS},
                                  True, BATCH, elev[i:i + BATCH], True)
            except _Stop:
                pass
    return [_phone_query(u, p) for u, p in calls]


class _Stop(Exception):
    pass


def _phone_query(url: str, params: dict) -> dict:
    """Request as the phone sends it: production params without the ``hourly`` block."""
    q = {k: v for k, v in params.items() if k != "hourly"}
    return {"url": url, "params": q}


def history_request_example(lattice: dict, start: str, end: str) -> list[dict]:
    pts = [(p[0], p[1]) for p in lattice["points"]]
    elev = [p[2] for p in lattice["points"]]
    out = []
    with capture_requests(lambda u, p: (_ for _ in ()).throw(_Stop())) as calls:
        for i in range(0, len(pts), BATCH):
            try:
                weather._om_fetch(weather.OPENMETEO_ARCHIVE, pts[i:i + BATCH],
                                  {"start_date": start, "end_date": end, "models": "era5"},
                                  False, BATCH, elev[i:i + BATCH], True)
            except _Stop:
                pass
        out = [_phone_query(u, p) for u, p in calls]
    return out


# ===========================================================================================
# Phone-equivalent engine on the package rasters (used for validation and the vectors)
# ===========================================================================================
class PhoneEngine:
    """The phone's computation, via ``app_engine``, on the package rasters."""

    def __init__(self, rasters: dict, geo: dict, model: dict):
        self.geo, self.model = geo, model
        self.mask = rasters["mask"].astype(bool)
        self.habitat = eng.decode_habitat(rasters["habitat"])
        self.elev = eng.decode_elev(rasters["elev"])
        lon, lat = eng.pixel_centres(geo)
        self.lon2, self.lat2 = np.meshgrid(lon, lat)
        lat_c = model["weather"]["lattice"]
        self.wts = eng.bilinear_weights(self.lat2, self.lon2, lat_c["lat0"], lat_c["lon0"],
                                        lat_c["step"], lat_c["n_lat"], lat_c["n_lon"])
        self.params = model["trigger"]["params"]
        self.season = model["score"]["season_factor"]
        self.reference = model["score"]["reference"]

    def day(self, coarse: dict, orog: np.ndarray, index: int, day: str) -> dict:
        """Score of one day; ``coarse`` = engine.coarse_features output (T, n_lat, n_lon)."""
        f = eng.downscale_features({k: v[index] for k, v in coarse.items()}, orog,
                                   np.nan_to_num(self.elev), self.wts)
        trig = eng.trigger_score(f, self.params)
        hab = np.nan_to_num(self.habitat)
        s = eng.season_factor(day, self.season)
        score = eng.scale_score(hab, s, trig, self.reference)
        for a in (trig, score):
            a[~self.mask] = np.nan
        return {"features": f, "trigger": trig, "score": score, "season": s}


def era5_inputs(day, model_params):
    """Production ERA5 weather for the 60 days up to ``day`` as engine inputs."""
    d = pd.Timestamp(day).date()
    ds = predict.Era5Source().load(d - timedelta(days=LOOKBACK_DAYS), d)
    co = predict.prepare_weather(ds)
    w = {k: co.w[k] for k in eng.VARS}
    return co, eng.coarse_features(w, model_params), co.orog


# ===========================================================================================
# Validation: phone-equivalent vs production
# ===========================================================================================
def _diff_stats(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> dict:
    d = (a - b)[m]
    return {"mean_abs": round(float(np.abs(d).mean()), 5), "rms": round(float(np.sqrt((d ** 2).mean())), 5),
            "p99_abs": round(float(np.percentile(np.abs(d), 99)), 4),
            "max_abs": round(float(np.abs(d).max()), 4), "bias": round(float(d.mean()), 5),
            "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 5)}


def compare_with_production(rasters: dict, geo: dict, model: dict,
                            days=("2025-09-20", "2024-07-25", "2023-10-10")) -> dict:
    """Score of the phone engine (package rasters, 360 m Mercator pixels) vs ``predict.score_day``
    (250 m grid), both from the same local ERA5 weather.  The production maps are resampled to
    the phone pixels (bilinear, Slovenian cells only) before comparing."""
    pe = PhoneEngine(rasters, geo, model)
    rc = _model_coords(geo)
    land = grid.load_mask()
    m = pe.mask
    out = {"days": {}, "weather": "local ERA5 0.25 deg history (data/interim/weather), no network",
           "note": "production maps are bilinearly resampled to the phone pixels before comparing"}
    # habitat alone
    hab_prod = resample_masked(np.nan_to_num(predict.load_habitat()), land, rc)
    out["habitat_raster_vs_resampled_source"] = _diff_stats(np.nan_to_num(pe.habitat), np.nan_to_num(hab_prod), m)
    elev_prod = resample_masked(predict.load_elev(), land, rc)
    out["elev_raster_vs_resampled_source_m"] = _diff_stats(np.nan_to_num(pe.elev), np.nan_to_num(elev_prod), m)
    for d in days:
        score_p, hab_p, trig_p = predict.score_day(d, "era5")
        sp = resample_masked(score_p, land, rc)
        tp = resample_masked(trig_p, land, rc)
        co, coarse, orog = era5_inputs(d, pe.params)
        i = (pd.Timestamp(d) - co.times[0]).days
        res = pe.day(coarse, orog, i, d)
        both = m & np.isfinite(sp) & np.isfinite(res["score"])
        a, b = res["score"], sp
        thr = np.nanpercentile(b[both], 95)
        top_p = both & (b >= thr)
        top_q = both & (a >= np.nanpercentile(a[both], 95))
        jac = float((top_p & top_q).sum() / max((top_p | top_q).sum(), 1))
        out["days"][d] = {"score": _diff_stats(a, b, both), "trigger": _diff_stats(res["trigger"], tp, both),
                          "mean_score_phone": round(float(a[both].mean()), 4),
                          "mean_score_production": round(float(b[both].mean()), 4),
                          "top5pct_overlap_jaccard": round(jac, 3),
                          "season_factor": res["season"], "n_pixels": int(both.sum())}
        # hotspots: production (250 m, export rule) vs phone (360 m px, same rule)
        hp = export.hotspots_geojson(score_p, hab_p, trig_p, predict.load_elev(), predict.load_mask())["features"]
        rank = eng.hotspot_rank(np.where(m, res["score"], np.nan), pe.habitat, res["trigger"])
        ph = eng.pick_hotspots(rank, m, 0.15, MIN_SEP_PX, 300)
        out["days"][d]["hotspots"] = _hotspot_compare(hp, ph, pe)
    return out


def _hotspot_compare(prod_feats: list, phone_cells: list, pe: PhoneEngine) -> dict:
    """How many production hotspots have a phone hotspot within 3 km (and vice versa)."""
    if not prod_feats or not phone_cells:
        return {"production": len(prod_feats), "phone": len(phone_cells)}
    plon = np.array([f["geometry"]["coordinates"][0] for f in prod_feats])
    plat = np.array([f["geometry"]["coordinates"][1] for f in prod_feats])
    rr, cc = np.array(phone_cells).T
    qlon, qlat = pe.lon2[rr, cc], pe.lat2[rr, cc]
    kx = 111.32 * math.cos(math.radians(46.1))
    dx = (plon[:, None] - qlon[None, :]) * kx
    dy = (plat[:, None] - qlat[None, :]) * 111.2
    dist = np.hypot(dx, dy)

    def share(d, n):
        return round(float((d[:n].min(axis=1) <= 3.0).mean()), 3)
    top = min(20, len(plon))
    return {"production": len(prod_feats), "phone": len(phone_cells),
            "share_of_production_top20_with_phone_hotspot_within_3km": share(dist, top),
            "share_of_production_all_with_phone_hotspot_within_3km": round(float((dist.min(axis=1) <= 3.0).mean()), 3),
            "share_of_phone_all_with_production_hotspot_within_3km": round(float((dist.min(axis=0) <= 3.0).mean()), 3)}


# ===========================================================================================
# model.json
# ===========================================================================================
PARAM_UNITS = {
    "t_opt": "degC", "t_sd": "degC", "t_window": "days", "p_window": "days", "p_half": "mm",
    "pulse_min": "mm per 3 days", "pulse_lag": "days [lo, hi]", "heat_tmax": "degC",
    "heat_window": "days", "frost_tmin": "degC", "frost_window": "days",
}


def build_model_json(geo: dict, lattice: dict, validation: dict | None) -> dict:
    tp = json.loads((config.PROCESSED / "trigger_params.json").read_text())
    ref = json.loads(predict.REFERENCE_PATH.read_text())
    params = trigger.params_to_dict(trigger.load_trigger_params())
    sf = {str(m): float(prior.season_factor(m)) for m in range(1, 13)}
    if tp["params"] != trigger.params_to_dict(trigger.params_from_dict(tp["params"])):
        raise RuntimeError("trigger_params.json does not round-trip")
    if ref["trigger_params"] != params:
        raise RuntimeError("score_reference.json was built with other trigger parameters: "
                           "run python -m jurcki.predict --build-reference")
    lut = eng.colour_lut(export.LEGEND, COLOR_LEVELS, export.MIN_SCORE, export.ALPHA_FULL_AT, export.ALPHA_MAX)
    reqs = forecast_requests(lattice)
    today_example = "2025-09-29"
    hist = history_request_example(lattice, (date.fromisoformat(today_example) - timedelta(days=LOOKBACK_DAYS)).isoformat(),
                                   today_example)
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec_doc": "docs/APP_ENGINE.md",
        "summary": "score(pixel, day) = clip(habitat(pixel) * season_factor(month(day)) * trigger(pixel, day) / reference, 0, 1)",
        "raster": {
            **geo,
            "description": "North-up, row 0 = maxLat, column 0 = minLon. Columns are even in longitude, rows even "
                           "in Web-Mercator y with the same pixel size, so lon/lat <-> row/col is analytic "
                           "(see formulas) and one image quad covers the map.",
            "pixel_centre": "lon = deg(x0 + (col + 0.5) * px / r_earth);  lat = deg(2*atan(exp((y_top - (row + 0.5) * px) / r_earth)) - pi/2)",
            "lonlat_to_rowcol": "col = (r_earth * rad(lon) - x0) / px;  row = (y_top - r_earth * ln(tan(pi/4 + rad(lat)/2))) / px;  "
                                "pixel = (floor(row), floor(col)); centres are at +0.5",
            "files": {
                "habitat": {"file": RASTER_FILES["habitat"][0], "dtype": "uint8", "value": "habitat = byte / 250 (0..1)",
                            "nodata": eng.HAB_NODATA, "scale": eng.HAB_SCALE},
                "elev": {"file": RASTER_FILES["elev"][0], "dtype": "int16 little endian", "value": "metres above sea level",
                         "nodata": eng.ELEV_NODATA},
                "mask": {"file": RASTER_FILES["mask"][0], "dtype": "uint8", "value": "1 = Slovenia land, 0 = outside"},
            },
            "encoding": "raw row-major array (nx * ny values, row 0 first), gzip-compressed; no header",
            "nodata_rule": "habitat and elev are nodata exactly where mask == 0; the score is undefined (transparent) there",
        },
        "trigger": {
            "params": params,
            "params_literature": trigger.params_to_dict(prior.PriorParams()),
            "units": PARAM_UNITS,
            "source": "data/processed/trigger_params.json (recommended = " + tp["recommended"] + ")",
            "note": "Only the trigger fields of PriorParams are used on the phone; the habitat fields of PriorParams "
                    "are not (habitat comes from the raster).",
            "formula": {
                "temp": "exp(-0.5 * ((t20 - t_opt) / t_sd)^2)",
                "rain": "0.6 * (p26 / (p26 + p_half)) + 0.4 * pulse",
                "heat": "clamp(1 - (tmax10 - heat_tmax) / 5, 0.2, 1.0)",
                "frost": "clamp((tmin7 - frost_tmin) / 3 + 1, 0.0, 1.0)",
                "trigger": "clamp(temp * rain * heat * frost, 0, 1)",
            },
        },
        "score": {
            "reference": ref["reference"],
            "reference_note": "fixed constant (99.9th percentile of habitat*season*trigger over forest cells, "
                              "Aug-Oct 2010-2025); NOT a per-day normalisation",
            "season_factor": sf,
            "season_factor_note": "key = month of the LOCAL calendar day (1..12) of the day being scored",
            "formula": "raw = habitat * season_factor * trigger;  score = clamp(raw / reference, 0, 1)",
            "habitat_nodata_in_mask": "treat as 0",
        },
        "lapse_rate_k_per_m": weather.LAPSE_RATE,
        "weather": {
            "provider": "Open-Meteo (https://open-meteo.com), no API key for non-commercial use",
            "timezone": weather.TZ,
            "lattice": lattice,
            "lattice_note": "points = [lat, lon, elevation_m]; order is latitude ascending (outer) then longitude "
                            "ascending (inner), index k = i_lat * n_lon + i_lon. elevation_m is the mean DEM height "
                            "(whole metres) of the 0.25 deg box around the point: pass it as `elevation` and use the "
                            "`elevation` echoed in the response as the point's orography.",
            "batch_size": BATCH,
            "daily_variables": OM_DAILY.split(","),
            "variable_map": {"tmean": "temperature_2m_mean", "tmax": "temperature_2m_max",
                             "tmin": "temperature_2m_min", "precip": "precipitation_sum"},
            "units": {"temperature": "degC", "precipitation": "mm per local day"},
            "daily_aggregation": "Do NOT request hourly data. Open-Meteo aggregates to local calendar days "
                                 "(00:00-24:00 Europe/Ljubljana, timezone=Europe/Ljubljana): mean / max / min of hourly 2 m "
                                 "temperature and the sum of hourly precipitation. The phone uses these four daily series as is.",
            "forecast_recent": {
                "use": "today and the following days (default), and any date within the last ~60 days",
                "url": weather.OPENMETEO_FORECAST,
                "past_days": predict.OM_PAST_DAYS, "forecast_days": predict.OM_FORECAST_DAYS,
                "models": "default (best_match) - do not pass `models`",
                "requests": reqs,
                "requests_note": "one GET per batch of 50 points (3 requests: 50, 50, 20); `latitude`, `longitude`, "
                                 "`elevation` are comma-separated lists of equal length in lattice order. These are the "
                                 "queries the production client builds, minus the `hourly` block the phone omits.",
                "window": "past_days + forecast_days = 69 daily values: today-62 .. today+6 (today = Europe/Ljubljana date)",
                "output_days": {"past": predict.OUT_PAST, "ahead": predict.OUT_AHEAD,
                                "meaning": "the app shows today-2 .. today+6 (9 days)"},
            },
            "history": {
                "use": "a past day older than the forecast window (backtest / browsing the past); needs 60 days before it",
                "url": weather.OPENMETEO_ARCHIVE,
                "models": "era5",
                "requests_example": hist,
                "window": f"start_date = D_first - {LOOKBACK_DAYS} days, end_date = D_last (inclusive, local dates)",
                "note": "models=era5 is the 0.25 deg ERA5 reanalysis the model was trained on. The archive lags real "
                        "time by a few days, so use `forecast_recent` for the last ~week. NOT verified from this build "
                        "environment (api hosts are blocked): check the first live response against the vectors' schema.",
            },
            "response_format": {
                "multiple_locations": "JSON array, one object per requested point, same order as the request",
                "single_location": "a single JSON object (the last batch can have 1 point if the lattice size changes)",
                "location_object": {"latitude": "echoed (nearest grid), ignore", "longitude": "echoed, ignore",
                                    "elevation": "metres used for the downscaling (= passed value) -> orography",
                                    "daily": {"time": ["YYYY-MM-DD", "..."], "temperature_2m_mean": ["number|null"],
                                              "temperature_2m_max": ["number|null"], "temperature_2m_min": ["number|null"],
                                              "precipitation_sum": ["number|null"]}},
                "errors": "HTTP 200 with {\"error\": true, \"reason\": \"...\"} is an error; HTTP 429 = rate limited "
                          "(wait ~60 s); retry 5xx with back-off",
                "match": "locations are matched to lattice points by POSITION in the array, not by the echoed coordinates",
            },
            "orography": "orography[i][j] = response.elevation of point (i, j) (falls back to the passed elevation_m). "
                         "It is the height at which the point's temperatures are valid.",
            "gap_filling": [
                "1. Days = contiguous calendar days from the earliest to the latest `time` returned by any point.",
                "2. null / missing value -> NaN, per point and per variable.",
                "3. Per point and variable: NaN inside the series -> linear interpolation in the day index; "
                "NaN after the last valid value -> that last value (carry forward); NaN before the first valid value -> "
                "the first valid value (carry backward). This applies to precipitation as well.",
                "4. A point/variable with no valid value at all -> copy the whole series of the nearest point that has "
                "data (Euclidean distance in lattice index space (i_lat, i_lon); ties: first in lattice order).",
            ],
        },
        "features": {
            "note": "All windows are TRAILING and INCLUDE the current day: window(t, w) = days t-w+1 .. t. "
                    "At the start of the series the window is truncated to the days that exist (means divide by the number "
                    "of days present, max / min / sum over the days present) - there is no NaN. They are computed per coarse "
                    "point on the unshifted Open-Meteo series (the lapse shift is applied later, per pixel).",
            "history_days_needed": LOOKBACK_DAYS,
            "defs": [
                {"name": "t20", "definition": "mean(tmean over window t_window)", "window_param": "t_window"},
                {"name": "p26", "definition": "sum(precip over window p_window)", "window_param": "p_window"},
                {"name": "tmax10", "definition": "max(tmax over window heat_window)", "window_param": "heat_window"},
                {"name": "tmin7", "definition": "min(tmin over window frost_window)", "window_param": "frost_window"},
                {"name": "pulse", "definition": "see pulse_steps"},
            ],
            "pulse_steps": [
                "rain3[t] = sum(precip over window 3 ending at t)  (truncated at the start)",
                "soak[t] = 1 if rain3[t] >= pulse_min else 0",
                "lo, hi = pulse_lag;  recent[t] = max(soak over window (hi - lo + 1) ending at t)  (truncated at the start)",
                "pulse[t] = recent[t - lo] if t >= lo else 0     (i.e. 1 if a soaking rain ended between lo and hi days ago)",
            ],
            "pulse_note": "pulse is a 0/1 flag at the coarse points; it becomes a smooth fraction 0..1 after bilinear "
                          "interpolation to the pixels (that is what production does).",
            "lapse_corrected": list(eng.TEMP_FEATURES),
            "not_lapse_corrected": ["p26", "pulse"],
        },
        "downscaling": {
            "method": "bilinear interpolation in lon/lat on the coarse lattice, of the FEATURES (not of the daily series)",
            "formulas": [
                "fi = clamp((lat - lat0) / step, 0, n_lat - 1);  fj = clamp((lon - lon0) / step, 0, n_lon - 1)",
                "i0 = min(floor(fi), n_lat - 2);  j0 = min(floor(fj), n_lon - 2);  di = fi - i0;  dj = fj - j0",
                "bil(F) = F[i0][j0]*(1-di)*(1-dj) + F[i0][j0+1]*(1-di)*dj + F[i0+1][j0]*di*(1-dj) + F[i0+1][j0+1]*di*dj",
                "orog_px = bil(orography);  dz = lapse_rate_k_per_m * (elev_px - orog_px)      (elev_px from the elev raster)",
                "t20_px = bil(t20) + dz;  tmax10_px = bil(tmax10) + dz;  tmin7_px = bil(tmin7) + dz;  p26_px = bil(p26);  pulse_px = bil(pulse)",
            ],
            "edges": "outside the lattice the nearest edge value is used (clamping, no extrapolation)",
            "pixel_lonlat": "pixel centre (see raster.pixel_centre)",
            "equivalence": "The lapse shift is constant over a window so it commutes with mean / max / min: shifting the "
                           "features equals shifting the daily series. The trigger is evaluated per pixel AFTER the "
                           "interpolation (it is not interpolated).",
        },
        "hotspots": {
            "source": "export.pick_hotspots / export.hotspots_geojson",
            "inputs": "score, habitat, trigger (and elev) of the pixels of one day",
            "rank": "rank = score + 0.001 * habitat * trigger   (Float32 arithmetic in production; the 0.001 term breaks "
                    "ties among clipped scores)",
            "candidates": "mask == 1 and rank >= min_score",
            "min_score": export.HOTSPOT_MIN_SCORE,
            "min_sep_px": MIN_SEP_PX,
            "max_n": export.HOTSPOT_MAX,
            "algorithm": [
                "sort candidates by descending rank; ties by ascending row, then ascending column (stable sort of the row-major scan)",
                "walk the list; skip candidates already suppressed; otherwise pick it and suppress every candidate "
                "(including itself) whose pixel distance sqrt(drow^2 + dcol^2) < min_sep_px  (strictly less)",
                "stop when max_n picks have been made",
            ],
            "min_sep_note": "production uses 1500 m on the 250 m grid (< 6 cells); the phone grid has ~250 m pixels, so 6 px.",
            "output": "picks sorted by descending rank: lon/lat of the pixel centre, score, habitat, trigger (0..1, 3 decimals), elev (m)",
        },
        "colormap": {
            "legend": export.LEGEND,
            "min_score": export.MIN_SCORE, "alpha_full_at": export.ALPHA_FULL_AT, "alpha_max": export.ALPHA_MAX,
            "levels": COLOR_LEVELS,
            "rule": ["level = clamp(rint(score * (levels - 1)), 0, levels - 1)   (rint = round half to even; NaN score -> transparent)",
                     "s = level / (levels - 1)",
                     "rgb: per channel linear interpolation between the legend stops at s, rounded half to even to 0..255",
                     "alpha = 0 if s < min_score else clamp((s - min_score) / (alpha_full_at - min_score), 0, 1) * alpha_max; "
                     "alpha255 = rint(alpha * 255)",
                     "pixels outside the mask are fully transparent"],
            "lut": [[int(v) for v in row] for row in lut],
            "lut_note": "RGBA for level 0..63; use this table directly - it is the exact output of the rule above",
        },
        "validation": validation,
        "tolerances": {
            "note": "The Kotlin engine may use Double or Float. Python (float64 here, float32 in production bilinear) "
                    "agrees with the vectors to these absolute tolerances",
            "daily_series": 1e-5, "features_coarse": 1e-4, "features_pixel": 1e-3,
            "trigger": 1e-4, "score": 2e-4, "pixel_rowcol": 1e-2,
        },
    }


# ===========================================================================================
# stats.json
# ===========================================================================================
VARIANT_LABEL = {
    "prior": ("Pravila iz literature (prior)", "Literature rules (prior)"),
    "core": ("Model: teren in pokrovnost", "Model: terrain and tree cover"),
    "full": ("Model: teren, pokrovnost in tip gozda (v aplikaciji)", "Model: terrain, tree cover and forest type (used in the app)"),
    "blend_core": ("Mešanica prior + model (teren)", "Blend prior + model (terrain)"),
    "blend_full": ("Mešanica prior + model (s tipom gozda)", "Blend prior + model (with forest type)"),
}
UNIT_LABEL = {
    "si_all": ("Vse slovenske najdbe", "All Slovenian finds"),
    "si_dated": ("Slovenske najdbe z datumom (GBIF)", "Slovenian dated finds (GBIF)"),
    "si_sfi": ("Slovenske najdbe ZGS (samo leto, ~3 km)", "Slovenian SFI records (year only, ~3 km)"),
    "foreign_dated": ("Najdbe v sosednjih državah z datumom", "Dated finds in neighbouring countries"),
    "foreign_sfi": ("Najdbe ZGS v tujini", "SFI records abroad"),
}
TRIGGER_ROWS = [
    ("season_only", "comb_tol", "Samo sezona (mesec)", "Season only (month)"),
    ("literature", "comb_tol", "Literatura: sezona x vreme", "Literature: season x weather"),
    ("tuned_in_sample", "comb_tol", "Prilagojeni parametri (vsa leta, optimistično)", "Tuned parameters (all years, optimistic)"),
    ("tuned_loyo", "comb_tol", "Prilagojeni parametri, preverjeno na izpuščenem letu", "Tuned parameters, leave-one-year-out"),
    ("lgbm_loyo", "comb_tol", "Strojno učenje (LightGBM), izpuščeno leto", "Machine learning (LightGBM), leave-one-year-out"),
    ("logit_loyo", "comb_tol", "Strojno učenje (logistična), izpuščeno leto", "Machine learning (logistic), leave-one-year-out"),
]
TRIGGER_ONLY_ROWS = [
    ("literature", "trig_tol", "Literatura: samo vreme", "Literature: weather only"),
    ("tuned_loyo", "trig_tol", "Prilagojeni parametri: samo vreme, izpuščeno leto", "Tuned: weather only, leave-one-year-out"),
]
PARAM_TEXT = {
    "t_opt": ("Najboljša povprečna temperatura zraka v zadnjih 20 dneh. Literatura (Bielefeld) navaja 13 °C; naši podatki iz Slovenije in Avstrije kažejo višji optimum.",
              "Best mean air temperature over the last 20 days. The literature (Bielefeld) says 13 degC; our Slovenian and Austrian finds point to a higher optimum."),
    "t_sd": ("Širina temperaturnega odziva: večja vrednost pomeni, da jurčki dobro prenašajo odmike od optimuma.",
             "Width of the temperature response: a larger value means more tolerance for departures from the optimum."),
    "t_window": ("Število dni za povprečno temperaturo (fiksno).", "Number of days for the mean temperature (fixed)."),
    "p_window": ("Število dni za seštevek padavin (fiksno).", "Number of days for the rain total (fixed)."),
    "p_half": ("Količina padavin v 26 dneh, pri kateri je dež-del sprožilca 50 %. Več dežja pomeni višji sprožilec, vendar z zasičenjem.",
               "Rain over 26 days at which the rain part of the trigger reaches 50 %. More rain raises the trigger, with saturation."),
    "pulse_min": ("Padavine v 3 dneh, ki štejejo za izdatno namočitev tal.",
                  "Rain in 3 days that counts as a soaking rain."),
    "pulse_lag": ("Koliko dni po izdatnem dežju se pojavijo plodovi: od-do.",
                  "How many days after a soaking rain the fruiting bodies appear: from-to."),
    "heat_tmax": ("Najvišja dnevna temperatura v zadnjih 10 dneh, nad katero sprožilec pada (vročinski val zaduši rast).",
                  "Daily maximum over the last 10 days above which the trigger drops (a heat spell suppresses fruiting)."),
    "heat_window": ("Število dni za vročinski kazalnik (fiksno).", "Number of days for the heat indicator (fixed)."),
    "frost_tmin": ("Najnižja nočna temperatura v zadnjih 7 dneh, pod katero sprožilec pada (slana).",
                   "Lowest night temperature over the last 7 days below which the trigger drops (frost)."),
    "frost_window": ("Število dni za kazalnik zmrzali (fiksno).", "Number of days for the frost indicator (fixed)."),
}
GLOSSARY = {
    "auc": ("AUC: verjetnost, da je prava najdba višje ocenjena kot naključna primerjava (0,5 = naključno, 1 = popolno). "
            "Pri 'kdaj' je primerjava z drugimi dnevi sezone na istem mestu, pri 'kje' z naključnimi gozdnimi celicami.",
            "AUC: the probability that a real find is ranked above a random comparison (0.5 = chance, 1 = perfect). "
            "For 'when' the comparison is the other days of the season at the same place; for 'where' random forest cells."),
    "top10": ("Delež najdb, ki so med najboljšimi 10 % (dnevi ali celice).",
              "Share of finds that fall in the best 10 % (days or cells)."),
    "top20": ("Delež najdb, ki so med najboljšimi 20 %.", "Share of finds that fall in the best 20 %."),
    "tol": ("Toleranca ±3 dni: najdba se zabeleži z zamudo in plodovi rastejo več dni, zato se šteje najboljši rezultat v oknu ±3 dni.",
            "Tolerance ±3 days: a find is logged with a delay and a flush lasts several days, so the best score within ±3 days counts."),
    "loyo": ("Izpuščeno leto (LOYO): za vsako leto so parametri izbrani na drugih letih in preverjeni na tem letu - poštena ocena.",
             "Leave-one-year-out (LOYO): for each year the parameters are chosen on the other years and tested on that year - an honest estimate."),
    "blocks": ("Prostorski bloki: cele kvadratne regije (10 km) so pri učenju izpuščene in nato napovedane.",
               "Spatial blocks: whole square regions (10 km) are held out during training and then predicted."),
}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(f"{float(o):.6g}")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _metric(m: dict | None) -> dict | None:
    if not m:
        return None
    return {"n": int(m["n"]), "auc": round(m["auc"], 4), "hit_top10": round(m["hit_top10"], 4),
            "hit_top20": round(m["hit_top20"], 4)}


def run_trigger_backtest(n_candidates: int = 1000, seed: int = 0) -> dict:
    """Re-run the trigger evaluation (deterministic, ERA5 local) and return the report plus the
    per-find held-out percentile vectors needed for finds.json / the per-year table."""
    rep = trigger.run(n_candidates, seed, ml=False)
    panel, cands, cv = rep["panel"], rep["cands"], rep["cv"]
    season = trigger.season_scores(panel)
    lit = trigger.all_variants(panel, trigger.trigger_scores(panel, cands[0]), season)
    base = trigger.all_variants(panel, None, season)
    rep["vecs"] = {"season": base, "literature": lit, "tuned_loyo": cv["held"]}
    return rep


def build_stats(rep: dict, finds_summary: dict) -> dict:
    hm = json.loads((config.PROCESSED / "habitat_metrics.json").read_text())
    tp = json.loads((config.PROCESSED / "trigger_params.json").read_text())
    saved = tp["metrics"]
    # the re-run must reproduce the saved numbers
    for key in ("literature", "tuned_loyo", "season_only"):
        a, b = rep[key]["comb_tol"]["auc"], saved[key]["comb_tol"]["auc"]
        if abs(a - b) > 1e-3:
            raise RuntimeError(f"backtest re-run differs from trigger_params.json for {key}: {a} vs {b}")
    # habitat
    cv_h = {}
    for km, cv in hm["cv"].items():
        cv_h[km] = {v: {u: _metric(cv[v].get(u)) | ({"fold_auc": [round(x, 3) for x in cv[v][u]["fold_auc"]]}
                                                    if "fold_auc" in cv[v].get(u, {}) else {})
                        for u in UNIT_LABEL if u in cv[v]} for v in VARIANT_LABEL if v in cv}
    habitat = {
        "title": ("Kje? Primernost rastišča", "Where? Habitat suitability"),
        "description": ("Statični zemljevid primernosti (nadmorska višina, naklon, ekspozicija, pokritost z gozdom, tip gozda, rob gozda). "
                        "Model je naučen na najdbah (GBIF/iNaturalist in ZGS) proti ozadju iz gozdnih celic in preverjen s prostorskimi bloki.",
                        "Static suitability map (elevation, slope, aspect, tree cover, forest type, forest edge). The model is trained on finds "
                        "(GBIF/iNaturalist and SFI) against a forest-cell background and checked with spatial blocks."),
        "final_variant": hm["final"]["variant"], "final_blend_with_prior": hm["final"]["blend_with_prior"],
        "block_km_main": hm["block_km"],
        "variants": {k: {"sl": v[0], "en": v[1]} for k, v in VARIANT_LABEL.items()},
        "units": {k: {"sl": v[0], "en": v[1]} for k, v in UNIT_LABEL.items()},
        "cv": cv_h,
        "n_units": hm["n_units"],
        "importance": {k: round(v, 4) for k, v in hm["importance_full"].items()},
        "sanity_final": hm["sanity"]["final"],
    }
    # trigger
    rows = []
    for key, var, sl, en in TRIGGER_ROWS + TRIGGER_ONLY_ROWS:
        src = saved.get(key) or rep.get(key)
        if src is None:
            continue
        rows.append({"key": f"{key}:{var}", "sl": sl, "en": en, "tolerant_3d": _metric(src.get(var)),
                     "exact_day": _metric(src.get(var.replace("tol", "exact")))})
    ci = {k: [round(x, 4) for x in v] for k, v in rep["gain_ci"].items()}
    # per year
    held = rep["vecs"]
    f = rep["panel"].finds
    years = rep["panel"].series["year"].to_numpy()[f["sid"].to_numpy()]
    per_year = []
    for y in sorted(set(years.tolist())):
        sel = years == y
        row = {"year": int(y), "n": int(sel.sum())}
        for name in ("tuned_loyo", "literature", "season"):
            p = held[name]["comb_tol"][sel]
            row[f"auc_{name}"] = round(float(np.nanmean(p)), 3)
            row[f"hit_top20_{name}"] = round(float(np.mean(p[np.isfinite(p)] >= 0.8)), 3)
        per_year.append(row)
    eff = rep["effort"]
    trig = {
        "title": ("Kdaj? Sprožilec rasti", "When? Fruiting trigger"),
        "description": ("Sprožilec iz vremena (povprečna temperatura 20 dni, padavine 26 dni, izdaten dež, vročina, zmrzal) krat sezonski faktor. "
                        "Vsaka najdba se primerja z drugimi dnevi iste sezone (1. 5. - 15. 11.) na istem mestu v istem letu.",
                        "Weather trigger (20-day mean temperature, 26-day rain, soaking rain, heat, frost) times a seasonal factor. "
                        "Each find is compared with the other days of the same season (1 May - 15 Nov) at the same place in the same year."),
        "primary_metric": tp["primary_metric"],
        "recommended": tp["recommended"],
        "rows": rows,
        "ml_wins": tp["ml_wins"],
        "ml_note": ("Strojno učenje ni bistveno boljše od preprostega modela s parametri iz literature, zato aplikacija uporablja slednjega."
                    if not tp["ml_wins"] else "Strojno učenje je boljše."),
        "ml_note_en": ("Machine learning is not clearly better than the simple parametric model, so the app uses the latter."
                       if not tp["ml_wins"] else "Machine learning is better."),
        "gain_ci_auc": {"description": "mean AUC difference [mean, 5 %, 95 %], bootstrap over years", "values": ci},
        "tuned_gain_loyo": tp["tuned_gain_loyo"],
        "n_finds": rep["n_finds"], "n_series": rep["n_series"], "years": rep["years"],
        "n_candidates": rep["n_candidates"],
    }
    # params
    lit, tuned = tp["literature"], tp["params"]
    params = []
    for k in trigger.TRIGGER_FIELDS:
        params.append({"name": k, "unit": PARAM_UNITS[k], "literature": lit[k], "tuned": tuned[k],
                       "used_in_app": "tuned", "sl": PARAM_TEXT[k][0], "en": PARAM_TEXT[k][1]})
    weekend = {
        "n": eff["n"], "off_day_share": round(eff["off_day_share"], 4), "baseline_share": round(eff["baseline_share"], 4),
        "p_value": float(f"{eff['p_value']:.3g}"),
        "sl": f"{eff['off_day_share']:.0%} najdb je bilo zabeleženih ob vikendih in praznikih, čeprav je takih dni v sezoni {eff['baseline_share']:.0%}. "
              "Najdbe torej delno merijo, kdaj ljudje hodijo v gozd, ne le kdaj jurčki rastejo; zato so ocene modela nekoliko optimistične ali popačene.",
        "en": f"{eff['off_day_share']:.0%} of finds were logged on weekends and public holidays, although such days are {eff['baseline_share']:.0%} of the season. "
              "The finds therefore partly measure when people go to the forest, not only when boletes fruit, which biases the backtest.",
    }
    return _clean({
        "schema_version": SCHEMA_VERSION,
        "generated": date.today().isoformat(),
        "habitat": habitat, "trigger": trig, "per_year": per_year,
        "per_year_note": {"sl": "Preverjeno na izpuščenem letu: parametri so izbrani na drugih letih. hit_top20 = delež najdb med najboljšimi 20 % dni.",
                          "en": "Held-out: parameters chosen on the other years. hit_top20 = share of finds among the best 20 % of days."},
        "params": params, "weekend_effect": weekend,
        "data": {**finds_summary, "backtest_finds_total": rep["n_finds"],
                 "backtest_note": {"sl": "Preizkus uporablja vse datirane najdbe 2010+ s koordinatno natančnostjo do 10 km v oknu ERA5 in v sezoni 1. 5. - 15. 11.; nekatere ležijo severno od zemljevida.",
                                   "en": "The backtest uses all dated 2010+ finds with coordinate uncertainty up to 10 km inside the ERA5 window and the 1 May - 15 Nov season; a few lie north of the map."}},
        "glossary": {k: {"sl": v[0], "en": v[1]} for k, v in GLOSSARY.items()},
    })


# ===========================================================================================
# finds.json
# ===========================================================================================
DATASET_NAMES = {
    "50c9509d-22c7-4a22-a47d-8c48425ef4a7": "iNaturalist",
    "8a863029-f435-446a-821e-275f4f641165": "Observation.org",
}


def build_finds(rep: dict | None, geo: dict) -> tuple[dict, dict]:
    """finds.json content and the data counts for stats.json (finds 2010+ inside the raster bounds)."""
    g = pd.read_parquet(predict.GBIF_PATH)
    g = g[g["is_jurcki"].astype(bool) & g["date"].notna() & (g["date"] >= "2010-01-01")].copy()
    b = geo["bounds"]
    g = g[g.decimallongitude.between(b[0], b[2]) & g.decimallatitude.between(b[1], b[3])]
    d = pd.DataFrame({
        "date": g["date"].dt.normalize().dt.strftime("%Y-%m-%d"),
        "lon": g.decimallongitude.round(3), "lat": g.decimallatitude.round(3),
        "species": g["species"], "country": g["countrycode"],
        "source": [DATASET_NAMES.get(k) or (None if pd.isna(i) else str(i))
                   for k, i in zip(g["datasetkey"], g["institutioncode"])],
        "precision_m": g["coordinateuncertaintyinmeters"].round().astype("Int64"),
    })
    n_raw = len(d)
    d = d.drop_duplicates(["date", "lon", "lat"]).reset_index(drop=True)   # same rule as trigger.load_finds
    rows, cols = grid.lonlat_to_rowcol(d["lon"].to_numpy(), d["lat"].to_numpy())
    hab_src = np.nan_to_num(predict.load_habitat())
    nr, nc = hab_src.shape
    ok = (rows >= 0) & (rows < nr) & (cols >= 0) & (cols < nc)
    hv = np.full(len(d), np.nan)
    hv[ok] = hab_src[rows[ok], cols[ok]]
    d["habitat"] = np.round(hv, 3)
    for c in ("p_trigger", "p_season_trigger"):
        d[c] = np.nan
    if rep is not None:
        f = rep["panel"].finds
        ser = rep["panel"].series
        key = {(pd.Timestamp(r.date).strftime("%Y-%m-%d"), round(ser["lon"].iat[r.sid], 3), round(ser["lat"].iat[r.sid], 3)): i
               for i, r in f.iterrows()}
        trig_v = rep["cv"]["held"]["trig_tol"]
        comb_v = rep["cv"]["held"]["comb_tol"]
        idx = [key.get((a, b_, c)) for a, b_, c in zip(d["date"], d["lon"], d["lat"])]
        for i, k in enumerate(idx):
            if k is not None:
                d.loc[i, "p_trigger"] = round(float(trig_v[k]), 3)
                d.loc[i, "p_season_trigger"] = round(float(comb_v[k]), 3)
    d = d.sort_values(["date", "lon", "lat"]).reset_index(drop=True)
    recs = []
    for r in d.itertuples(index=False):
        recs.append({
            "date": r.date, "lon": float(r.lon), "lat": float(r.lat), "species": r.species, "country": r.country,
            "source": r.source, "precision_m": None if pd.isna(r.precision_m) else int(r.precision_m),
            "habitat": None if not np.isfinite(r.habitat) else float(r.habitat),
            "p_trigger": None if not np.isfinite(r.p_trigger) else float(r.p_trigger),
            "p_season_trigger": None if not np.isfinite(r.p_season_trigger) else float(r.p_season_trigger),
        })
    out = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "GBIF occurrences (incl. iNaturalist), Boletus edulis / reticulatus / pinophilus / aereus, 2010+, inside the raster bounds",
        "fields": {
            "date": "YYYY-MM-DD (observation day)", "lon": "WGS84, 3 decimals (~100 m)", "lat": "WGS84, 3 decimals",
            "species": "GBIF species name", "country": "ISO 3166 alpha-2", "source": "dataset / institution",
            "precision_m": "reported coordinate uncertainty in metres, null if unknown",
            "habitat": "habitat value 0..1 of the 250 m model grid at the find, null if outside the grid",
            "p_trigger": "backtest: temporal percentile (0..1) of the find day among the other days of the May 1 - Nov 15 "
                         "season at the same place, using the WEATHER TRIGGER only, +-3 day tolerant, leave-one-year-out "
                         "(0.5 = chance, 1 = best day of the season); null = not part of the backtest",
            "p_season_trigger": "same for season factor x trigger (the app's 'when' score)",
        },
        "n": len(recs), "finds": recs,
    }
    n_bt = int(d["p_trigger"].notna().sum())
    summary = {
        "finds_2010plus_in_bounds": len(recs), "duplicates_removed": n_raw - len(recs), "in_backtest": n_bt,
        "by_country": {k: int(v) for k, v in d["country"].value_counts().items()},
        "by_source": {str(k): int(v) for k, v in d["source"].fillna("unknown").value_counts().items()},
        "by_species": {k: int(v) for k, v in d["species"].value_counts().items()},
        "by_year": {str(k): int(v) for k, v in pd.to_datetime(d["date"]).dt.year.value_counts().sort_index().items()},
        "years_covered": [int(pd.to_datetime(d["date"]).dt.year.min()), int(pd.to_datetime(d["date"]).dt.year.max())],
    }
    hm = json.loads((config.PROCESSED / "habitat_metrics.json").read_text())
    summary["habitat_training_units"] = hm["n_units"]
    summary["habitat_training_note"] = {
        "sl": "Model rastišča je naučen na datiranih najdbah GBIF in na zapisih Zavoda za gozdove Slovenije (samo leto, ~3 km natančnost); ti niso na zemljevidu najdb.",
        "en": "The habitat model is trained on dated GBIF finds and on Slovenian Forestry Institute records (year only, ~3 km accuracy); the latter are not in the finds list."}
    return out, summary


# ===========================================================================================
# Parity test vectors
# ===========================================================================================
VECTOR_TODAY = "2025-09-29"       # the synthetic "today" of the vectors
VECTOR_DAYS = ("2025-09-30", "2025-10-02")   # Sep / Oct: different season factor


def synthetic_openmeteo(lattice: dict, today: str = VECTOR_TODAY, seed: int = 11) -> list[dict]:
    """Open-Meteo-style forecast response (daily block only) for the lattice, deterministic.

    62 past days + today + 6 forecast days.  Shared synoptic anomaly + elevation + two spatially
    patchy soaking rains (so the pulse is fractional after interpolation) + a few nulls."""
    rng = np.random.default_rng(seed)
    t0 = date.fromisoformat(today)
    days = [t0 + timedelta(days=i) for i in range(-predict.OM_PAST_DAYS, predict.OM_FORECAST_DAYS)]
    T = len(days)
    pts = lattice["points"]
    lat = np.array([p[0] for p in pts]); lon = np.array([p[1] for p in pts]); el = np.array([p[2] for p in pts], float)
    doy = np.array([d.timetuple().tm_yday for d in days], float)
    anom = np.zeros(T)
    for t in range(1, T):
        anom[t] = 0.8 * anom[t - 1] + rng.normal(0, 1.6)
    base = 16.5 + 7.0 * np.cos(2 * np.pi * (doy - 205) / 365.0)
    tmean = base[:, None] + anom[:, None] - 0.0065 * el[None, :] + 0.6 * (46.0 - lat)[None, :] + rng.normal(0, 0.35, (T, len(pts)))
    tmax = tmean + 4.5 + rng.uniform(0, 2.5, tmean.shape)
    tmin = tmean - 4.5 - rng.uniform(0, 2.5, tmean.shape)
    wet = rng.random(T) < 0.30
    amount = rng.exponential(2.2, T)
    patch = np.clip(1.0 + 0.25 * np.sin(lon[None, :] * 2.1 + np.arange(T)[:, None]) + rng.normal(0, 0.15, (T, len(pts))), 0, None)
    precip = (wet * amount)[:, None] * patch
    ix = {d.isoformat(): i for i, d in enumerate(days)}
    # soaking rain A (west) 17 days before today, B (south) 13 days before
    a = ix[(t0 - timedelta(days=17)).isoformat()]
    b = ix[(t0 - timedelta(days=13)).isoformat()]
    precip[a] = 30.0 * np.clip(1.2 - (lon - 13.25) / 2.0, 0, 1.2)      # soaking only west of ~15 E
    precip[b] = 28.0 * np.clip((46.2 - lat) / 0.9, 0, 1.2)             # soaking only south of ~46 N
    precip = np.round(precip, 1)
    tmean, tmax, tmin = (np.round(x, 1) for x in (tmean, tmax, tmin))
    daily = {"temperature_2m_mean": tmean, "temperature_2m_max": tmax,
             "temperature_2m_min": tmin, "precipitation_sum": precip}
    times = [d.isoformat() for d in days]
    out = []
    for k, (la, lo, z) in enumerate(pts):
        dd = {"time": list(times)}
        for name, arr in daily.items():
            dd[name] = [float(x) for x in arr[:, k]]
        out.append({"latitude": la, "longitude": lo, "elevation": float(z), "utc_offset_seconds": 7200,
                    "timezone": weather.TZ, "daily": dd})
    # nulls: interior gap, leading and trailing gaps, single holes
    n_pts = len(pts)
    k1, k2, k3, k4 = 3, n_pts // 2, n_pts // 2 + 1, n_pts - 4
    for name in daily:
        _null(out[k1], name, T - 39, T - 37)      # two days in a row, ~5 weeks ago (inside every window)
        _null(out[k2], name, 0, 1)                # first day missing
        _null(out[k3], name, T - 1, T)            # last forecast day missing
    _null(out[k4], "precipitation_sum", T - 25, T - 24)    # one missing rain day
    _null(out[k4], "temperature_2m_max", T - 12, T - 11)   # one missing tmax day
    return out


def _null(loc: dict, name: str, a: int, b: int) -> None:
    for i in range(a, b):
        loc["daily"][name][i] = None


SMALL_INDEX = [i * 15 + j for i in (3, 4) for j in (4, 5, 6)]   # a 2 x 3 block of the lattice


def small_case(full: list[dict], lattice: dict) -> tuple[list[dict], list[list]]:
    """2 x 3 lattice block with a variety of nulls: the two western points have no data at all
    (filled from their unique nearest valid neighbour), the others interior / leading / trailing /
    single holes.  Fresh copies of the responses (no nulls of the full set)."""
    resp = json.loads(json.dumps([full[k] for k in SMALL_INDEX]))
    pts = [lattice["points"][k] for k in SMALL_INDEX]
    T = len(resp[0]["daily"]["time"])
    names = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
    for r in (resp[0], resp[3]):
        for n in names:
            _null(r, n, 0, T)                           # entire point missing
    for n in names:
        _null(resp[1], n, T - 41, T - 38)               # 3-day interior gap, ~5.5 weeks ago
        _null(resp[4], n, 0, 2)                         # first 2 days
        _null(resp[2], n, T - 2, T)                     # last 2 days (forecast)
    _null(resp[5], "precipitation_sum", T - 24, T - 23)       # single rain hole
    _null(resp[5], "temperature_2m_mean", T - 33, T - 32)     # single tmean hole
    _null(resp[5], "temperature_2m_max", T - 10, T - 9)       # single tmax hole
    _null(resp[5], "temperature_2m_min", T - 28, T - 27)      # single tmin hole
    return resp, pts


def _production_payload(responses: list[dict]) -> list[dict]:
    """Add the (unused) hourly block that the production client requires."""
    out = []
    for loc in responses:
        n = len(loc["daily"]["time"])
        t0 = date.fromisoformat(loc["daily"]["time"][0])
        hours = [(datetime(t0.year, t0.month, t0.day) + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(24 * n)]
        hourly = {"time": hours, "snow_depth": [0.0] * len(hours)}
        for k in weather._OM_HOURLY_FORECAST_LEAN:
            hourly.setdefault(k, [0.2] * len(hours))
        out.append({**loc, "hourly": hourly})
    return out


def production_dataset(responses: list[dict], lattice_points: list):
    """Open-Meteo-style responses -> xarray Dataset through the production client code."""
    payload = _production_payload(responses)
    pts = [(p[0], p[1]) for p in lattice_points]
    elev = [p[2] for p in lattice_points]

    def answer(url, params):
        la = [float(x) for x in params["latitude"].split(",")]
        lo = [float(x) for x in params["longitude"].split(",")]
        idx = [next(i for i, (a, b) in enumerate(pts) if a == x and b == y) for x, y in zip(la, lo)]
        res = [payload[i] for i in idx]
        return res if len(res) > 1 else res[0]

    with capture_requests(answer):
        ds = weather.fetch_openmeteo_forecast(pts, past_days=predict.OM_PAST_DAYS,
                                              forecast_days=predict.OM_FORECAST_DAYS, batch=BATCH,
                                              elevations=elev, lean=True)
    return ds


def _round_list(a, n=6):
    return [None if not np.isfinite(x) else round(float(x), n) for x in np.ravel(a)]


def build_vectors(rasters: dict, geo: dict, model: dict, out_dir: Path) -> dict:
    """Write the JVM test vectors (all values come from the production code paths)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lattice = model["weather"]["lattice"]
    params = trigger.load_trigger_params()
    pdict = model["trigger"]["params"]
    reference = model["score"]["reference"]
    tol = model["tolerances"]
    written = {}

    def dump(name, obj):
        p = out_dir / name
        p.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        written[name] = p.stat().st_size

    # ---- (a) small weather response: daily series + features + trigger per point/day -----
    full = synthetic_openmeteo(lattice)
    # (a) 2 x 3 block with nulls, through the production client + prepare_weather + dynamic_feature_arrays
    resp_s, pts_s = small_case(full, lattice)
    ds_s = production_dataset(resp_s, pts_s)
    co = predict.prepare_weather(ds_s)
    fa = features.dynamic_feature_arrays(co.w, params)
    trig_c = np.clip(prior.trigger_score(fa, params), 0, 1)
    dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in co.times]
    ij = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]    # lat-major: point k = i * 3 + j
    small = {
        "description": "Open-Meteo forecast response (daily block only, as the phone requests it) for a 2 x 3 block of the "
                       "lattice (lat-major order: lats 46.0, 46.25; lons 14.25, 14.5, 14.75), 69 days (62 past + today + 6), "
                       "with null gaps. Points 0 and 3 have NO data at all (filled from the nearest valid point: point 1 and 4). "
                       "`expected` is what production computes from it: the daily series after gap filling, the five coarse "
                       "features and the trigger (clamped 0..1) at each coarse point. Arrays are [point][day].",
        "today": VECTOR_TODAY, "params": pdict, "lattice": {"lat0": pts_s[0][0], "lon0": pts_s[0][1], "step": 0.25,
                                                            "n_lat": 2, "n_lon": 3},
        "points": pts_s, "tolerance": {"daily": tol["daily_series"], "features": tol["features_coarse"],
                                       "trigger": tol["features_coarse"]},
        "responses": resp_s,
        "expected": {
            "dates": dates,
            "orography": [float(co.orog[i, j]) for i, j in ij],
            "daily": {v: [_round_list(co.w[v][:, i, j], 6) for i, j in ij] for v in eng.VARS},
            "features": {n: [_round_list(fa[n][:, i, j], 6) for i, j in ij] for n in eng.TRIGGER_FEATURES},
            "trigger": [_round_list(trig_c[:, i, j], 6) for i, j in ij],
        },
    }
    dump("weather_small.json", small)

    # ---- (b) cells: downscaled features, trigger, score for 2 days ------------------------
    ds_full = production_dataset(full, lattice["points"])
    reqs = forecast_requests(lattice)
    dump("weather_full.json", {
        "description": "Synthetic Open-Meteo forecast response for ALL lattice points (array in lattice order, 69 days, with "
                       "nulls) and the 3 requests the phone sends for it (query parameters).",
        "today": VECTOR_TODAY, "requests": reqs, "responses": full})
    cells, pe = _pick_cells(rasters, geo, model)
    cells_fg = predict.FineGrid(np.array([[c["lon"] for c in cells]]), np.array([[c["lat"] for c in cells]]),
                                np.array([[c["elev"] for c in cells]], "float32"))
    habitat = np.array([[c["habitat"] for c in cells]], "float32")
    src = predict.OpenMeteoSource(points=[(p[0], p[1]) for p in lattice["points"]],
                                  elevations=[p[2] for p in lattice["points"]], fetch=lambda *a, **k: ds_full)
    feats_by_day = {d.isoformat(): f for d, f in predict.iter_fine_features(
        VECTOR_DAYS, params, src, predict.TRIGGER_FEATURES, cells_fg)}
    results = {r.day.isoformat(): r for r in predict.score_days(VECTOR_DAYS, src, params, reference, cells_fg, habitat)}
    exp_days = {}
    for day in VECTOR_DAYS:
        f, r = feats_by_day[day], results[day]
        exp_days[day] = {
            "season_factor": r.season,
            "features": {n: _round_list(f[n][0], 7) for n in predict.TRIGGER_FEATURES},
            "trigger": _round_list(np.clip(prior.trigger_score({k: v[0] for k, v in f.items()}, params), 0, 1), 7),
            "raw": _round_list(predict.raw_score(habitat[0], r.season, r.trigger[0]), 7),
            "score": _round_list(r.score[0], 7),
        }
    dump("cells.json", {
        "description": "Pixels of the package raster (and 3 synthetic edge cases) with expected downscaled features, trigger "
                       "and score for two days, computed by production predict.iter_fine_features / score_days on the "
                       "weather in weather_full.json. habitat and elev are the DECODED raster values (habitat = byte/250). "
                       "`row`/`col` are null for synthetic cells.",
        "weather_file": "weather_full.json", "reference": reference, "params": pdict,
        "season_factor_table": model["score"]["season_factor"], "days": list(VECTOR_DAYS),
        "tolerance": {"features": tol["features_pixel"], "trigger": tol["trigger"], "score": tol["score"]},
        "cells": cells, "expected": exp_days,
    })

    # ---- (c) geometry --------------------------------------------------------------------
    places = {"Ljubljana": (14.5058, 46.0569), "Maribor": (15.6459, 46.5547), "Koper": (13.7301, 45.5481),
              "Bled": (14.1146, 46.3683), "Murska Sobota": (16.1664, 46.6625), "Novo mesto": (15.1689, 45.8010),
              "Triglav": (13.8368, 46.3786), "NW corner": (geo["bounds"][0], geo["bounds"][3]),
              "SE corner": (geo["bounds"][2], geo["bounds"][1]), "outside west": (13.0, 46.0),
              "outside north": (14.5, 47.2), "Gornja Radgona": (15.9893, 46.6772)}
    pts = []
    size = (geo["ny"], geo["nx"])
    for name, (lo, la) in places.items():
        r, c = export.lonlat_to_pixel(lo, la, geo["bounds"], size)
        r2, c2 = eng.lonlat_to_pixel(lo, la, geo)
        assert abs(float(r) - float(r2)) < 1e-2 and abs(float(c) - float(c2)) < 1e-2
        inside = 0 <= r2 < geo["ny"] and 0 <= c2 < geo["nx"]
        pts.append({"name": name, "lon": lo, "lat": la, "row": round(float(r2), 6), "col": round(float(c2), 6),
                    "pixel": [int(np.floor(r2)), int(np.floor(c2))] if inside else None,
                    "export_row": round(float(r), 6), "export_col": round(float(c), 6)})
    rc = [(0, 0), (0, geo["nx"] - 1), (geo["ny"] - 1, 0), (geo["ny"] - 1, geo["nx"] - 1), (300, 500), (123, 456), (500, 900)]
    lon_c, lat_c = eng.pixel_centres(geo)
    centres = [{"row": r, "col": c, "lon": round(float(lon_c[c]), 9), "lat": round(float(lat_c[r]), 9)} for r, c in rc]
    dump("geometry.json", {
        "description": "lon/lat -> fractional (row, col) of the phone raster (pixel centres at +0.5; `pixel` = floor) and "
                       "pixel (row, col) -> centre lon/lat. `export_row/col` is the PNG helper export.lonlat_to_pixel with the "
                       "rounded model.json bounds (agrees within `tolerance`).",
        "raster": {k: geo[k] for k in ("nx", "ny", "px", "x0", "y_top", "r_earth", "bounds")},
        "tolerance": {"pixel_rowcol": tol["pixel_rowcol"], "lonlat_deg": 1e-7},
        "lonlat_to_pixel": pts, "pixel_centres": centres})

    # ---- (d) scoring, colours, hotspots ---------------------------------------------------
    cases = [(0.8, 1.0, 0.9), (0.5, 0.9, 0.7), (1.0, 1.0, 1.0), (0.0, 1.0, 0.8), (0.3, 0.0, 0.8), (0.6, 0.35, 0.2), (0.95, 0.85, 0.99)]
    scoring = [{"habitat": h, "season": s, "trigger": t,
                "raw": float(predict.raw_score(h, s, t)), "score": float(predict.scale_score(predict.raw_score(h, s, t), reference))}
               for h, s, t in cases]
    levels = [0.0, 0.01, 0.039, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999, 1.0]
    lut = export.colour_lut(COLOR_LEVELS)
    colours = [{"score": s, "rgba": [int(x) for x in export.colorize(np.array([[s]]), COLOR_LEVELS)[0, 0]]} for s in levels]
    colours.append({"score": None, "rgba": [int(x) for x in export.colorize(np.array([[np.nan]]), COLOR_LEVELS)[0, 0]]})
    hs = _hotspot_vector()
    dump("misc.json", {
        "description": "(1) score scaling: raw = habitat*season*trigger, score = clamp(raw/reference); (2) colours: score -> RGBA "
                       "(null score = NaN/outside mask); (3) hotspot selection on a small grid, rule of model.json hotspots.",
        "reference": reference, "scoring": scoring, "colours": colours,
        "lut": [[int(v) for v in row] for row in lut], "hotspots": hs})
    return written


def _pick_cells(rasters: dict, geo: dict, model: dict):
    pe = PhoneEngine(rasters, geo, model)
    named = {"Ljubljana": (14.5058, 46.0569), "Pohorje": (15.50, 46.47), "Pokljuka": (13.95, 46.34),
             "Kocevski Rog": (14.80, 45.67), "Snežnik": (14.43, 45.58), "Goričko": (16.05, 46.85),
             "Triglav": (13.84, 46.38), "Koper": (13.73, 45.55), "Bovec": (13.55, 46.34),
             "Haloze": (15.95, 46.35), "Kamnik Alps": (14.60, 46.35), "Idrija hills": (14.03, 46.00),
             "Gorjanci": (15.30, 45.80), "Krim": (14.47, 45.93),
             "Trojane (pulse edge)": (14.88, 46.20), "Kozjansko (pulse edge)": (14.82, 46.02),
             "Posavje (pulse edge)": (14.93, 45.95)}
    cells = []
    for name, (lo, la) in named.items():
        r, c = eng.lonlat_to_pixel(lo, la, geo)
        r, c = int(np.floor(r)), int(np.floor(c))
        if not pe.mask[r, c]:      # nudge to the nearest land pixel
            rr, cc = np.nonzero(pe.mask[max(0, r - 15):r + 16, max(0, c - 15):c + 16])
            if rr.size == 0:
                continue
            k = int(np.argmin((rr - min(15, r)) ** 2 + (cc - min(15, c)) ** 2))
            r, c = max(0, r - 15) + int(rr[k]), max(0, c - 15) + int(cc[k])
        cells.append({"name": name, "row": r, "col": c, "lon": round(float(pe.lon2[r, c]), 9),
                      "lat": round(float(pe.lat2[r, c]), 9), "elev": float(pe.elev[r, c]),
                      "habitat": float(pe.habitat[r, c]), "synthetic": False})
    for name, lo, la, z in (("synthetic: far NW of the lattice", 12.9, 47.3, 500.0),
                            ("synthetic: on lattice node (14.5, 46.0)", 14.5, 46.0, 350.0),
                            ("synthetic: on upper lattice corner (16.75, 47.0)", 16.75, 47.0, 800.0)):
        cells.append({"name": name, "row": None, "col": None, "lon": lo, "lat": la, "elev": z,
                      "habitat": 0.6, "synthetic": True})
    return cells, pe


def _hotspot_vector() -> dict:
    """Small score / habitat / trigger grids and the production picks (export.hotspots rule)."""
    rng = np.random.default_rng(5)
    ny, nx = 28, 36
    yy, xx = np.mgrid[0:ny, 0:nx]
    s = np.zeros((ny, nx))
    for cy, cx, w, a in ((8, 10, 5, 1.2), (20, 26, 6, 1.1), (6, 30, 3, 0.9), (22, 8, 4, 0.7)):
        s += a * np.exp(-(((yy - cy) / w) ** 2 + ((xx - cx) / w) ** 2))
    score = np.clip(s + rng.normal(0, 0.01, s.shape), 0, 1)
    score = np.round(score, 3).astype("float32")
    score[score > 0.98] = 1.0                    # clipped plateau: ties
    hab = np.round(np.clip(0.4 + 0.6 * rng.random((ny, nx)), 0, 1), 3).astype("float32")
    trig = np.round(np.clip(0.3 + 0.7 * rng.random((ny, nx)), 0, 1), 3).astype("float32")
    mask = np.ones((ny, nx), bool)
    mask[:, :3] = False
    mask[10:13, 20:24] = False
    rank = np.asarray(score, "float32") + 1e-3 * np.nan_to_num(hab * trig)
    cells = export.pick_hotspots(rank, mask, export.HOTSPOT_MIN_SCORE, export.HOTSPOT_MIN_SEP_M, export.HOTSPOT_MAX)
    return {"ny": ny, "nx": nx, "min_score": export.HOTSPOT_MIN_SCORE, "min_sep_px": MIN_SEP_PX,
            "max_n": export.HOTSPOT_MAX, "rank_formula": "rank = score + 0.001 * habitat * trigger (Float32)",
            "score": [[float(x) for x in r] for r in score], "habitat": [[float(x) for x in r] for r in hab],
            "trigger": [[float(x) for x in r] for r in trig], "mask": [[int(x) for x in r] for r in mask],
            "expected_picks": [[r, c] for r, c in cells]}


# ===========================================================================================
# Driver
# ===========================================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ASSETS_DIR))
    ap.add_argument("--vectors", default=str(VECTORS_DIR))
    ap.add_argument("--skip-stats", action="store_true", help="skip the (60 s) trigger backtest: no stats.json, null backtest in finds.json")
    ap.add_argument("--skip-validation", action="store_true", help="skip the phone-vs-production comparison")
    ap.add_argument("--skip-vectors", action="store_true")
    a = ap.parse_args(argv)
    out, vec = Path(a.out), Path(a.vectors)
    t0 = time.time()
    geo = phone_geometry()
    print(f"raster {geo['nx']} x {geo['ny']} px, bounds {geo['bounds']}, ground ~{geo['ground_px_m_at_46N']} m at 46.1N")
    rasters = build_rasters(geo)
    sizes = write_rasters(rasters, out)
    rasters = load_package_rasters(out, geo)      # what the phone sees
    print("land pixels", int(rasters["mask"].sum()), "sizes", sizes)
    lattice = coarse_lattice()
    model = build_model_json(geo, lattice, None)
    validation = None
    if not a.skip_validation:
        validation = compare_with_production(rasters, geo, model)
        model["validation"] = validation
        print(json.dumps(validation, indent=1))
    (out / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    sizes["model.json"] = (out / "model.json").stat().st_size
    rep = None if a.skip_stats else run_trigger_backtest()
    finds, summary = build_finds(rep, geo)
    (out / "finds.json").write_text(json.dumps(finds, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    sizes["finds.json"] = (out / "finds.json").stat().st_size
    if rep is not None:
        stats = build_stats(rep, summary)
        (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
        sizes["stats.json"] = (out / "stats.json").stat().st_size
    if not a.skip_vectors:
        vs = build_vectors(rasters, geo, model, vec)
        print("vectors", vs)
    total = sum(sizes.values())
    print("package files:")
    for k, v in sizes.items():
        print(f"  {k:16s} {v / 1e3:8.1f} kB")
    print(f"  total            {total / 1e6:8.2f} MB   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
