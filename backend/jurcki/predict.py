"""Daily prediction: weather -> fruiting trigger on the 250 m grid -> score maps for the app.

    score = clip(habitat x season_factor(month) x trigger / reference, 0, 1)

* ``habitat``  static map ``data/interim/model/habitat.tif`` (``model.habitat``).
* ``season_factor``  ``prior.season_factor(month)``, the same monthly step table that
  ``model.trigger`` used when it tuned the trigger parameters (its "comb" score is
  ``season x trigger``), so the tuned parameters are applied exactly as evaluated.
* ``trigger``  ``prior.trigger_score`` with the tuned ``PriorParams`` from
  ``data/processed/trigger_params.json``.
* ``reference``  a fixed constant (``data/processed/score_reference.json``): the 99.9th
  percentile of the raw product over Slovenian forest cells on every 2nd day of
  Aug-Oct 2010-2025.  The raw product tops out at ~0.88 (the trigger is rarely above ~0.8
  and the habitat is mostly < 1), which would leave the top of the app's colour map
  unused, so it is divided by this constant.  It is NOT a per-day normalisation: a dry or
  cold day stays dark, and the same weather always gives the same score.  The 99.9th (not
  the 99th) percentile is used so that very few cells reach the clip at 1: with the 99th,
  >10 % of forest cells tied at 1.0 on the best days, which flattens the colour map and
  makes the hotspot ranking (by score) arbitrary.  Rebuild it only when the model changes:
  ``python -m jurcki.predict --build-reference``.

Weather -> fine grid
--------------------
Features are computed on the coarse weather grid (ERA5 0.25 deg, 8 x 15 points) and only
then brought to the 250 m grid (728 k cells), because the rolling windows are the expensive
part and are cheap on 120 points.  Bringing the *features* rather than the raw days to the
fine grid is equivalent for the linear ones (window means / sums, and a constant lapse shift
commutes with them, also with the max / min of a window) and a close approximation for the
non-linear ones (``pulse`` is a 0/1 flag at the coarse points, so it becomes a smooth
fraction between them; the trigger is evaluated afterwards on the fine grid, not
interpolated).  Temperature features (t20, t7, tmax10, tmin7, stl1_20) are shifted by the
6.5 K/km lapse rate between the cell elevation (DEM) and the coarse orography (bilinearly
interpolated) -- the same correction ``features.point_weather`` applies to the daily series.

Sources
-------
``era5``       local history ``data/interim/weather`` (backfills, backtests).
``openmeteo``  forecast API: ``past_days`` + ``forecast_days`` for the 0.25 deg ERA5 points.
               Open-Meteo downscales temperature to a target elevation (default: its 90 m DEM
               at the point).  We pass ``elevation`` = the mean DEM height of the 0.25 deg box
               (like ERA5's grid-mean orography, and smooth between points) and use the
               ``elevation`` the API reports back as the orography for the lapse correction,
               so the correction is consistent whatever the API did.

Static inputs come from ``data/interim`` when it exists, else from the small runtime pack
``data/processed/runtime`` (``python -m jurcki.predict --pack-runtime``), which is what the
GitHub Actions runner uses.

CLI::

    python -m jurcki.predict --source era5 --date 2026-10-10 --days 1 --out data/output/backfill
    python -m jurcki.predict --source openmeteo --out data/output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

from . import config, export, features, grid
from .model import prior, trigger
from .sources import weather

log = logging.getLogger(__name__)

TZ = "Europe/Ljubljana"
LOOKBACK = 60                      # days of weather before the first predicted day
OM_PAST_DAYS = 62                  # Open-Meteo: 2 spare days over LOOKBACK
OM_FORECAST_DAYS = 7               # including today, i.e. today .. today + 6
OUT_PAST, OUT_AHEAD = 2, 6         # openmeteo output: today - 2 .. today + 6
TEMP_FEATURES = ("t20", "t7", "tmax10", "tmin7", "stl1_20")   # lapse-corrected
TRIGGER_FEATURES = ("t20", "p26", "pulse", "tmax10", "tmin7")  # what trigger_score reads
CHUNK_DAYS = 8                     # days downscaled at once (memory)
REPORT_DAYS = 30

RUNTIME_DIR = config.PROCESSED / "runtime"
REFERENCE_PATH = config.PROCESSED / "score_reference.json"
GBIF_PATH = config.PROCESSED / "gbif_boletus.parquet"
REFERENCE_MONTHS = (8, 9, 10)
REFERENCE_YEARS = (2010, 2025)
REFERENCE_QUANTILE = 99.9

# name -> (path under data/interim, file name in the runtime pack)
STATIC_FILES = {
    "habitat": (config.INTERIM / "model" / "habitat.tif", "habitat.tif"),
    "elev": (config.INTERIM / "static" / "elev.tif", "elev.tif"),
    "mask": (grid.MASK_PATH, "mask.tif"),
}


# ===========================================================================================
# Static inputs (data/interim, or the runtime pack when it is absent)
# ===========================================================================================
def static_path(name: str, runtime: bool | None = None) -> Path:
    """Path of a static layer. ``runtime``: None = interim if present else the pack."""
    interim, packed = STATIC_FILES[name]
    packed = RUNTIME_DIR / packed
    if runtime is None:
        runtime = os.environ.get("JURCKI_RUNTIME") == "1" or not interim.exists()
    p = packed if runtime else interim
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found; build the layers (data/interim) or the runtime pack "
            f"(python -m jurcki.predict --pack-runtime)")
    return p


@lru_cache(maxsize=8)
def _static(name: str, path: Path) -> np.ndarray:
    a = grid.read_tif(path)
    return a.astype(bool) if name == "mask" else a.astype("float32")


def load_habitat(runtime: bool | None = None) -> np.ndarray:
    return _static("habitat", static_path("habitat", runtime))


def load_elev(runtime: bool | None = None) -> np.ndarray:
    return _static("elev", static_path("elev", runtime))


def load_mask(runtime: bool | None = None) -> np.ndarray:
    return _static("mask", static_path("mask", runtime))


def pack_runtime(out_dir=None) -> dict:
    """Copy the layers needed at prediction time into compressed GeoTIFFs (lossless)."""
    out_dir = Path(out_dir or RUNTIME_DIR)
    sizes = {}
    for name, (src, fname) in STATIC_FILES.items():
        if not src.exists():
            raise FileNotFoundError(f"{src} missing, cannot pack {name}")
        a = grid.read_tif(src)
        if name == "mask":
            a = a.astype("uint8")
            kw = dict(nodata=None, predictor=2)
        else:
            a = a.astype("float32")
            kw = dict(predictor=3)     # floating-point predictor: much smaller deflate output
        p = grid.write_tif(out_dir / fname, a, **kw)
        sizes[fname] = p.stat().st_size
    sizes["total"] = sum(sizes.values())
    return sizes


# ===========================================================================================
# Fine grid
# ===========================================================================================
@dataclass
class FineGrid:
    """Cell-centre lon / lat and elevation (m) of the output grid (all 2-D, same shape)."""
    lon: np.ndarray
    lat: np.ndarray
    elev: np.ndarray


@lru_cache(maxsize=1)
def _cell_lonlat() -> tuple[np.ndarray, np.ndarray]:
    lon, lat = grid.cell_lonlat()
    return lon.astype("float64"), lat.astype("float64")


def model_grid(runtime: bool | None = None) -> FineGrid:
    """The 250 m model grid with the DEM elevation."""
    lon, lat = _cell_lonlat()
    return FineGrid(lon, lat, load_elev(runtime))


def _bilinear_weights(lat_axis, lon_axis, fg: FineGrid):
    """Corner indices and fractions of the fine cells in a regular (ascending) coarse grid.

    Cells outside the coarse grid take its nearest edge value (no extrapolation).
    """
    lat_axis, lon_axis = np.asarray(lat_axis, float), np.asarray(lon_axis, float)
    fi = np.clip((fg.lat - lat_axis[0]) / (lat_axis[1] - lat_axis[0]), 0, len(lat_axis) - 1)
    fj = np.clip((fg.lon - lon_axis[0]) / (lon_axis[1] - lon_axis[0]), 0, len(lon_axis) - 1)
    i0 = np.minimum(np.floor(fi).astype(np.intp), len(lat_axis) - 2)
    j0 = np.minimum(np.floor(fj).astype(np.intp), len(lon_axis) - 2)
    return i0, j0, (fi - i0).astype("float32"), (fj - j0).astype("float32")


def bilinear(field: np.ndarray, w) -> np.ndarray:
    """Bilinear interpolation of ``field`` (..., n_lat, n_lon) to the fine grid (..., rows, cols)."""
    i0, j0, di, dj = w
    f = np.asarray(field, dtype="float32")
    a = f[..., i0, j0] * (1 - di) * (1 - dj)
    a += f[..., i0, j0 + 1] * ((1 - di) * dj)
    a += f[..., i0 + 1, j0] * (di * (1 - dj))
    a += f[..., i0 + 1, j0 + 1] * (di * dj)
    return a


# ===========================================================================================
# Weather sources
# ===========================================================================================
class WeatherSource(Protocol):
    name: str

    def load(self, first_day: date, last_day: date) -> xr.Dataset:
        """Daily Dataset (time, lat ascending, lon ascending) in the ``sources.weather``
        schema incl. ``orography``; covers ``last_day`` and as much as possible back to
        ``first_day``."""


@lru_cache(maxsize=1)
def _era5_history() -> xr.Dataset:
    return weather.open_history().load()


class Era5Source:
    """Local ERA5 daily history (``data/interim/weather``)."""
    name = "era5"

    def load(self, first_day, last_day) -> xr.Dataset:
        ds = _era5_history()
        end = pd.Timestamp(ds.time.values[-1])
        if pd.Timestamp(last_day) > end:
            raise ValueError(f"ERA5 history ends {end:%Y-%m-%d}, cannot predict {last_day}")
        return ds.sel(time=slice(pd.Timestamp(first_day), pd.Timestamp(last_day)))


def era5_points() -> list[tuple[float, float]]:
    """(lat, lon) of the ERA5 0.25 deg points over ``config.BBOX_WGS84`` (ascending)."""
    lat, lon = weather._window_coords()
    return [(round(float(a), 2), round(float(o), 2)) for a in sorted(lat) for o in sorted(lon)]


def coarse_mean_elevation(points, fg: FineGrid | None = None, res: float = 0.25) -> list[float]:
    """Mean DEM height (m) of the ``res`` deg box around each (lat, lon) point.

    Points whose box holds no model-grid cell (the corners outside the grid) get the value
    of the nearest point that has one.
    """
    from scipy.ndimage import distance_transform_edt
    fg = fg or model_grid()
    lats = sorted({p[0] for p in points})
    lons = sorted({p[1] for p in points})
    lat0, lon0 = lats[0], lons[0]
    fi = np.round((fg.lat - lat0) / res).astype(int)
    fj = np.round((fg.lon - lon0) / res).astype(int)
    ok = ((np.abs(fg.lat - (lat0 + fi * res)) <= res / 2) & (np.abs(fg.lon - (lon0 + fj * res)) <= res / 2)
          & (fi >= 0) & (fi < len(lats)) & (fj >= 0) & (fj < len(lons)) & np.isfinite(fg.elev))
    idx = (fi * len(lons) + fj)[ok]
    n = len(lats) * len(lons)
    cnt = np.bincount(idx, minlength=n)
    tot = np.bincount(idx, weights=fg.elev[ok], minlength=n)
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).reshape(len(lats), len(lons))
    bad = np.isnan(mean)
    if bad.all():
        raise ValueError("no model-grid cell inside the weather points' boxes")
    if bad.any():
        near = distance_transform_edt(bad, return_distances=False, return_indices=True)
        mean = mean[near[0], near[1]]
    return [float(mean[lats.index(la), lons.index(lo)]) for la, lo in points]


class OpenMeteoSource:
    """Open-Meteo forecast API: ``past_days`` + ``forecast_days`` at the ERA5 0.25 deg points.

    The data is fetched once per instance (one batch of requests, ~3 calls for 120 points).
    ``fetch`` is injectable for tests.
    """
    name = "openmeteo"

    def __init__(self, past_days: int = OM_PAST_DAYS, forecast_days: int = OM_FORECAST_DAYS,
                 points=None, elevations=None, fetch=None, batch: int = 50, pause: float = 0.0):
        self.past_days, self.forecast_days = past_days, forecast_days
        self.points = list(points) if points is not None else era5_points()
        self.elevations = elevations
        self.fetch = fetch or weather.fetch_openmeteo_forecast
        self.batch, self.pause = batch, pause
        self._ds: xr.Dataset | None = None

    def dataset(self) -> xr.Dataset:
        if self._ds is None:
            elev = self.elevations if self.elevations is not None else coarse_mean_elevation(self.points)
            t0 = time.time()
            self._ds = self.fetch(self.points, past_days=self.past_days,
                                  forecast_days=self.forecast_days, batch=self.batch,
                                  elevations=elev, lean=True, pause=self.pause)
            log.info("Open-Meteo: %d points, %d days in %.1f s", len(self.points),
                     self._ds.sizes["time"], time.time() - t0)
        return self._ds

    def load(self, first_day, last_day) -> xr.Dataset:
        ds = self.dataset()
        end = pd.Timestamp(ds.time.values[-1])
        if pd.Timestamp(last_day) > end:
            raise ValueError(f"Open-Meteo data ends {end:%Y-%m-%d}, cannot predict {last_day}")
        return ds.sel(time=slice(pd.Timestamp(first_day), pd.Timestamp(last_day)))


def make_source(source) -> WeatherSource:
    if isinstance(source, str):
        return {"era5": Era5Source, "openmeteo": OpenMeteoSource}[source]()
    return source


@dataclass
class Coarse:
    """Gap-free daily weather on the coarse grid: arrays (time, lat, lon)."""
    times: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    orog: np.ndarray
    w: dict


def prepare_weather(ds: xr.Dataset) -> Coarse:
    """Dataset -> contiguous daily arrays without NaN (time-interpolated, spatially filled)."""
    from scipy.ndimage import distance_transform_edt
    ds = ds.sortby("lat").sortby("lon")
    times = pd.date_range(pd.Timestamp(ds.time.values[0]), pd.Timestamp(ds.time.values[-1]), freq="D")
    ds = ds.reindex(time=times)
    w = {}
    for v in features.WEATHER_VARS:
        a = ds[v].values.astype("float64")
        flat = pd.DataFrame(a.reshape(a.shape[0], -1))
        flat = flat.interpolate(method="linear", limit_area="inside").ffill().bfill()
        a = flat.to_numpy().reshape(a.shape)
        if np.isnan(a).any():
            if v == "snow_depth":
                a = np.nan_to_num(a, nan=0.0)
            else:   # a whole point missing: copy from the nearest valid point
                bad = np.isnan(a[0])
                if bad.all():
                    raise ValueError(f"weather variable {v} has no data")
                near = distance_transform_edt(bad, return_distances=False, return_indices=True)
                a = a[:, near[0], near[1]]
        w[v] = a
    orog = ds["orography"].values.astype("float64")
    if np.isnan(orog).any():
        bad = np.isnan(orog)
        near = distance_transform_edt(bad, return_distances=False, return_indices=True)
        orog = orog[near[0], near[1]]
    return Coarse(times, ds.lat.values, ds.lon.values, orog, w)


# ===========================================================================================
# Trigger on the fine grid
# ===========================================================================================
def _as_day(d) -> date:
    return pd.Timestamp(d).date()


def iter_fine_features(days, params=None, weather_source=None, names=TRIGGER_FEATURES,
                       fg: FineGrid | None = None) -> Iterator[tuple[date, dict]]:
    """Yield ``(day, {feature: 2-D float32 array on the fine grid})`` for each day.

    Features are computed on the coarse weather grid (needs ``LOOKBACK`` days of history,
    see the module docstring), bilinearly interpolated, and the temperature ones shifted
    by the lapse rate to the cell elevation.  ``names`` are keys of
    ``features.DYNAMIC_NAMES`` (``trigger`` is not available here, see ``trigger_grid``).
    """
    params = params or trigger.load_trigger_params()
    src = make_source(weather_source or "era5")
    days = [_as_day(d) for d in days]
    if not days:
        return
    names = list(names)
    ds = src.load(min(days) - timedelta(days=LOOKBACK), max(days))
    co = prepare_weather(ds)
    warm = (min(days) - co.times[0].date()).days
    if warm < 30:
        log.warning("only %d days of weather before %s (wanted %d): rolling windows are truncated",
                    warm, min(days), LOOKBACK)
    feats = features.dynamic_feature_arrays(co.w, params)
    unknown = [n for n in names if n not in feats or n == "trigger"]
    if unknown:
        raise ValueError(f"unknown features {unknown}")
    fg = fg or model_grid()
    w = _bilinear_weights(co.lat, co.lon, fg)
    dz = None
    if any(n in TEMP_FEATURES for n in names):
        dz = (fg.elev - bilinear(co.orog, w)).astype("float32")
        dz *= np.float32(weather.LAPSE_RATE)
    for k in range(0, len(days), CHUNK_DAYS):
        chunk = days[k:k + CHUNK_DAYS]
        idx = []
        for d in chunk:
            i = (pd.Timestamp(d) - co.times[0]).days
            if not 0 <= i < len(co.times):
                raise ValueError(f"{d} is outside the weather data {co.times[0]:%F}..{co.times[-1]:%F}")
            idx.append(i)
        fine = {}
        for n in names:
            a = bilinear(feats[n][idx], w)
            if n in TEMP_FEATURES:
                a += dz
            fine[n] = a
        for j, d in enumerate(chunk):
            yield d, {n: fine[n][j] for n in names}


def iter_trigger(days, params=None, weather_source=None, fg: FineGrid | None = None
                 ) -> Iterator[tuple[date, np.ndarray]]:
    """Yield ``(day, trigger)``, trigger in 0..1 on the fine grid (float32)."""
    params = params or trigger.load_trigger_params()
    for d, f in iter_fine_features(days, params, weather_source, TRIGGER_FEATURES, fg):
        yield d, np.clip(prior.trigger_score(f, params), 0, 1).astype("float32")


def trigger_grid(days, params=None, weather_source=None, fg: FineGrid | None = None) -> np.ndarray:
    """Fruiting trigger for each day on the 250 m grid: array (n_days, rows, cols), 0..1."""
    out = [t for _, t in iter_trigger(days, params, weather_source, fg)]
    if not out:
        shape = fg.lat.shape if fg else grid.grid_shape()
        return np.zeros((0, *shape), "float32")
    return np.stack(out)


# ===========================================================================================
# Score
# ===========================================================================================
def season_factor(day) -> float:
    """Monthly seasonal prior (same as in the trigger backtest)."""
    return float(prior.season_factor(_as_day(day).month))


def load_reference(path=REFERENCE_PATH) -> float:
    try:
        with open(path) as f:
            return float(json.load(f)["reference"])
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} missing; run python -m jurcki.predict --build-reference") from None


def raw_score(habitat, season, trig) -> np.ndarray:
    """habitat x season x trigger (unscaled product, float32)."""
    return np.asarray(habitat, "float32") * np.float32(season) * np.asarray(trig, "float32")


def scale_score(raw, reference: float) -> np.ndarray:
    """Fixed-reference rescaling: ``clip(raw / reference, 0, 1)``."""
    if not reference > 0:
        raise ValueError("reference must be positive")
    return np.clip(np.asarray(raw, "float32") / np.float32(reference), 0, 1).astype("float32")


@dataclass
class DayResult:
    day: date
    score: np.ndarray      # 0..1, rescaled with the fixed reference
    habitat: np.ndarray
    trigger: np.ndarray
    season: float


def score_days(days, weather_source=None, params=None, reference: float | None = None,
               fg: FineGrid | None = None, habitat: np.ndarray | None = None) -> Iterator[DayResult]:
    """Score each day (weather loaded once for all)."""
    params = params or trigger.load_trigger_params()
    reference = load_reference() if reference is None else reference
    habitat = np.nan_to_num(load_habitat() if habitat is None else habitat, nan=0.0)
    for d, trig in iter_trigger(days, params, weather_source, fg):
        s = season_factor(d)
        yield DayResult(d, scale_score(raw_score(habitat, s, trig), reference), habitat, trig, s)


def score_day(day, weather_source=None, params=None, reference: float | None = None,
              fg: FineGrid | None = None, habitat: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(score, habitat, trigger)`` for one day, all 2-D float32 in 0..1 on the 250 m grid."""
    r = next(score_days([day], weather_source, params, reference, fg, habitat))
    return r.score, r.habitat, r.trigger


# ===========================================================================================
# Score reference
# ===========================================================================================
def build_reference(path=REFERENCE_PATH, years=REFERENCE_YEARS, months=REFERENCE_MONTHS,
                    step: int = 2, quantile: float = REFERENCE_QUANTILE, n_cells: int = 60_000,
                    seed: int = 0) -> dict:
    """Fixed score scale: ``quantile`` of habitat x season x trigger over in-season history.

    Population: ``n_cells`` random Slovenian forest cells (mask & habitat > 0.05), every
    ``step``-th day of ``months`` in ``years`` (inclusive), ERA5 weather, tuned params.
    """
    params = trigger.load_trigger_params()
    habitat = np.nan_to_num(load_habitat(), nan=0.0)
    forest = load_mask() & (habitat > 0.05)
    rr, cc = np.nonzero(forest)
    pick = np.random.default_rng(seed).choice(len(rr), size=min(n_cells, len(rr)), replace=False)
    rr, cc = rr[pick], cc[pick]
    hab = habitat[rr, cc]
    days = [d.date() for d in pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31")
            if d.month in months][::step]
    chunks = []
    t0 = time.time()
    for d, trig in iter_trigger(days, params, "era5"):
        chunks.append(raw_score(hab, season_factor(d), trig[rr, cc]))
    vals = np.concatenate(chunks)
    qs = {f"p{q:g}": float(np.percentile(vals, q)) for q in (50, 75, 90, 95, 99, 99.5, 99.9)}
    ref = float(np.percentile(vals, quantile))
    out = {"reference": round(ref, 5), "quantile": quantile, "months": list(months),
           "years": list(years), "day_step": step, "n_days": len(days), "n_cells": int(len(rr)),
           "n_values": int(vals.size), "raw_quantiles": {k: round(v, 5) for k, v in qs.items()},
           "max_raw": round(float(vals.max()), 5), "trigger_params": trigger.params_to_dict(params),
           "season_factor": {str(m): prior.season_factor(m) for m in range(1, 13)},
           "generated": date.today().isoformat(),
           "note": "score = clip(habitat * season_factor(month) * trigger / reference, 0, 1)"}
    Path(path).write_text(json.dumps(out, indent=1))
    log.info("reference %.4f from %d values in %.0f s", ref, vals.size, time.time() - t0)
    return out


# ===========================================================================================
# Reports (recent GBIF finds)
# ===========================================================================================
def gbif_reports(ref_day, days: int = REPORT_DAYS, path=GBIF_PATH) -> pd.DataFrame:
    """Jurcki finds from the last ``days`` days up to ``ref_day`` as a ``write_reports`` frame."""
    cols = ["lon", "lat", "date", "source", "precision_m", "text", "url"]
    if not Path(path).exists():
        return pd.DataFrame(columns=cols)
    g = pd.read_parquet(path)
    if "is_jurcki" in g:
        g = g[g["is_jurcki"].astype(bool)]
    d = pd.to_datetime(g["date"])
    ref = pd.Timestamp(ref_day)
    g = g[(d > ref - pd.Timedelta(days=days)) & (d <= ref)]
    b = export.png_geometry()["bounds"]
    g = g[g.decimallongitude.between(b[0], b[2]) & g.decimallatitude.between(b[1], b[3])]
    return pd.DataFrame({
        "lon": g.decimallongitude.to_numpy(), "lat": g.decimallatitude.to_numpy(),
        "date": pd.to_datetime(g["date"]).dt.date.to_numpy(), "source": "gbif",
        "precision_m": g.coordinateuncertaintyinmeters.to_numpy(),
        "text": g.species.to_numpy(),
        "url": ["https://www.gbif.org/occurrence/%d" % int(i) for i in g.gbifid],
    }, columns=cols)


# ===========================================================================================
# Driver
# ===========================================================================================
def run(days, weather_source, out_dir, first_forecast_day, reports_day, quiet=False) -> dict:
    """Score ``days`` and write PNG + hotspots per day, plus meta.json and reports.geojson."""
    out_dir = Path(out_dir)
    src = make_source(weather_source)
    reference = load_reference()
    mask, elev = load_mask(), load_elev()
    t0 = time.time()
    done = []
    for r in score_days(days, src, reference=reference):
        t1 = time.time()
        res = export.export_day(r.day, r.score, r.habitat, r.trigger, out_dir, mask=mask, elev=elev)
        done.append(r.day)
        if not quiet:
            m = mask & (r.habitat > 0.05)
            print(f"{r.day}: mean score {r.score[m].mean():.3f} trigger {r.trigger[m].mean():.3f} "
                  f"max {r.score[mask].max():.2f}, {res['n_hotspots']} hotspots, "
                  f"png {res['png'].stat().st_size / 1e3:.0f} kB (export {time.time() - t1:.1f} s)")
    export.write_meta(done, first_forecast_day, out_dir,
                      extra={"source": src.name, "score_reference": reference})
    rep = gbif_reports(reports_day)
    export.write_reports(rep, out_dir)
    if not quiet:
        print(f"{len(done)} days, {len(rep)} reports -> {out_dir} in {time.time() - t0:.1f} s")
    return {"days": done, "n_reports": len(rep)}


def today_ljubljana() -> date:
    return datetime.now(ZoneInfo(TZ)).date()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["era5", "openmeteo"], default="openmeteo")
    ap.add_argument("--date", help="era5: first day (YYYY-MM-DD); openmeteo: ignored (today)")
    ap.add_argument("--days", type=int, default=1, help="era5: number of days from --date")
    ap.add_argument("--out", default=str(config.OUTPUT))
    ap.add_argument("--runtime", action="store_true",
                    help="use the runtime pack (data/processed/runtime) even if data/interim exists")
    ap.add_argument("--pack-runtime", action="store_true",
                    help="write the runtime pack from data/interim and exit")
    ap.add_argument("--build-reference", action="store_true",
                    help="recompute data/processed/score_reference.json (ERA5 history) and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(message)s")
    if a.runtime:
        os.environ["JURCKI_RUNTIME"] = "1"
    if a.pack_runtime:
        for k, v in pack_runtime().items():
            print(f"{k:12s} {v / 1e6:6.2f} MB")
        return
    if a.build_reference:
        print(json.dumps(build_reference(), indent=1))
        return
    if a.source == "era5":
        if not a.date:
            ap.error("--date is required for --source era5")
        first = date.fromisoformat(a.date)
        days = [first + timedelta(days=i) for i in range(a.days)]
        run(days, "era5", a.out, days[-1] + timedelta(days=1), days[-1])
    else:
        today = today_ljubljana()
        days = [today + timedelta(days=i) for i in range(-OUT_PAST, OUT_AHEAD + 1)]
        run(days, OpenMeteoSource(), a.out, today + timedelta(days=1), today)


if __name__ == "__main__":
    main()
