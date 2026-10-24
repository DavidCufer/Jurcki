"""Daily weather history and forecast for the study area.

Two interchangeable sources produce the same daily ``xarray.Dataset``
(dims ``time, lat, lon``; time = local calendar day, Europe/Ljubljana):

* ``fetch_era5_history``     ERA5 0.25 deg reanalysis, free bulk download.
* ``fetch_openmeteo_archive`` / ``fetch_openmeteo_forecast``  Open-Meteo
  (ERA5-Land 0.1 deg for history, NWP for the forecast).

Schema (all float32)::

    tmean, tmin, tmax   deg C, 2 m air temperature (hourly mean/min/max)
    precip              mm, daily total
    swvl1               m3/m3, volumetric soil water, 0-7 cm, daily mean
    stl1                deg C, soil temperature, 0-7 cm, daily mean
    dewpoint            deg C, 2 m dewpoint, daily mean
    rh                  %, 2 m relative humidity, daily mean
    snow_depth          m, snow depth, daily mean
    orography           m, source-grid surface height (lat, lon only)

ERA5 data sources
-----------------
Everything on the reachable buckets is hourly, so daily values are
aggregated here in *local* time (a day is 00:00-24:00 Europe/Ljubljana;
23/25 h on DST changes).

* NSF NCAR ``nsf-ncar-era5`` (S3, netCDF4, chunked ``27 h x 139 x 277``):
  a Slovenia window of one month is ~30 MB, so state variables are read
  with direct byte-range requests of the needed HDF5 chunks.  Fast, but
  ends a few months before today.
* Google ARCO-ERA5 ``full_37-1h-0p25deg-chunk-1.zarr-v3`` (GCS, chunked
  ``1 h x globe``): ~100 chunk/s, i.e. ~23 min per variable for 16 years.
  Used for precipitation (the NCAR forecast files are chunked
  ``12 h x globe``, ~11 MB per chunk) and for months NCAR lacks.

ERA5 total precipitation is the accumulation over the hour *ending* at the
timestamp (metres); we convert to mm and label each hour by its start so
that it falls in the correct local day.  ERA5 ``sd`` is metres of water
equivalent; depth is estimated with a fixed density of 100 kg/m3 (x10).

Run ``python -m jurcki.sources.weather --start 2010-01-01 --end 2025-12-31``.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import time
import zlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import xarray as xr

from jurcki.config import BBOX_WGS84, CRS, CRS_WGS84, INTERIM

log = logging.getLogger(__name__)

TZ = "Europe/Ljubljana"
WEATHER_DIR = INTERIM / "weather"
VARS = ["tmean", "tmin", "tmax", "precip", "swvl1", "stl1", "dewpoint", "rh", "snow_depth"]
UNITS = {"tmean": "degC", "tmin": "degC", "tmax": "degC", "precip": "mm",
         "swvl1": "m3 m-3", "stl1": "degC", "dewpoint": "degC", "rh": "%",
         "snow_depth": "m", "orography": "m"}
LAPSE_RATE = -6.5e-3  # K per metre
G0 = 9.80665


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _retry(fn, tries=5, wait=3.0):
    """Call ``fn()``, retrying on any exception with linear back-off."""
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # network hiccups of all kinds
            if i == tries - 1:
                raise
            log.warning("retry %d after %s: %s", i + 1, type(exc).__name__, exc)
            time.sleep(wait * (i + 1))


def _local_range_to_utc(first_day, last_day):
    """UTC [t0, t1) hour range covering local days first_day..last_day."""
    t0 = pd.Timestamp(first_day).tz_localize(TZ).tz_convert("UTC").tz_localize(None)
    t1 = (pd.Timestamp(last_day) + pd.Timedelta(days=1)).tz_localize(TZ)
    return t0, t1.tz_convert("UTC").tz_localize(None)


def _magnus_rh(t, td):
    """Relative humidity (%) from temperature and dewpoint (deg C)."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td / (b + td) - a * t / (b + t))


def daily_from_hourly(hourly: xr.Dataset) -> xr.Dataset:
    """Aggregate a UTC hourly dataset to local-day values (see module schema).

    ``hourly`` has t2m/d2m (deg C), tp (mm per hour, labelled by hour start),
    swvl1, stl1, snow_depth and dims (time [naive UTC], lat, lon).
    """
    local = hourly.time.to_index().tz_localize("UTC").tz_convert(TZ)
    day = local.normalize().tz_localize(None)
    h = hourly.assign_coords(day=("time", day.values))
    h["rh"] = _magnus_rh(h.t2m, h.d2m).clip(0, 100)
    g = h.groupby("day")
    mean, mn, mx, tot = g.mean(skipna=False), g.min(skipna=False), g.max(skipna=False), g.sum(skipna=False)
    out = xr.Dataset({
        "tmean": mean.t2m, "tmin": mn.t2m, "tmax": mx.t2m, "precip": tot.tp,
        "swvl1": mean.swvl1, "stl1": mean.stl1, "dewpoint": mean.d2m,
        "rh": mean.rh, "snow_depth": mean.snow_depth,
    }).rename(day="time")
    return out.astype("float32")


def _finish(ds: xr.Dataset, orography: xr.DataArray) -> xr.Dataset:
    ds = ds.sortby("lat").sortby("lon")
    ds["orography"] = orography.sortby("lat").sortby("lon").astype("float32")
    for k, u in UNITS.items():
        ds[k].attrs["units"] = u
    ds.attrs["timezone"] = TZ
    return ds


# --------------------------------------------------------------------------
# ERA5: low-level readers
# --------------------------------------------------------------------------
def _grid_window():
    """Row/col slices (0.25 deg global grid, 90N/0E origin) covering the bbox."""
    minx, miny, maxx, maxy = BBOX_WGS84
    r0 = int(np.floor((90 - np.ceil(maxy * 4) / 4) * 4))
    r1 = int(np.floor((90 - np.floor(miny * 4) / 4) * 4)) + 1
    c0, c1 = int(np.floor(minx * 4)), int(np.ceil(maxx * 4)) + 1
    return slice(r0, r1), slice(c0, c1)


def _window_coords():
    rows, cols = _grid_window()
    lat = 90 - 0.25 * np.arange(rows.start, rows.stop)
    lon = 0.25 * np.arange(cols.start, cols.stop)
    return lat, lon


@lru_cache(maxsize=1)
def _s3():
    import fsspec
    # small blocks: h5py only needs a few KB of metadata per file
    return fsspec.filesystem("s3", anon=True, default_cache_type="readahead",
                             default_block_size=1 << 20)


def _decode_chunk(raw: bytes, shape, dtype, shuffled=True) -> np.ndarray:
    """Decode one HDF5 chunk stored with (shuffle +) deflate."""
    buf = zlib.decompress(raw)
    if shuffled:
        size = np.dtype(dtype).itemsize
        buf = np.frombuffer(buf, np.uint8).reshape(size, -1).T.tobytes()
    return np.frombuffer(buf, dtype).reshape(shape)


def _read_h5_window(path: str, var: str, sel: tuple[slice, ...]) -> np.ndarray:
    """Read ``var[sel]`` from an S3 netCDF4 file, fetching only needed chunks.

    The chunks' byte ranges are looked up with h5py and downloaded in
    parallel with ranged GETs, which is much faster than letting h5py
    read through a file object.
    """
    import h5py
    fs = _s3()
    with h5py.File(fs.open(path), "r") as h:
        d = h[var]
        chunks, shuffled = d.chunks, d.shuffle
        bounds = [s.indices(n)[:2] for s, n in zip(sel, d.shape)]
        grid = [range(a // c, (b - 1) // c + 1) for (a, b), c in zip(bounds, chunks)]
        keys = list(itertools.product(*grid))
        infos = [d.id.get_chunk_info_by_coord(tuple(k * c for k, c in zip(key, chunks)))
                 for key in keys]
        dtype = d.dtype
    starts = [i.byte_offset for i in infos]
    stops = [i.byte_offset + i.size for i in infos]
    raws = _retry(lambda: fs.cat_ranges([path] * len(keys), starts, stops))
    out = np.empty([b - a for a, b in bounds], np.float32)
    for key, raw in zip(keys, raws):
        arr = _decode_chunk(raw, chunks, dtype, shuffled)
        src, dst = [], []
        for k, c, (a, b) in zip(key, chunks, bounds):
            lo, hi = max(a, k * c), min(b, (k + 1) * c)
            src.append(slice(lo - k * c, hi - k * c))
            dst.append(slice(lo - a, hi - a))
        out[tuple(dst)] = arr[tuple(src)]
    out[np.abs(out) > 1e30] = np.nan
    return out


_NCAR = "nsf-ncar-era5"
_NCAR_VARS = {  # our name -> (parameter code, netCDF variable)
    "t2m": ("128_167_2t", "VAR_2T"),
    "d2m": ("128_168_2d", "VAR_2D"),
    "swvl1": ("128_039_swvl1", "SWVL1"),
    "stl1": ("128_139_stl1", "STL1"),
    "snow_we": ("128_141_sd", "SD"),
}


@lru_cache(maxsize=1)
def _ncar_months() -> frozenset[str]:
    """Months (YYYYMM) available in the NCAR analysis archive."""
    names = _retry(lambda: _s3().ls(f"{_NCAR}/e5.oper.an.sfc"))
    return frozenset(n.rsplit("/", 1)[-1] for n in names)


def _ncar_hourly(name: str, t0: pd.Timestamp, t1: pd.Timestamp) -> np.ndarray:
    """Hourly values for UTC [t0, t1) from monthly NCAR files: (hours, lat, lon)."""
    code, var = _NCAR_VARS[name]
    rows, cols = _grid_window()
    parts = []
    for month in pd.period_range(t0, t1 - pd.Timedelta(hours=1), freq="M"):
        m0 = month.start_time
        a = max(t0, m0)
        b = min(t1, (month + 1).start_time)
        i0, i1 = int((a - m0) / pd.Timedelta(hours=1)), int((b - m0) / pd.Timedelta(hours=1))
        last = month.end_time.normalize().strftime("%Y%m%d") + "23"
        path = (f"{_NCAR}/e5.oper.an.sfc/{month:%Y%m}/"
                f"e5.oper.an.sfc.{code}.ll025sc.{month:%Y%m}0100_{last}.nc")
        parts.append(_read_h5_window(path, var, (slice(i0, i1), rows, cols)))
    return np.concatenate(parts)


@lru_cache(maxsize=1)
def _arco() -> xr.Dataset:
    import zarr
    zarr.config.set({"async.concurrency": 64})
    return xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                        storage_options={"token": "anon"}, chunks=None, consolidated=False)


_ARCO_VARS = {"t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature",
              "swvl1": "volumetric_soil_water_layer_1", "stl1": "soil_temperature_level_1",
              "snow_we": "snow_depth", "tp": "total_precipitation"}
_ARCO_EPOCH = pd.Timestamp("1900-01-01")


def _arco_hourly(name: str, t0: pd.Timestamp, t1: pd.Timestamp) -> np.ndarray:
    """Hourly values for UTC [t0, t1) from ARCO-ERA5: (hours, lat, lon)."""
    rows, cols = _grid_window()
    i0, i1 = (int((t - _ARCO_EPOCH) / pd.Timedelta(hours=1)) for t in (t0, t1))
    da = _arco()[_ARCO_VARS[name]]
    return _retry(lambda: da.isel(time=slice(i0, i1), latitude=rows, longitude=cols)
                  .values.astype("float32"))


def _arco_last_day() -> pd.Timestamp:
    """Last local day for which ARCO has complete t2m and precipitation."""
    rows, cols = _grid_window()
    ds = _arco()
    day = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    for _ in range(30):
        _, end = _local_range_to_utc(day, day)
        i = int((end - _ARCO_EPOCH) / pd.Timedelta(hours=1))
        ok = all(not np.isnan(ds[v].isel(time=i, latitude=rows.start, longitude=cols.start).values)
                 for v in ("total_precipitation", "2m_temperature"))
        if ok:
            return day
        day -= pd.Timedelta(days=1)
    raise RuntimeError("no recent ARCO-ERA5 data found")


@lru_cache(maxsize=1)
def _orography() -> xr.DataArray:
    rows, cols = _grid_window()
    i = int((pd.Timestamp("2000-01-01") - _ARCO_EPOCH) / pd.Timedelta(hours=1))  # static field, any valid time
    z = _arco()["geopotential_at_surface"].isel(time=i, latitude=rows, longitude=cols).values
    lat, lon = _window_coords()
    return xr.DataArray(z / G0, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


# --------------------------------------------------------------------------
# ERA5: monthly build and driver
# --------------------------------------------------------------------------
def _hourly_month(first_day, last_day, use_ncar=True) -> xr.Dataset:
    """Hourly ERA5 window (UTC) covering local days first_day..last_day."""
    t0, t1 = _local_range_to_utc(first_day, last_day)
    times = pd.date_range(t0, t1 - pd.Timedelta(hours=1), freq="h")
    ncar_ok = use_ncar and {f"{t:%Y%m}" for t in (t0, t1 - pd.Timedelta(hours=1))} <= _ncar_months()
    read = _ncar_hourly if ncar_ok else _arco_hourly
    lat, lon = _window_coords()
    dims = ("time", "lat", "lon")
    raw = {k: read(k, t0, t1) for k in ("t2m", "d2m", "swvl1", "stl1", "snow_we")}
    # precipitation: hour-ending stamps t0+1h..t1 -> relabel by hour start
    tp = _arco_hourly("tp", t0 + pd.Timedelta(hours=1), t1 + pd.Timedelta(hours=1))
    h = xr.Dataset(coords={"time": times, "lat": lat, "lon": lon})
    h["t2m"] = (dims, raw["t2m"] - 273.15)
    h["d2m"] = (dims, raw["d2m"] - 273.15)
    h["swvl1"] = (dims, np.clip(raw["swvl1"], 0, None))
    h["stl1"] = (dims, raw["stl1"] - 273.15)
    h["snow_depth"] = (dims, raw["snow_we"] * 10.0)
    h["tp"] = (dims, np.clip(tp, 0, None) * 1000.0)
    return h


def _month_file(month: pd.Period):
    return WEATHER_DIR / "monthly" / f"era5_daily_{month:%Y%m}.nc"


def _build_month(month: pd.Period, first_day, last_day, use_ncar=True) -> str:
    """Compute and save one month of daily ERA5 (skips if already complete)."""
    path = _month_file(month)
    if path.exists():
        with xr.open_dataset(path) as old:
            have = pd.Timestamp(old.time.values[0]), pd.Timestamp(old.time.values[-1])
            gaps = bool(old.tmean.isnull().all(["lat", "lon"]).any())  # source holes: retry
            if not gaps and have[0] <= pd.Timestamp(first_day) and have[1] >= pd.Timestamp(last_day):
                return f"{month} cached"
    t = time.time()
    daily = daily_from_hourly(_hourly_month(first_day, last_day, use_ncar))
    ds = _finish(daily, _orography())
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    return f"{month} done in {time.time() - t:.0f}s"


def _concat_files(files) -> xr.Dataset:
    """Concatenate daily netCDF files along time (loaded into memory)."""
    parts = [xr.load_dataset(f) for f in files]
    ds = xr.concat([p[VARS] for p in parts], "time")
    ds["orography"] = parts[0]["orography"]
    ds.attrs = parts[0].attrs
    for k, u in UNITS.items():
        ds[k].attrs["units"] = u
    return ds


def _write_years(years):
    """Concatenate monthly files into era5_daily_YYYY.nc (time, lat, lon)."""
    enc = {k: {"zlib": True, "complevel": 4} for k in VARS}
    for y in sorted(set(years)):
        files = sorted((WEATHER_DIR / "monthly").glob(f"era5_daily_{y}??.nc"))
        if files:
            _concat_files(files).to_netcdf(WEATHER_DIR / f"era5_daily_{y}.nc", encoding=enc)


def fetch_era5_history(start="2010-01-01", end=None, workers=8, use_ncar=True):
    """Download ERA5 for the study area and write ``era5_daily_YYYY.nc``.

    Resumable: finished months are cached in ``weather/monthly`` and reused.
    ``end=None`` means the latest complete day available.  Returns the list
    of yearly files.
    """
    last = _arco_last_day()
    end = last if end is None else min(pd.Timestamp(end), last)
    start = pd.Timestamp(start)
    jobs = []
    for month in pd.period_range(start, end, freq="M"):
        first = max(start, month.start_time)
        final = min(end, month.end_time.normalize())
        jobs.append((month, first, final))
    log.info("ERA5: %d months, %s .. %s", len(jobs), start.date(), end.date())
    # processes, not threads: h5py serialises metadata lookups behind a global
    # lock, which dominates the NCAR reads; "spawn" keeps fsspec loops clean
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_build_month, m, a, b, use_ncar) for m, a, b in jobs]
        for f in futs:
            log.info(f.result())
    _write_years({m.year for m, _, _ in jobs})
    return sorted(WEATHER_DIR.glob("era5_daily_[0-9][0-9][0-9][0-9].nc"))


def open_history(years=None) -> xr.Dataset:
    """Open the yearly ERA5 files as one dataset (time, lat, lon)."""
    files = sorted(WEATHER_DIR.glob("era5_daily_[0-9][0-9][0-9][0-9].nc"))
    if years is not None:
        files = [f for f in files if int(f.stem[-4:]) in set(years)]
    return _concat_files(files)


# --------------------------------------------------------------------------
# Open-Meteo client
# --------------------------------------------------------------------------
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
_OM_DAILY = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
# ERA5-Land (archive) has 0-7 cm layers; the NWP forecast has thin layers
_OM_HOURLY_ARCHIVE = ["soil_moisture_0_to_7cm", "soil_temperature_0_to_7cm",
                      "relative_humidity_2m", "dew_point_2m", "snow_depth"]
_OM_HOURLY_FORECAST = ["soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm",
                       "soil_temperature_0cm", "soil_temperature_6cm",
                       "relative_humidity_2m", "dew_point_2m", "snow_depth"]


def grid_points(res=0.1, bbox=BBOX_WGS84):
    """(lat, lon) points on a regular grid over the bbox (ERA5-Land: 0.1 deg)."""
    minx, miny, maxx, maxy = bbox
    lats = np.round(np.arange(np.floor(miny / res) * res, maxy + res / 2, res), 4)
    lons = np.round(np.arange(np.floor(minx / res) * res, maxx + res / 2, res), 4)
    return [(float(a), float(o)) for a in lats for o in lons]


def _get_json(url, params):
    """GET with retries; Open-Meteo reports errors as ``{"error": true, "reason": ...}``."""
    for i in range(5):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (i + 1))
            continue
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Open-Meteo: {data.get('reason')}")
        r.raise_for_status()
        return data
    raise RuntimeError(f"Open-Meteo request failed: HTTP {r.status_code}")


def _om_frame(loc: dict, forecast: bool) -> pd.DataFrame:
    """One Open-Meteo location response -> daily DataFrame in our schema."""
    daily = pd.DataFrame(loc["daily"]).set_index("time")
    daily.index = pd.to_datetime(daily.index)
    out = pd.DataFrame(index=daily.index)
    out["tmean"] = daily["temperature_2m_mean"]
    out["tmin"] = daily["temperature_2m_min"]
    out["tmax"] = daily["temperature_2m_max"]
    out["precip"] = daily["precipitation_sum"]
    hourly = pd.DataFrame(loc["hourly"]).set_index("time")
    hourly.index = pd.to_datetime(hourly.index)
    if forecast:  # weight the thin layers to a 0-7 cm layer
        hourly["swvl1"] = (hourly.soil_moisture_0_to_1cm + 2 * hourly.soil_moisture_1_to_3cm
                           + 4 * hourly.soil_moisture_3_to_9cm) / 7
        hourly["stl1"] = (hourly.soil_temperature_0cm + hourly.soil_temperature_6cm) / 2
    else:
        hourly["swvl1"] = hourly.soil_moisture_0_to_7cm
        hourly["stl1"] = hourly.soil_temperature_0_to_7cm
    hourly = hourly.rename(columns={"dew_point_2m": "dewpoint", "relative_humidity_2m": "rh"})
    means = hourly[["swvl1", "stl1", "dewpoint", "rh", "snow_depth"]].astype(float)
    means = means.groupby(means.index.normalize()).mean()  # local-day means
    return out.join(means).reindex(columns=VARS).astype(float)


def _om_dataset(frames: dict, elev: dict) -> xr.Dataset:
    """{(lat, lon): daily DataFrame} -> Dataset (time, lat, lon)."""
    lats = sorted({p[0] for p in frames})
    lons = sorted({p[1] for p in frames})
    times = sorted(set().union(*(f.index for f in frames.values())))
    arr = np.full((len(VARS), len(times), len(lats), len(lons)), np.nan, np.float32)
    orog = np.full((len(lats), len(lons)), np.nan, np.float32)
    for (la, lo), f in frames.items():
        i, j = lats.index(la), lons.index(lo)
        arr[:, :, i, j] = f.reindex(times)[VARS].to_numpy().T
        orog[i, j] = elev.get((la, lo), np.nan)
    ds = xr.Dataset({v: (("time", "lat", "lon"), arr[k]) for k, v in enumerate(VARS)},
                    coords={"time": pd.DatetimeIndex(times), "lat": lats, "lon": lons})
    return _finish(ds, xr.DataArray(orog, coords={"lat": lats, "lon": lons}, dims=("lat", "lon")))


def _om_fetch(url, points, params, forecast, batch=50) -> xr.Dataset:
    """Query Open-Meteo for many points (comma-separated, ``batch`` per call)."""
    frames: dict = {}
    elev: dict = {}
    base = {"timezone": TZ, "daily": ",".join(_OM_DAILY),
            "hourly": ",".join(_OM_HOURLY_FORECAST if forecast else _OM_HOURLY_ARCHIVE), **params}
    for i in range(0, len(points), batch):
        pts = points[i:i + batch]
        q = {**base, "latitude": ",".join(str(p[0]) for p in pts),
             "longitude": ",".join(str(p[1]) for p in pts)}
        data = _get_json(url, q)
        locs = data if isinstance(data, list) else [data]
        for p, loc in zip(pts, locs):  # responses keep the request order
            f = _om_frame(loc, forecast)
            frames[p] = pd.concat([frames[p], f]) if p in frames else f
            elev[p] = loc.get("elevation", np.nan)
    for p, f in frames.items():
        frames[p] = f[~f.index.duplicated(keep="last")].sort_index()
    return _om_dataset(frames, elev)


def fetch_openmeteo_archive(points, start, end, model="era5_land", batch=50) -> xr.Dataset:
    """Daily history from the Open-Meteo archive API (ERA5-Land by default).

    ``points``: list of (lat, lon) forming a regular grid (see ``grid_points``).
    Requests are split by year to keep responses small.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for y in range(start.year, end.year + 1):
        a, b = max(start, pd.Timestamp(y, 1, 1)), min(end, pd.Timestamp(y, 12, 31))
        params = {"start_date": f"{a:%Y-%m-%d}", "end_date": f"{b:%Y-%m-%d}", "models": model}
        parts.append(_om_fetch(OPENMETEO_ARCHIVE, list(points), params, False, batch))
    return xr.concat(parts, "time") if len(parts) > 1 else parts[0]


def fetch_openmeteo_forecast(points, past_days=7, forecast_days=16, batch=50) -> xr.Dataset:
    """Recent days plus forecast from the Open-Meteo forecast API (best match)."""
    params = {"past_days": past_days, "forecast_days": forecast_days}
    return _om_fetch(OPENMETEO_FORECAST, list(points), params, True, batch)


# --------------------------------------------------------------------------
# Downscaling to the model grid
# --------------------------------------------------------------------------
def _cell_lonlat(elev_grid: xr.DataArray):
    """2-D lat/lon of every cell of ``elev_grid`` (model grid or lat/lon grid)."""
    if {"lat", "lon"} <= set(elev_grid.coords):
        lat, lon = elev_grid["lat"], elev_grid["lon"]
        if lat.ndim == 1:
            lat, lon = xr.broadcast(lat, lon)
        return lat.transpose(*elev_grid.dims), lon.transpose(*elev_grid.dims)
    from pyproj import Transformer
    ydim, xdim = elev_grid.dims[-2:]
    x2, y2 = np.meshgrid(elev_grid[xdim].values, elev_grid[ydim].values)
    lon, lat = Transformer.from_crs(CRS, CRS_WGS84, always_xy=True).transform(x2, y2)
    return (xr.DataArray(lat, dims=(ydim, xdim), coords=elev_grid.coords),
            xr.DataArray(lon, dims=(ydim, xdim), coords=elev_grid.coords))


def _fill_nearest(da: xr.DataArray) -> xr.DataArray:
    """Fill NaN cells (sea, etc.) from the nearest valid cell in lat/lon."""
    from scipy.ndimage import distance_transform_edt
    first = da.isel(time=0) if "time" in da.dims else da
    bad = np.isnan(first.values)
    if not bad.any() or bad.all():
        return da
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    filled = da.values[..., idx[0], idx[1]]
    return da.copy(data=filled)


def downscale_to_grid(ds: xr.Dataset, elev_grid: xr.DataArray,
                      lapse_rate=LAPSE_RATE, lapse_vars=("tmean", "tmin", "tmax")) -> xr.Dataset:
    """Bilinearly interpolate ``ds`` to the cells of ``elev_grid`` (metres).

    ``elev_grid`` is a DEM on the model grid: either dims (y, x) with 1-D
    x/y coords in EPSG:3794 (``config.CRS``), or with ``lat``/``lon`` coords.
    Temperatures in ``lapse_vars`` are shifted by ``lapse_rate`` (K/m) times
    the difference between the cell elevation and the interpolated source
    orography (``ds["orography"]``).  Other variables are interpolated only.
    NaNs on the source grid (e.g. sea) are filled from the nearest cell first.
    Pass a limited time window: the output holds (time, y, x) arrays.
    """
    lat2d, lon2d = _cell_lonlat(elev_grid)
    ydim, xdim = elev_grid.dims[-2:]
    out = xr.Dataset(coords={"time": ds.time, ydim: elev_grid[ydim], xdim: elev_grid[xdim]})

    def interp(da):
        da = _fill_nearest(da)
        r = da.interp(lat=lat2d, lon=lon2d, method="linear", kwargs={"fill_value": None})
        return r.drop_vars(["lat", "lon"], errors="ignore")

    orog = interp(ds["orography"])
    for name in ds.data_vars:
        if name == "orography" or "time" not in ds[name].dims:
            continue
        v = interp(ds[name])
        if name in lapse_vars:
            v = v + lapse_rate * (elev_grid - orog)
        out[name] = v.astype("float32")
        out[name].attrs.update(ds[name].attrs)
    out["orography"] = orog.astype("float32")
    out["elevation"] = elev_grid.astype("float32")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="default: latest complete day")
    ap.add_argument("--workers", type=int, default=8, help="months processed in parallel (processes)")
    ap.add_argument("--arco-only", action="store_true", help="skip the NCAR fast path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    files = fetch_era5_history(args.start, args.end, args.workers, use_ncar=not args.arco_only)
    log.info("wrote %d files in %s", len(files), WEATHER_DIR)


if __name__ == "__main__":
    main()
