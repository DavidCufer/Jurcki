"""Offline tests for jurcki.sources.weather (no network)."""
import zlib

import numpy as np
import pandas as pd
import xarray as xr

from jurcki.sources import weather as w


def _om_location(lat, lon, forecast=False, days=3):
    """A fake Open-Meteo location payload with a known pattern."""
    d = pd.date_range("2024-07-01", periods=days, freq="D")
    h = pd.date_range("2024-07-01", periods=24 * days, freq="h")
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in h],
              "relative_humidity_2m": [60.0] * len(h),
              "dew_point_2m": [10.0] * len(h),
              "snow_depth": [0.0] * len(h)}
    if forecast:
        hourly.update(soil_moisture_0_to_1cm=[0.1] * len(h), soil_moisture_1_to_3cm=[0.2] * len(h),
                      soil_moisture_3_to_9cm=[0.3] * len(h),
                      soil_temperature_0cm=[20.0] * len(h), soil_temperature_6cm=[16.0] * len(h))
    else:
        hourly.update(soil_moisture_0_to_7cm=[0.25] * len(h), soil_temperature_0_to_7cm=[15.0] * len(h))
    return {"latitude": lat, "longitude": lon, "elevation": 300.0 + lat,
            "daily": {"time": [t.strftime("%Y-%m-%d") for t in d],
                      "temperature_2m_mean": [18.0, 19.0, 20.0][:days],
                      "temperature_2m_max": [25.0, 26.0, 27.0][:days],
                      "temperature_2m_min": [10.0, 11.0, 12.0][:days],
                      "precipitation_sum": [0.0, 5.5, 1.0][:days]},
            "hourly": hourly}


def test_openmeteo_archive_mocked(monkeypatch):
    calls = []

    def fake_get(url, params):
        calls.append((url, params))
        lats = [float(x) for x in params["latitude"].split(",")]
        lons = [float(x) for x in params["longitude"].split(",")]
        return [_om_location(a, o) for a, o in zip(lats, lons)]

    monkeypatch.setattr(w, "_get_json", fake_get)
    pts = [(46.0, 14.0), (46.0, 14.1), (46.1, 14.0), (46.1, 14.1)]
    ds = w.fetch_openmeteo_archive(pts, "2024-07-01", "2024-07-03")
    url, params = calls[0]
    assert url == w.OPENMETEO_ARCHIVE
    assert params["models"] == "era5_land" and params["timezone"] == w.TZ
    assert params["start_date"] == "2024-07-01" and params["latitude"].startswith("46.0,46.0")
    assert ds.tmean.dims == ("time", "lat", "lon") and ds.tmean.shape == (3, 2, 2)
    assert float(ds.precip.sel(time="2024-07-02").mean()) == 5.5
    assert float(ds.swvl1.mean()) == 0.25 and float(ds.stl1.mean()) == 15.0
    assert float(ds.rh.mean()) == 60.0 and float(ds.dewpoint.mean()) == 10.0
    assert np.isclose(float(ds.orography.sel(lat=46.1, lon=14.0)), 346.1)
    assert set(w.VARS) | {"orography"} == set(ds.data_vars)


def test_openmeteo_forecast_mocked(monkeypatch):
    def fake_get(url, params):
        assert url == w.OPENMETEO_FORECAST and params["past_days"] == 7
        return _om_location(46.0, 14.0, forecast=True)  # single point -> dict

    monkeypatch.setattr(w, "_get_json", fake_get)
    ds = w.fetch_openmeteo_forecast([(46.0, 14.0)])
    assert np.isclose(float(ds.swvl1.mean()), (0.1 + 0.4 + 1.2) / 7)
    assert float(ds.stl1.mean()) == 18.0
    assert ds.tmax.shape == (3, 1, 1)


def test_schema_matches_across_sources(monkeypatch):
    monkeypatch.setattr(w, "_get_json", lambda u, p: _om_location(46.0, 14.0))
    om = w.fetch_openmeteo_archive([(46.0, 14.0)], "2024-07-01", "2024-07-03")
    t = pd.date_range("2024-06-30 22:00", periods=72, freq="h")  # UTC, CEST = +2
    h = xr.Dataset({k: (("time", "lat", "lon"), np.full((72, 1, 1), 1.0, np.float32))
                    for k in ("t2m", "d2m", "tp", "swvl1", "stl1", "snow_depth")},
                   coords={"time": t, "lat": [46.0], "lon": [14.0]})
    era = w._finish(w.daily_from_hourly(h), xr.DataArray([[300.0]], coords={"lat": [46.0], "lon": [14.0]}, dims=("lat", "lon")))
    assert set(era.data_vars) == set(om.data_vars)
    assert era.tmean.dims == om.tmean.dims


def test_daily_local_time_and_precip_labelling():
    # 3 local days (CEST): 2024-07-01 .. 07-03  = UTC 06-30 22:00 .. 07-03 22:00
    t0, t1 = w._local_range_to_utc("2024-07-01", "2024-07-03")
    assert t0 == pd.Timestamp("2024-06-30 22:00") and t1 == pd.Timestamp("2024-07-03 22:00")
    times = pd.date_range(t0, t1 - pd.Timedelta(hours=1), freq="h")
    n = len(times)
    temp = np.arange(n, dtype=np.float32)
    zeros = np.zeros((n, 1, 1), np.float32)
    h = xr.Dataset({"t2m": (("time", "lat", "lon"), temp.reshape(n, 1, 1)),
                    "d2m": (("time", "lat", "lon"), temp.reshape(n, 1, 1) - 5),
                    "tp": (("time", "lat", "lon"), np.ones((n, 1, 1), np.float32)),
                    "swvl1": (("time", "lat", "lon"), zeros + 0.3),
                    "stl1": (("time", "lat", "lon"), zeros + 15),
                    "snow_depth": (("time", "lat", "lon"), zeros)},
                   coords={"time": times, "lat": [46.0], "lon": [14.0]})
    d = w.daily_from_hourly(h)
    assert list(d.time.values) == list(pd.date_range("2024-07-01", periods=3).values)
    assert d.tmin.values.ravel().tolist() == [0.0, 24.0, 48.0]
    assert d.tmax.values.ravel().tolist() == [23.0, 47.0, 71.0]
    assert d.precip.values.ravel().tolist() == [24.0, 24.0, 24.0]
    assert np.allclose(d.tmean.values.ravel(), [11.5, 35.5, 59.5])
    assert (d.rh.values > 0).all() and (d.rh.values <= 100).all()


def test_decode_chunk_shuffle_deflate():
    a = np.random.default_rng(0).normal(size=(3, 4, 5)).astype("<f4")
    shuffled = np.frombuffer(a.tobytes(), np.uint8).reshape(-1, 4).T.tobytes()
    out = w._decode_chunk(zlib.compress(shuffled), a.shape, np.dtype("<f4"), True)
    assert np.array_equal(out, a)
    out = w._decode_chunk(zlib.compress(a.tobytes()), a.shape, np.dtype("<f4"), False)
    assert np.array_equal(out, a)


def test_grid_window_covers_bbox():
    lat, lon = w._window_coords()
    minx, miny, maxx, maxy = w.BBOX_WGS84
    assert lat.max() >= maxy and lat.min() <= miny and lon.min() <= minx and lon.max() >= maxx
    assert len(lat) <= 9 and len(lon) <= 16


def _source_ds():
    lat = np.array([45.5, 45.75, 46.0, 46.25])
    lon = np.array([14.0, 14.25, 14.5, 14.75])
    time = pd.date_range("2024-07-01", periods=2)
    orog = np.full((4, 4), 500.0)
    t = np.full((2, 4, 4), 20.0)
    p = 2.0 + lon[None, None, :] * 0 + lat[None, :, None] * 0 + np.zeros((2, 4, 4))
    p[:, :, 2:] = 6.0  # step in precip along lon
    ds = xr.Dataset({"tmean": (("time", "lat", "lon"), t), "precip": (("time", "lat", "lon"), p),
                     "orography": (("lat", "lon"), orog)},
                    coords={"time": time, "lat": lat, "lon": lon})
    return ds


def test_downscale_lapse_rate_latlon_grid():
    ds = _source_ds()
    lat = np.array([45.9, 46.0, 46.1])
    lon = np.array([14.3, 14.4])
    elev = xr.DataArray(np.array([[500.0, 1500.0], [500.0, 500.0], [0.0, 500.0]]),
                        dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = w.downscale_to_grid(ds, elev)
    assert out.tmean.shape == (2, 3, 2)
    # 1000 m higher than the source orography -> 6.5 K colder; 500 m lower -> 3.25 K warmer
    assert np.isclose(out.tmean.values[0, 0, 1], 20 - 6.5)
    assert np.isclose(out.tmean.values[0, 1, 0], 20.0)
    assert np.isclose(out.tmean.values[0, 2, 0], 20 + 3.25)
    # precip is bilinear only (between 2 and 6 where lon crosses 14.5)
    assert 2.0 <= out.precip.values.min() and out.precip.values.max() <= 6.0
    assert np.allclose(out.elevation.values, elev.values)


def test_downscale_projected_grid_and_nan_fill():
    from pyproj import Transformer
    ds = _source_ds()
    ds["tmean"] = ds.tmean.where(~((ds.lat == 45.5) & (ds.lon == 14.0)))  # a "sea" cell
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:3794", always_xy=True).transform(14.4, 45.7)
    xs = x0 + 250 * np.arange(3)
    ys = y0 + 250 * np.arange(3)[::-1]
    elev = xr.DataArray(np.full((3, 3), 500.0), dims=("y", "x"), coords={"y": ys, "x": xs})
    out = w.downscale_to_grid(ds, elev)
    assert out.tmean.dims == ("time", "y", "x")
    assert np.isfinite(out.tmean.values).all()
    assert np.allclose(out.tmean.values, 20.0, atol=1e-4)
