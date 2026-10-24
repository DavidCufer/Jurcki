"""Offline tests for jurcki.predict (synthetic weather and grids, no data/interim, no network)."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from jurcki import grid, predict as P
from jurcki.model import prior
from jurcki.sources import weather as w

LAT = np.array([46.0, 46.25, 46.5])
LON = np.array([14.0, 14.25, 14.5, 14.75])


def _fine(rows=12, cols=16, elev=500.0):
    """A small fine grid inside the coarse one (lon 14.1..14.6, lat 46.05..46.45)."""
    lon, lat = np.meshgrid(np.linspace(14.1, 14.6, cols), np.linspace(46.45, 46.05, rows))
    return P.FineGrid(lon, lat, np.full(lon.shape, elev, "float32"))


def _dataset(days, tmean=14.0, precip=5.0, orog=500.0, lat=LAT, lon=LON, fields=None):
    t = pd.date_range(days[0], days[1], freq="D")
    shape = (len(t), len(lat), len(lon))
    base = dict(tmean=tmean, tmin=tmean - 6, tmax=tmean + 6, precip=precip, swvl1=0.3,
                stl1=tmean, dewpoint=5.0, rh=70.0, snow_depth=0.0)
    ds = xr.Dataset({k: (("time", "lat", "lon"), np.full(shape, v, "float32"))
                     for k, v in base.items()},
                    coords={"time": t, "lat": lat, "lon": lon})
    for k, a in (fields or {}).items():
        ds[k] = (("time", "lat", "lon"), a.astype("float32"))
    ds["orography"] = (("lat", "lon"), np.full((len(lat), len(lon)), orog, "float32"))
    return ds


class FakeSource:
    name = "fake"

    def __init__(self, ds):
        self.ds = ds

    def load(self, first_day, last_day):
        return self.ds.sel(time=slice(pd.Timestamp(first_day), pd.Timestamp(last_day)))


# --- downscaling ----------------------------------------------------------------------------
def test_bilinear_reproduces_linear_field_and_clamps_edges():
    fg = _fine()
    wts = P._bilinear_weights(LAT, LON, fg)
    field = 2.0 * LAT[:, None] + 3.0 * LON[None, :]
    out = P.bilinear(field, wts)
    assert out.shape == fg.lon.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, 2.0 * fg.lat + 3.0 * fg.lon, atol=2e-4)
    far = P.FineGrid(np.array([[10.0]]), np.array([[50.0]]), np.array([[0.0]], "float32"))
    edge = P.bilinear(field, P._bilinear_weights(LAT, LON, far))
    assert np.isclose(edge[0, 0], field[-1, 0])        # NW of the window -> its (max lat, min lon) corner
    assert P.bilinear(np.stack([field, field + 1]), wts).shape == (2, *fg.lon.shape)


def test_temperature_features_get_lapse_correction_others_do_not():
    day = date(2025, 9, 20)
    ds = _dataset(("2025-07-01", "2025-09-20"), tmean=14.0, precip=5.0, orog=500.0)
    fg = _fine(elev=1500.0)                            # 1000 m above the coarse orography
    names = ("t20", "tmax10", "tmin7", "t7", "stl1_20", "p26", "p14", "swvl1_7")
    (d, f), = P.iter_fine_features([day], prior.PriorParams(), FakeSource(ds), names, fg)
    assert d == day
    shift = w.LAPSE_RATE * 1000.0                      # -6.5 K
    np.testing.assert_allclose(f["t20"], 14.0 + shift, atol=1e-3)
    np.testing.assert_allclose(f["t7"], 14.0 + shift, atol=1e-3)
    np.testing.assert_allclose(f["tmax10"], 20.0 + shift, atol=1e-3)
    np.testing.assert_allclose(f["tmin7"], 8.0 + shift, atol=1e-3)
    np.testing.assert_allclose(f["stl1_20"], 14.0 + shift, atol=1e-3)
    np.testing.assert_allclose(f["p26"], 26 * 5.0, atol=1e-3)     # rain is not shifted
    np.testing.assert_allclose(f["p14"], 14 * 5.0, atol=1e-3)
    np.testing.assert_allclose(f["swvl1_7"], 0.3, atol=1e-5)
    assert f["t20"].shape == fg.lon.shape


def test_features_interpolate_spatially():
    ds = _dataset(("2025-07-01", "2025-09-20"))
    ramp = np.broadcast_to(10 + 2 * (LON - LON[0]) / 0.25, (ds.sizes["time"], len(LAT), len(LON)))
    ds["tmean"] = (("time", "lat", "lon"), ramp.astype("float32").copy())
    fg = _fine(elev=500.0)                             # same height as orography: no lapse term
    (_, f), = P.iter_fine_features(["2025-09-20"], prior.PriorParams(), FakeSource(ds), ("t20",), fg)
    np.testing.assert_allclose(f["t20"], 10 + 2 * (fg.lon - LON[0]) / 0.25, atol=2e-3)


def test_trigger_grid_shape_bounds_and_elevation_effect():
    days = [date(2025, 9, 20), date(2025, 9, 21)]
    ds = _dataset(("2025-07-01", "2025-09-21"), tmean=17.0, precip=6.0)
    lo, hi = _fine(elev=500.0), _fine(elev=2000.0)
    a = P.trigger_grid(days, prior.PriorParams(), FakeSource(ds), lo)
    b = P.trigger_grid(days, prior.PriorParams(), FakeSource(ds), hi)
    assert a.shape == (2, *lo.lon.shape) and a.dtype == np.float32
    assert a.min() >= 0 and a.max() <= 1 and a.max() > 0.2
    assert (b < a).all()                               # colder at 2000 m: away from the 17.6 C optimum
    assert P.trigger_grid([], None, FakeSource(ds), lo).shape[0] == 0


def test_trigger_matches_trigger_score_of_the_same_weather():
    p = prior.PriorParams()
    ds = _dataset(("2025-07-01", "2025-09-20"), tmean=15.0, precip=3.0)
    fg = _fine(elev=500.0)
    got = P.trigger_grid(["2025-09-20"], p, FakeSource(ds), fg)[0]
    n = ds.sizes["time"]
    const = lambda v: np.full(n, v)
    f = prior.weather_features(const(15.0), const(9.0), const(21.0), const(3.0), p)
    want = prior.trigger_score({k: v[-1] for k, v in f.items()}, p)
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_missing_days_are_interpolated_and_future_day_rejected():
    ds = _dataset(("2025-07-01", "2025-09-20"))
    ds = ds.isel(time=[i for i in range(ds.sizes["time"]) if i not in (30, 31)])   # gap
    co = P.prepare_weather(ds)
    assert len(co.times) == 82 and np.isfinite(co.w["tmean"]).all()
    with pytest.raises(ValueError, match="outside"):
        P.trigger_grid(["2025-09-21"], None, FakeSource(ds), _fine())


# --- score ----------------------------------------------------------------------------------
def test_scale_score_is_fixed_and_clipped():
    raw = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(P.scale_score(raw, 0.5), [0, 0.5, 1, 1, 1])
    np.testing.assert_allclose(P.scale_score(raw, 0.5), P.scale_score(raw, 0.5))   # deterministic
    with pytest.raises(ValueError):
        P.scale_score(raw, 0.0)
    assert P.raw_score(np.float32(0.8), 0.5, np.float32(0.5)) == pytest.approx(0.2)


def test_score_day_bounds_determinism_and_no_per_day_normalisation():
    fg = _fine()
    hab = np.linspace(0, 1, fg.lon.size, dtype="float32").reshape(fg.lon.shape)
    good = FakeSource(_dataset(("2025-07-01", "2025-09-20"), tmean=15.0, precip=6.0))
    dry = FakeSource(_dataset(("2025-07-01", "2025-09-20"), tmean=15.0, precip=0.0))
    kw = dict(params=prior.PriorParams(), reference=0.6, fg=fg, habitat=hab)
    s1, h1, t1 = P.score_day("2025-09-20", good, **kw)
    s2, _, _ = P.score_day("2025-09-20", good, **kw)
    sd, _, td = P.score_day("2025-09-20", dry, **kw)
    np.testing.assert_array_equal(s1, s2)
    assert s1.shape == hab.shape and s1.min() >= 0 and s1.max() <= 1
    np.testing.assert_array_equal(h1, hab)
    np.testing.assert_allclose(s1, np.clip(hab * 1.0 * t1 / 0.6, 0, 1), atol=1e-6)   # September: 1.0
    assert sd.max() < s1.max() and td.max() < t1.max()  # a bad day stays low (not renormalised)
    # August uses its own month factor
    ds = _dataset(("2025-06-10", "2025-08-20"), tmean=15.0, precip=6.0)
    sa, _, ta = P.score_day("2025-08-20", FakeSource(ds), **kw)
    np.testing.assert_allclose(sa, np.clip(hab * 0.85 * ta / 0.6, 0, 1), atol=1e-6)
    # winter -> nothing
    ds = _dataset(("2025-11-20", "2026-01-15"), tmean=0.0, precip=2.0)
    sw, _, _ = P.score_day("2026-01-15", FakeSource(ds), **kw)
    assert sw.max() == 0.0


def test_season_factor_matches_prior_month_table():
    assert P.season_factor("2025-09-30") == prior.season_factor(9) == 1.0
    assert P.season_factor(date(2026, 1, 15)) == 0.0


def test_committed_reference_is_sane():
    ref = P.load_reference()
    assert 0.3 < ref < 1.0
    meta = json.loads(P.REFERENCE_PATH.read_text())
    assert meta["quantile"] >= 99 and meta["raw_quantiles"]["p99"] <= ref * 1.01 + 1e-9


# --- Open-Meteo (mocked) --------------------------------------------------------------------
TODAY = pd.Timestamp("2026-10-24")


def _om_forecast_location(lat, lon, elevation, past=62, ahead=7, temp=12.0):
    d = pd.date_range(TODAY - pd.Timedelta(days=past), periods=past + ahead, freq="D")
    h = pd.date_range(d[0], periods=24 * len(d), freq="h")
    n, nh = len(d), len(h)
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in h], "snow_depth": [0.0] * nh,
              "soil_moisture_0_to_1cm": [0.1] * nh, "soil_moisture_1_to_3cm": [0.2] * nh,
              "soil_moisture_3_to_9cm": [0.3] * nh, "soil_temperature_0cm": [temp + 2] * nh,
              "soil_temperature_6cm": [temp - 2] * nh}
    return {"latitude": lat, "longitude": lon, "elevation": elevation, "timezone": "Europe/Ljubljana",
            "daily": {"time": [t.strftime("%Y-%m-%d") for t in d],
                      "temperature_2m_mean": [temp + 0.1 * (lat - 46)] * n,
                      "temperature_2m_max": [temp + 6] * n, "temperature_2m_min": [temp - 6] * n,
                      "precipitation_sum": [4.0] * n},
            "hourly": hourly}


@pytest.fixture
def om_calls(monkeypatch):
    calls = []

    def fake_get(url, params):
        calls.append((url, params))
        lats = [float(x) for x in params["latitude"].split(",")]
        lons = [float(x) for x in params["longitude"].split(",")]
        els = [float(x) for x in params["elevation"].split(",")]
        assert len(lats) == len(lons) == len(els)
        out = [_om_forecast_location(a, o, e, params["past_days"], params["forecast_days"])
               for a, o, e in zip(lats, lons, els)]
        return out if len(out) > 1 else out[0]

    monkeypatch.setattr(w, "_get_json", fake_get)
    return calls


def test_era5_points_cover_the_bbox_with_120_points():
    pts = P.era5_points()
    assert len(pts) == 120 and pts[0] == (45.25, 13.25) and pts[-1] == (47.0, 16.75)


def test_coarse_mean_elevation_boxes_and_fill():
    fg = _fine(elev=0.0)
    fg.elev = (fg.lat - 46.0) * 1000.0                 # 50 .. 450 m, increases northwards
    pts = [(la, lo) for la in (45.75, 46.0, 46.25, 46.5) for lo in (14.0, 14.25, 14.5, 14.75)]
    z = dict(zip(pts, P.coarse_mean_elevation(pts, fg)))
    assert z[(46.25, 14.25)] == pytest.approx(250.0, abs=30)
    assert z[(46.5, 14.5)] > z[(46.25, 14.5)] > z[(46.0, 14.5)] - 1
    assert all(np.isfinite(v) for v in z.values())     # point with no cells in its box is filled


def test_openmeteo_source_requests_and_parsing(om_calls):
    pts = P.era5_points()
    elev = [100.0 + i for i in range(len(pts))]
    src = P.OpenMeteoSource(points=pts, elevations=elev)
    ds = src.dataset()
    assert len(om_calls) == 3                          # 120 points / 50 per call
    url, q = om_calls[0]
    assert url == w.OPENMETEO_FORECAST
    assert q["past_days"] == 62 and q["forecast_days"] == 7 and q["timezone"] == "Europe/Ljubljana"
    assert len(q["latitude"].split(",")) == 50 and len(q["elevation"].split(",")) == 50
    assert q["elevation"].split(",")[:2] == ["100", "101"]
    assert "relative_humidity_2m" not in q["hourly"] and "snow_depth" in q["hourly"]
    assert len(om_calls[2][1]["latitude"].split(",")) == 20
    assert ds.tmean.dims == ("time", "lat", "lon") and ds.tmean.shape == (69, 8, 15)
    assert ds.time.values[0] == (TODAY - pd.Timedelta(days=62)).to_datetime64()
    assert ds.time.values[-1] == (TODAY + pd.Timedelta(days=6)).to_datetime64()
    # the orography is the elevation the API echoes back (= the one we passed)
    assert float(ds.orography.sel(lat=45.25, lon=13.25)) == 100.0
    assert float(ds.orography.sel(lat=47.0, lon=16.75)) == 219.0
    assert float(ds.swvl1.mean()) == pytest.approx((0.1 + 2 * 0.2 + 4 * 0.3) / 7, abs=1e-6)
    assert float(ds.stl1.mean()) == pytest.approx(12.0, abs=1e-6)
    assert float(ds.precip.mean()) == 4.0 and np.isnan(ds.rh.values).all()
    assert src.dataset() is ds and len(om_calls) == 3  # fetched once


def test_openmeteo_end_to_end_trigger_and_horizon(om_calls):
    src = P.OpenMeteoSource(points=P.era5_points(), elevations=[300.0] * 120)
    fg = P.FineGrid(*np.meshgrid(np.linspace(13.6, 16.4, 20), np.linspace(46.7, 45.5, 10)),
                    np.full((10, 20), 300.0, "float32"))
    days = [(TODAY + pd.Timedelta(days=i)).date() for i in range(-2, 7)]
    t = P.trigger_grid(days, prior.PriorParams(), src, fg)
    assert t.shape == (9, 10, 20) and np.isfinite(t).all() and t.min() >= 0 and t.max() <= 1
    assert len(om_calls) == 3
    with pytest.raises(ValueError, match="ends"):
        P.trigger_grid([(TODAY + pd.Timedelta(days=7)).date()], None, src, fg)


def test_openmeteo_count_mismatch_is_an_error(monkeypatch):
    monkeypatch.setattr(w, "_get_json", lambda u, p: [_om_forecast_location(46.0, 14.0, 300.0)])
    with pytest.raises(RuntimeError, match="locations"):
        w.fetch_openmeteo_forecast([(46.0, 14.0), (46.0, 14.25)], past_days=62, forecast_days=7)


# --- static inputs / pack / reports ----------------------------------------------------------
def test_runtime_pack_roundtrip_and_fallback(tmp_path, monkeypatch):
    interim = {n: tmp_path / "interim" / f"{n}.tif" for n in P.STATIC_FILES}
    monkeypatch.setattr(P, "STATIC_FILES", {n: (interim[n], f"{n}.tif") for n in P.STATIC_FILES})
    monkeypatch.setattr(P, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.delenv("JURCKI_RUNTIME", raising=False)
    P._static.cache_clear()
    with pytest.raises(FileNotFoundError):             # neither interim nor pack
        P.load_habitat()
    shape = grid.grid_shape()
    rng = np.random.default_rng(0)
    arrays = {"habitat": rng.random(shape).astype("float32"),
              "elev": (rng.random(shape) * 2000).astype("float32"),
              "mask": rng.random(shape) > 0.5}
    for n, a in arrays.items():
        grid.write_tif(interim[n], a.astype("uint8") if n == "mask" else a,
                       **({"nodata": None} if n == "mask" else {}))
    sizes = P.pack_runtime()
    assert sizes["total"] == sum(v for k, v in sizes.items() if k != "total")
    for n in interim:                                  # remove data/interim -> the pack is used
        interim[n].unlink()
    P._static.cache_clear()
    assert P.static_path("habitat") == tmp_path / "runtime" / "habitat.tif"
    np.testing.assert_array_equal(P.load_habitat(), arrays["habitat"])   # lossless
    np.testing.assert_array_equal(P.load_elev(), arrays["elev"])
    np.testing.assert_array_equal(P.load_mask(), arrays["mask"])
    P._static.cache_clear()


def test_gbif_reports_last_30_days(tmp_path):
    df = pd.DataFrame({
        "gbifid": [1, 2, 3, 4, 5], "species": ["Boletus edulis"] * 5,
        "decimallatitude": [46.0, 46.1, 46.2, 48.5, 46.3],
        "decimallongitude": [14.0, 14.1, 14.2, 14.0, 14.3],
        "coordinateuncertaintyinmeters": [10.0, np.nan, 500.0, 5.0, 20.0],
        "date": pd.to_datetime(["2026-10-20", "2026-09-20", "2026-09-25", "2026-10-20", "2026-10-01"]),
        "is_jurcki": [True, True, True, True, False]})
    p = tmp_path / "g.parquet"
    df.to_parquet(p)
    r = P.gbif_reports("2026-10-24", path=p)
    assert len(r) == 2        # ids 1 and 3; 2 is too old, 4 is outside the map, 5 is not a jurcki
    assert set(r.source) == {"gbif"} and sorted(u.rsplit("/", 1)[1] for u in r.url) == ["1", "3"]
    assert P.gbif_reports("2026-10-24", path=tmp_path / "none.parquet").empty


def test_cli_openmeteo_days_window():
    # today - 2 .. today + 6 with the first forecast day = today + 1
    assert (P.OUT_PAST, P.OUT_AHEAD, P.OM_FORECAST_DAYS) == (2, 6, 7)
    assert P.OM_PAST_DAYS >= P.LOOKBACK + P.OUT_PAST


def test_runtime_pack_is_sane():
    """The committed runtime pack must decode to the real layers (it is all CI has)."""
    P._static.cache_clear()
    mask = P.load_mask(runtime=True)
    elev = P.load_elev(runtime=True)
    assert 320_000 < mask.sum() < 330_000          # Slovenia ~ 20 273 km2 of 250 m cells
    assert not mask[0].any() and not mask[-1].any()
    assert 50 < np.median(elev[mask]) < 1000 and elev[mask].max() < 2900
    P._static.cache_clear()
