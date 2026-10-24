"""Tests for the on-phone model package (jurcki.app_package) and its reference engine.

* the spec engine (``app_engine``) reproduces production code (``model.prior``, ``predict``,
  ``export``) on random inputs;
* the written package round-trips (decode -> equals the source rasters within quantisation);
* the parity vectors are self-consistent: the spec engine reproduces every expected value.

The package / vector tests need the files written by ``python -m jurcki.app_package`` and the
runtime rasters (``data/processed/runtime``); they are skipped when those are missing.  No network.
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from jurcki import app_engine as E
from jurcki import app_package as A
from jurcki import export, predict
from jurcki.model import prior, trigger

PKG = A.ASSETS_DIR
VEC = A.VECTORS_DIR
needs_pkg = pytest.mark.skipif(not (PKG / "model.json").exists(), reason="package not built")
needs_vec = pytest.mark.skipif(not (VEC / "cells.json").exists(), reason="vectors not built")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# spec engine == production on random data
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("params", [prior.PriorParams(), trigger.load_trigger_params(),
                                    prior.PriorParams(t_window=7, p_window=14, heat_window=3, frost_window=14,
                                                      pulse_lag=(3, 6))])
@pytest.mark.parametrize("n_days", [9, 30, 70])
def test_coarse_features_match_prior(params, n_days):
    rng = np.random.default_rng(n_days)
    shp = (n_days, 3, 4)
    tmean = rng.normal(14, 7, shp)
    w = {"tmean": tmean, "tmax": tmean + rng.uniform(1, 9, shp), "tmin": tmean - rng.uniform(1, 9, shp),
         "precip": np.maximum(0, rng.normal(-1, 7, shp))}
    ref = prior.weather_features(w["tmean"], w["tmin"], w["tmax"], w["precip"], params)
    got = E.coarse_features(w, trigger.params_to_dict(params))
    for k in E.TRIGGER_FEATURES:
        np.testing.assert_allclose(got[k], ref[k], atol=1e-9, err_msg=k)
    np.testing.assert_allclose(E.trigger_score(got, trigger.params_to_dict(params)),
                               np.clip(prior.trigger_score(ref, params), 0, 1), atol=1e-12)


def test_gap_filling_matches_prepare_weather():
    """Interior / leading / trailing holes and a point without any data, through the production path."""
    import xarray as xr
    rng = np.random.default_rng(3)
    T, nl, no = 40, 2, 3
    times = pd.date_range("2025-08-01", periods=T)
    base = {v: rng.normal(10, 3, (T, nl, no)) for v in E.VARS}
    for v in base:
        base[v][5:8, 0, 1] = np.nan
        base[v][:3, 1, 1] = np.nan
        base[v][-2:, 1, 2] = np.nan
        base[v][:, 0, 0] = np.nan
        base[v][:, 1, 0] = np.nan
    ds = xr.Dataset({k: (("time", "lat", "lon"), v) for k, v in {**base,
                     "swvl1": np.full((T, nl, no), 0.3), "stl1": np.full((T, nl, no), 10.0),
                     "snow_depth": np.zeros((T, nl, no))}.items()},
                    coords={"time": times, "lat": [46.0, 46.25], "lon": [14.25, 14.5, 14.75]})
    ds["orography"] = (("lat", "lon"), np.full((nl, no), 500.0))
    ref = predict.prepare_weather(ds)
    resp = []
    for i in range(nl):
        for j in range(no):
            d = {"time": [t.strftime("%Y-%m-%d") for t in times]}
            for v, name in E.OM_DAILY.items():
                d[name] = [None if np.isnan(x) else float(x) for x in base[v][:, i, j]]
            resp.append({"elevation": 500.0, "daily": d})
    dates, w, orog = E.daily_series(resp, nl, no)
    assert dates == [t.strftime("%Y-%m-%d") for t in times]
    for v in E.VARS:
        np.testing.assert_allclose(w[v], ref.w[v], atol=1e-9, err_msg=v)


def test_bilinear_and_downscale_match_predict():
    rng = np.random.default_rng(1)
    lat_axis = np.array([46.0, 46.25, 46.5, 46.75])
    lon_axis = np.array([14.0, 14.25, 14.5, 14.75, 15.0])
    lon = rng.uniform(13.7, 15.3, (6, 7))
    lat = rng.uniform(45.8, 47.0, (6, 7))
    elev = rng.uniform(0, 1500, lon.shape).astype("float32")
    fg = predict.FineGrid(lon, lat, elev)
    wp = predict._bilinear_weights(lat_axis, lon_axis, fg)
    we = E.bilinear_weights(lat, lon, 46.0, 14.0, 0.25, 4, 5)
    field = rng.normal(size=(2, 4, 5))
    np.testing.assert_allclose(E.bilinear(field, we), predict.bilinear(field, wp), atol=1e-5)
    # the upper corner and the far corner use the clamped edge
    one = E.bilinear_weights(np.array([47.5]), np.array([12.0]), 46.0, 14.0, 0.25, 4, 5)
    assert np.isclose(E.bilinear(field[0], one)[0], field[0][-1, 0])


def test_hotspots_match_export_rule():
    rng = np.random.default_rng(2)
    score = np.round(np.clip(rng.random((60, 70)) * 1.3 - 0.2, 0, 1), 2).astype("float32")  # many ties, many clipped
    hab = np.round(rng.random(score.shape), 2).astype("float32")
    trig = np.round(rng.random(score.shape), 2).astype("float32")
    mask = rng.random(score.shape) > 0.1
    rank = score + 1e-3 * np.nan_to_num(hab * trig)
    ref = export.pick_hotspots(rank, mask)
    got = E.pick_hotspots(rank, mask, 0.15, 6.0, 300)
    assert got == ref and len(ref) > 5


def test_colour_rule_matches_export():
    lut = E.colour_lut(export.LEGEND, 64, export.MIN_SCORE, export.ALPHA_FULL_AT, export.ALPHA_MAX)
    np.testing.assert_array_equal(lut, export.colour_lut(64))
    s = np.concatenate([np.linspace(0, 1, 301), [np.nan]]).reshape(1, -1)
    np.testing.assert_array_equal(E.colourise(s, lut), export.colorize(s, 64))


def test_geometry_matches_export_helpers():
    geo = A.phone_geometry()
    lon = np.array([13.4, 14.5058, 16.6, 15.0])
    lat = np.array([46.8, 46.0569, 45.42, 46.3])
    r, c = E.lonlat_to_pixel(lon, lat, geo)
    r2, c2 = export.lonlat_to_pixel(lon, lat, geo["bounds"], (geo["ny"], geo["nx"]))
    np.testing.assert_allclose(r, r2, atol=1e-2)
    np.testing.assert_allclose(c, c2, atol=1e-2)
    # centre of pixel (row, col) maps back to (row + 0.5, col + 0.5)
    lo, la = E.pixel_centres(geo)
    r3, c3 = E.lonlat_to_pixel(lo[[0, 500, 1008]], la[[0, 300, 660]], geo)
    np.testing.assert_allclose(r3, [0.5, 300.5, 660.5], atol=1e-6)
    np.testing.assert_allclose(c3, [0.5, 500.5, 1008.5], atol=1e-6)
    assert geo["nx"] * geo["ny"] < 800_000
    assert abs(geo["ground_px_m_at_46N"] - 250) < 2


# --------------------------------------------------------------------------------------------
# package round trip
# --------------------------------------------------------------------------------------------
@needs_pkg
def test_package_files_small():
    total = sum(p.stat().st_size for p in PKG.iterdir())
    assert total < 4_000_000


@needs_pkg
def test_rasters_round_trip_within_quantisation():
    geo = A.phone_geometry()
    model = _load(PKG / "model.json")
    assert {k: model["raster"][k] for k in ("nx", "ny", "px", "x0", "y_top")} == \
           {k: geo[k] for k in ("nx", "ny", "px", "x0", "y_top")}
    r = A.load_package_rasters(PKG, geo)
    assert r["habitat"].dtype == np.uint8 and r["elev"].dtype == np.dtype("<i2") and r["mask"].dtype == np.uint8
    mask = r["mask"].astype(bool)
    # nodata exactly outside the mask
    assert (r["habitat"][~mask] == E.HAB_NODATA).all() and (r["elev"][~mask] == E.ELEV_NODATA).all()
    assert (r["habitat"][mask] <= 250).all() and (r["elev"][mask] > -100).all()
    # compare with a fresh resampling of the source (quantisation: 1/500 habitat, 0.5 m)
    fresh = A.build_rasters(geo)
    np.testing.assert_array_equal(fresh["mask"], r["mask"])
    np.testing.assert_array_equal(fresh["habitat"], r["habitat"])
    np.testing.assert_array_equal(fresh["elev"], r["elev"])
    rc = A._model_coords(geo)
    land = predict.load_mask()
    hab = A.resample_masked(np.nan_to_num(predict.load_habitat()), land, rc)
    el = A.resample_masked(predict.load_elev(), land, rc)
    assert np.abs(E.decode_habitat(r["habitat"])[mask] - hab[mask]).max() <= 0.5 / 250 + 1e-9
    assert np.abs(E.decode_elev(r["elev"])[mask] - el[mask]).max() <= 0.5 + 1e-9
    # Slovenia area: land pixels x pixel area ~ 20 273 km2 (ground pixel ~250 m)
    area = mask.sum() * (geo["ground_px_m_at_46N"] / 1000) ** 2
    assert 19_500 < area < 21_200


@needs_pkg
def test_model_json_matches_sources():
    m = _load(PKG / "model.json")
    tp = _load(A.config.PROCESSED / "trigger_params.json")
    assert m["trigger"]["params"] == tp["params"]
    assert m["score"]["reference"] == _load(predict.REFERENCE_PATH)["reference"]
    assert m["score"]["season_factor"] == {str(k): prior.season_factor(k) for k in range(1, 13)}
    assert m["colormap"]["legend"] == export.LEGEND
    assert m["colormap"]["lut"] == export.colour_lut(64).tolist()
    assert m["lapse_rate_k_per_m"] == -6.5e-3
    lat = m["weather"]["lattice"]
    assert (lat["n_lat"], lat["n_lon"], len(lat["points"])) == (8, 15, 120)
    assert [(p[0], p[1]) for p in lat["points"]] == predict.era5_points()
    reqs = m["weather"]["forecast_recent"]["requests"]
    assert len(reqs) == 3 and all("hourly" not in q["params"] for q in reqs)
    assert [len(q["params"]["latitude"].split(",")) for q in reqs] == [50, 50, 20]
    assert m["weather"]["forecast_recent"]["past_days"] == 62 and m["weather"]["forecast_recent"]["forecast_days"] == 7
    assert m["validation"] is None or m["validation"]["days"]
    json.dumps(m, allow_nan=False)


@needs_pkg
def test_stats_and_finds_consistent():
    s = _load(PKG / "stats.json")
    f = _load(PKG / "finds.json")
    tp = _load(A.config.PROCESSED / "trigger_params.json")
    rows = {r["key"]: r for r in s["trigger"]["rows"]}
    assert rows["tuned_loyo:comb_tol"]["tolerant_3d"]["auc"] == pytest.approx(tp["metrics"]["tuned_loyo"]["comb_tol"]["auc"])
    assert rows["season_only:comb_tol"]["tolerant_3d"]["n"] == tp["n_finds"]
    assert sum(r["n"] for r in s["per_year"]) == tp["n_finds"]
    assert {p["name"] for p in s["params"]} == set(trigger.TRIGGER_FIELDS)
    assert all(p["sl"] and p["en"] for p in s["params"])
    assert s["weekend_effect"]["off_day_share"] > s["weekend_effect"]["baseline_share"]
    b = A.phone_geometry()["bounds"]
    assert f["n"] == len(f["finds"]) > 100
    for r in f["finds"]:
        assert r["date"] >= "2010-01-01" and b[0] <= r["lon"] <= b[2] and b[1] <= r["lat"] <= b[3]
        assert set(r) == {"date", "lon", "lat", "species", "country", "source", "precision_m", "habitat",
                          "p_trigger", "p_season_trigger"}      # no recorder / gbif ids
        for k in ("habitat", "p_trigger", "p_season_trigger"):
            assert r[k] is None or 0 <= r[k] <= 1
    assert sum(r["p_trigger"] is not None for r in f["finds"]) == s["data"]["in_backtest"]


# --------------------------------------------------------------------------------------------
# parity vectors: the spec engine reproduces production's numbers
# --------------------------------------------------------------------------------------------
@needs_vec
def test_vector_small_weather():
    v = _load(VEC / "weather_small.json")
    tol, ex = v["tolerance"], v["expected"]
    nl, no = v["lattice"]["n_lat"], v["lattice"]["n_lon"]
    dates, w, orog = E.daily_series(v["responses"], nl, no)
    assert dates == ex["dates"]
    np.testing.assert_allclose(orog.ravel(), ex["orography"])
    for k in E.VARS:
        np.testing.assert_allclose(w[k].reshape(len(dates), -1).T, np.array(ex["daily"][k], float), atol=tol["daily"], err_msg=k)
    cf = E.coarse_features(w, v["params"])
    for k in E.TRIGGER_FEATURES:
        np.testing.assert_allclose(cf[k].reshape(len(dates), -1).T, np.array(ex["features"][k], float), atol=tol["features"], err_msg=k)
    trig = E.trigger_score(cf, v["params"])
    np.testing.assert_allclose(trig.reshape(len(dates), -1).T, np.array(ex["trigger"], float), atol=tol["trigger"])
    # the vector really exercises the interesting cases
    assert any(x is None for r in v["responses"] for x in r["daily"]["temperature_2m_mean"])
    assert 0 < np.mean(np.array(ex["features"]["pulse"]) > 0) < 1


@needs_vec
def test_vector_cells():
    v = _load(VEC / "cells.json")
    wf = _load(VEC / v["weather_file"])
    model = _load(PKG / "model.json")
    lat = model["weather"]["lattice"]
    dates, w, orog = E.daily_series(wf["responses"], lat["n_lat"], lat["n_lon"])
    cf = E.coarse_features(w, v["params"])
    cells = v["cells"]
    lon = np.array([c["lon"] for c in cells]); la = np.array([c["lat"] for c in cells])
    elev = np.array([c["elev"] for c in cells]); hab = np.array([c["habitat"] for c in cells])
    wts = E.bilinear_weights(la, lon, lat["lat0"], lat["lon0"], lat["step"], lat["n_lat"], lat["n_lon"])
    tol = v["tolerance"]
    fractional = False
    for day in v["days"]:
        ex = v["expected"][day]
        i = dates.index(day)
        f = E.downscale_features({k: a[i] for k, a in cf.items()}, orog, elev, wts)
        for k in E.TRIGGER_FEATURES:
            np.testing.assert_allclose(f[k], np.array(ex["features"][k], float), atol=tol["features"], err_msg=f"{day} {k}")
        fractional |= bool(((f["pulse"] > 0.01) & (f["pulse"] < 0.99)).any())
        trig = E.trigger_score(f, v["params"])
        np.testing.assert_allclose(trig, np.array(ex["trigger"], float), atol=tol["trigger"])
        s = E.season_factor(day, v["season_factor_table"])
        assert s == ex["season_factor"]
        raw = hab * s * trig
        np.testing.assert_allclose(raw, np.array(ex["raw"], float), atol=tol["score"])
        np.testing.assert_allclose(E.scale_score(hab, s, trig, v["reference"]), np.array(ex["score"], float), atol=tol["score"])
    assert fractional, "vector should contain fractional pulse values"
    assert v["expected"][v["days"][0]]["season_factor"] != v["expected"][v["days"][1]]["season_factor"]


@needs_vec
def test_vector_cells_match_package_rasters():
    """Raster cells in cells.json carry the decoded package values and the pixel-centre coordinates."""
    v = _load(VEC / "cells.json")
    geo = A.phone_geometry()
    r = A.load_package_rasters(PKG, geo)
    lon, lat = E.pixel_centres(geo)
    for c in v["cells"]:
        if c["synthetic"]:
            continue
        assert r["mask"][c["row"], c["col"]] == 1
        assert c["habitat"] == pytest.approx(r["habitat"][c["row"], c["col"]] / 250.0)
        assert c["elev"] == float(r["elev"][c["row"], c["col"]])
        assert c["lon"] == pytest.approx(lon[c["col"]], abs=1e-8) and c["lat"] == pytest.approx(lat[c["row"]], abs=1e-8)


@needs_vec
def test_vector_geometry():
    v = _load(VEC / "geometry.json")
    geo = v["raster"]
    tol = v["tolerance"]
    for p in v["lonlat_to_pixel"]:
        r, c = E.lonlat_to_pixel(p["lon"], p["lat"], geo)
        assert abs(r - p["row"]) < 1e-5 and abs(c - p["col"]) < 1e-5
        assert abs(r - p["export_row"]) < tol["pixel_rowcol"] and abs(c - p["export_col"]) < tol["pixel_rowcol"]
        inside = 0 <= r < geo["ny"] and 0 <= c < geo["nx"]
        assert (p["pixel"] is not None) == inside
        if inside:
            assert p["pixel"] == [int(np.floor(r)), int(np.floor(c))]
    assert any(p["pixel"] is None for p in v["lonlat_to_pixel"])
    lon, lat = E.pixel_centres(geo)
    for p in v["pixel_centres"]:
        assert abs(lon[p["col"]] - p["lon"]) < tol["lonlat_deg"] and abs(lat[p["row"]] - p["lat"]) < tol["lonlat_deg"]


@needs_vec
def test_vector_misc():
    v = _load(VEC / "misc.json")
    ref = v["reference"]
    for c in v["scoring"]:
        raw = c["habitat"] * c["season"] * c["trigger"]
        assert abs(raw - c["raw"]) < 1e-6
        assert abs(E.scale_score(raw, 1.0, 1.0, ref) - c["score"]) < 1e-6
    assert any(c["score"] == 1.0 for c in v["scoring"]) and any(c["score"] == 0.0 for c in v["scoring"])
    model = _load(PKG / "model.json")
    lut = E.colour_lut(model["colormap"]["legend"], 64)
    assert lut.tolist() == v["lut"]
    for c in v["colours"]:
        s = np.array([[np.nan if c["score"] is None else c["score"]]])
        assert E.colourise(s, lut)[0, 0].tolist() == c["rgba"]
    h = v["hotspots"]
    score, hab, trig = (np.array(h[k], "float32") for k in ("score", "habitat", "trigger"))
    rank = score + np.float32(1e-3) * np.nan_to_num(hab * trig)
    picks = E.pick_hotspots(rank, np.array(h["mask"], bool), h["min_score"], h["min_sep_px"], h["max_n"])
    assert [list(p) for p in picks] == h["expected_picks"]
    assert len(picks) >= 8


@needs_vec
def test_phone_engine_vs_production_cells_end_to_end():
    """End to end on the real rasters: the spec engine's score for one pixel equals production's value
    at the same coordinates (production code on a 1-cell FineGrid)."""
    v = _load(VEC / "cells.json")
    wf = _load(VEC / v["weather_file"])
    model = _load(PKG / "model.json")
    lat = model["weather"]["lattice"]
    from jurcki import app_package as AP
    ds = AP.production_dataset(wf["responses"], lat["points"])
    src = predict.OpenMeteoSource(points=[(p[0], p[1]) for p in lat["points"]],
                                  elevations=[p[2] for p in lat["points"]], fetch=lambda *a, **k: ds)
    params = trigger.params_from_dict(model["trigger"]["params"])
    c = v["cells"][1]
    fg = predict.FineGrid(np.array([[c["lon"]]]), np.array([[c["lat"]]]), np.array([[c["elev"]]], "float32"))
    res = next(predict.score_days([date.fromisoformat(v["days"][0])], src, params, v["reference"], fg,
                                  np.array([[c["habitat"]]], "float32")))
    assert res.score[0, 0] == pytest.approx(v["expected"][v["days"][0]]["score"][1], abs=1e-5)
