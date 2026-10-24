import numpy as np
import pandas as pd
import pytest
import xarray as xr

from jurcki import features
from jurcki.model import prior, trigger

N, YEAR = 4, 2020


def make_panel(score_seed=0, finds_t=(120, 150, 180, 200)):
    """Four series of one year with random weather; one find each."""
    rng = np.random.default_rng(score_seed)
    t = np.arange(trigger.T)[:, None]
    base = 13 + 8 * np.sin((t - 60) / 260 * 2 * np.pi)
    w = {"tmean": base + rng.normal(0, 2, (trigger.T, N)), "precip": rng.gamma(0.4, 8, (trigger.T, N)),
         "swvl1": 0.3 + 0.05 * rng.random((trigger.T, N)), "stl1": base + rng.normal(0, 1, (trigger.T, N)),
         "snow_depth": np.zeros((trigger.T, N))}
    w["tmin"], w["tmax"] = w["tmean"] - 5, w["tmean"] + 6
    series = pd.DataFrame({"lon": 14.5, "lat": 46.0, "elev": 600.0, "year": YEAR}, index=range(N))
    dates = pd.Timestamp(f"{YEAR}-03-01") + pd.to_timedelta(finds_t, unit="D")
    finds = pd.DataFrame({"sid": range(N), "date": dates})
    return trigger.assemble(series, w, np.ones((trigger.T, N), bool), finds)


def test_assemble_drops_finds_outside_season_or_data():
    series = pd.DataFrame({"lon": [14.5], "lat": [46.0], "elev": [600.0], "year": [YEAR]})
    w = {k: np.zeros((trigger.T, 1)) for k in features.WEATHER_VARS}
    valid = np.ones((trigger.T, 1), bool)
    valid[200:] = False   # history ends
    d = pd.to_datetime(["2020-04-10", "2020-06-15", "2020-09-20", "2020-10-30"])
    finds = pd.DataFrame({"sid": 0, "date": d})
    p = trigger.assemble(series, w, valid, finds)
    assert list(p.finds["date"].dt.month) == [6]   # April is before the window, later ones lack weather
    assert p.dates[0, 0] == np.datetime64("2020-03-01") and p.dates[-1, 0] == np.datetime64("2020-11-15")


@pytest.mark.parametrize("tol", [0, 3])
@pytest.mark.parametrize("matched", [False, True])
def test_fast_percentiles_match_evaluate(tol, matched):
    p = make_panel()
    score = np.round(np.random.default_rng(1).random((trigger.T, N)), 1)   # many ties
    a = trigger.percentiles(p, score, tol, matched)
    b = trigger.percentiles_ref(p, score, tol, matched)
    assert np.allclose(a, b, equal_nan=True)


def test_peak_on_find_day_ranks_top_and_flat_is_half():
    p = make_panel()
    score = np.zeros((trigger.T, N))
    score[p.finds["t"].to_numpy(), np.arange(N)] = 1.0
    assert np.allclose(trigger.percentiles(p, score), 1.0)
    assert np.allclose(trigger.percentiles(p, np.ones((trigger.T, N))), 0.5)


def test_tolerant_variant_credits_a_flush():
    p = make_panel()
    score = np.zeros((trigger.T, N))
    t = p.finds["t"].to_numpy()
    score[t + 3, np.arange(N)] = 1.0      # the signal peaks 3 days after the logged find
    exact, tol = trigger.percentiles(p, score), trigger.percentiles(p, score, trigger.TOL)
    assert np.all(exact < 0.5) and np.all(tol > 0.95)


def test_weekend_matched_background():
    p = make_panel()
    off = p.off_day[:, 0]
    score = np.where(off, 1.0, 0.0)[:, None] * np.ones((1, N))   # high only on weekend days
    p.finds["t"] = p.finds["t"].where(~off[p.finds["t"]], p.finds["t"] + 1)   # finds on workdays
    p.finds["t"] = np.where(off[p.finds["t"]], p.finds["t"] + 2, p.finds["t"])
    t = p.finds["t"].to_numpy()
    if off[t].any():
        pytest.skip("synthetic finds landed on a weekend")
    assert trigger.percentiles(p, score)[0] < 0.5                    # workday looks bad against all days
    assert trigger.percentiles(p, score, 0, True)[0] == 0.5          # but equal among workdays


def test_is_off_day():
    d = pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-04", "2020-04-13", "2020-05-09", "2020-06-24"])
    assert list(trigger.is_off_day(d)) == [True, True, False, True, True, False]


def test_sample_params():
    c = trigger.sample_params(50, seed=3)
    base = prior.PriorParams()
    assert c[0] == base and len(c) == 50
    assert all(p.t_window == 20 and p.p_window == 26 for p in c)
    assert all(p.pulse_lag[1] > p.pulse_lag[0] and 8 <= p.t_opt <= 20 for p in c)
    free = trigger.sample_params(50, seed=3, free_windows=True)
    assert len({p.t_window for p in free}) > 1
    assert c[1].t_opt == free[1].t_opt            # same stream of the other parameters
    assert trigger.sample_params(5, seed=3) == trigger.sample_params(5, seed=3)


def test_loyo_selection_never_sees_the_held_out_year():
    years = np.array([2010, 2010, 2011, 2011, 2012, 2012])
    m = np.array([[0.9, 0.9, 0.3, 0.3, 0.3, 0.3],     # perfect in 2010 only
                  [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]])
    cv = trigger.loyo_select({"comb_tol": m}, years)
    assert cv["chosen"][2010] == 1 and cv["chosen"][2011] == 0     # 0 wins on the rest only when 2010 is in
    assert np.allclose(cv["held"]["comb_tol"], [0.6, 0.6, 0.3, 0.3, 0.3, 0.3])
    assert cv["train_auc"][2010] == pytest.approx(0.6)


def test_search_includes_literature_and_scores_every_find():
    p = make_panel()
    c = trigger.sample_params(6, seed=1)
    mats = trigger.search(p, c)
    assert mats["comb_tol"].shape == (6, len(p.finds))
    lit = trigger.all_variants(p, trigger.trigger_scores(p, c[0]), trigger.season_scores(p))
    assert np.allclose(mats["comb_exact"][0], lit["comb_exact"])
    assert np.all((mats["comb_tol"] >= 0) & (mats["comb_tol"] <= 1))


def test_year_bootstrap_interval_brackets_mean():
    years = np.repeat(np.arange(2010, 2020), 5)
    a = np.random.default_rng(0).random(50) + 0.1
    m, lo, hi = trigger.year_bootstrap(years, a, np.zeros(50), n=200)
    assert lo <= m <= hi


def test_ml_pipeline_runs_and_scores_the_panel():
    p = make_panel()
    X = trigger.ml_matrix(p)
    assert X.shape == (trigger.T, N, len(trigger.ML_FEATURES))
    for kind in ("lgbm", "logit"):
        _, predict = trigger.fit_ml(p, X, np.arange(N), kind)
        assert predict(X).shape == (trigger.T, N)
    _, predict = trigger.fit_ml(p, X, np.arange(N), "logit", ["t20", "doy"])
    assert np.isfinite(predict(X)).all()


def test_params_file_roundtrip(tmp_path):
    q = prior.PriorParams(t_opt=15.5, pulse_lag=(9, 18), p_window=26)
    d = trigger.params_to_dict(q)
    path = tmp_path / "p.json"
    import json
    path.write_text(json.dumps({"params": d}))
    got = trigger.load_trigger_params(path)
    assert got.t_opt == 15.5 and got.pulse_lag == (9, 18) and got.elev_sd == prior.PriorParams().elev_sd
    assert trigger.load_trigger_params(tmp_path / "missing.json") == prior.PriorParams()
    assert trigger.load_trigger_model(tmp_path / "missing.txt") is None


# --- features: dynamic section ---------------------------------------------
def fake_history(nan_days=()):
    time = pd.date_range("2020-01-01", "2020-12-31")
    lat, lon = [46.0, 46.25], [14.0, 14.25]
    shape = (len(time), 2, 2)
    ds = xr.Dataset(coords={"time": time, "lat": lat, "lon": lon})
    for k in features.WEATHER_VARS:
        ds[k] = (("time", "lat", "lon"), np.full(shape, 10.0))
    ds["precip"][:] = 2.0
    ds["snow_depth"][:] = 0.0
    ds["tmean"][:] = 10.0 + 0.001 * np.arange(len(time))[:, None, None]
    for i in nan_days:
        ds["tmean"][i] = np.nan
    ds["orography"] = (("lat", "lon"), np.full((2, 2), 500.0))
    return ds


def test_point_weather_lapse_interpolation_and_bounds(monkeypatch):
    monkeypatch.setattr(features, "_history", lambda: fake_history(nan_days=[10, 11]))
    df = features.point_weather(14.1, 46.1, 1000.0, "2020-01-01", "2020-03-01")
    assert df.attrs["orography"] == pytest.approx(500.0)
    assert df["tmean"].notna().all()
    assert df["tmean"].iloc[10] == pytest.approx(10.0 + 0.0095 - 3.25, abs=1e-3)   # filled across the gap, lapse -3.25 K
    assert df["tmin"].iloc[0] == pytest.approx(10.0 - 3.25)
    assert df["precip"].iloc[0] == pytest.approx(2.0)                              # not lapse-corrected
    same = features.point_weather(14.1, 46.1, None, "2020-01-01", "2020-01-05")
    assert same["tmean"].iloc[0] == pytest.approx(10.0)
    short = features.point_weather(14.1, 46.1, 500.0, "2020-12-20", "2021-01-10")
    assert short.index[-1] == pd.Timestamp("2020-12-31")                              # history ends
    with pytest.raises(ValueError):
        features.point_weather(20.0, 46.1, 500.0, "2020-01-01", "2020-01-05")


def test_dynamic_features_columns_and_snow_flag(monkeypatch):
    monkeypatch.setattr(features, "_history", lambda: fake_history())
    df = features.point_weather(14.1, 46.1, 500.0, "2020-01-01", "2020-06-30")
    df["snow_depth"] = 0.0
    df.loc[df.index[40], "snow_depth"] = 0.1
    df["swvl1"] = np.arange(len(df)) / 100
    f = features.dynamic_features(df)
    assert list(f.columns) == features.DYNAMIC_NAMES and len(f) == len(df)
    assert f["snow"].iloc[40:47].eq(1).all() and f["snow"].iloc[47] == 0 and f["snow"].iloc[39] == 0
    assert f["swvl1_7"].iloc[30] == pytest.approx(df["swvl1"].iloc[24:31].mean())
    assert f["swvl1_14"].iloc[30] == pytest.approx(df["swvl1"].iloc[17:31].mean())
    assert f["stl1_20"].iloc[50] == pytest.approx(10.0)
    assert f["p26"].iloc[60] == pytest.approx(26 * 2.0)
    ref = prior.weather_features(df["tmean"].to_numpy(), df["tmin"].to_numpy(), df["tmax"].to_numpy(), df["precip"].to_numpy())
    assert np.allclose(f["t20"], ref["t20"]) and np.allclose(f["trigger"], prior.trigger_score(ref))
