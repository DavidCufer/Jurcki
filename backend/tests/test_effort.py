"""Offline tests for jurcki.model.effort (synthetic data, no network)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from jurcki import features
from jurcki.model import effort as E


# --- statistics -------------------------------------------------------------------------
def test_wilson_ci_known_values():
    lo, hi = E.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = E.wilson_ci(0, 20)
    assert lo == 0 and 0.1 < hi < 0.2
    lo, hi = E.wilson_ci([1, 0], [0, 0])
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_poisson_ci_known_values():
    lo, hi = E.poisson_ci(0)
    assert lo == 0 and hi == pytest.approx(3.689, abs=1e-3)
    lo, hi = E.poisson_ci(10)
    assert lo == pytest.approx(4.795, abs=1e-3) and hi == pytest.approx(18.390, abs=1e-3)


def test_week_and_season():
    d = pd.to_datetime(["2020-01-01", "2020-01-07", "2020-01-08", "2020-12-31", "2021-12-31"])
    assert list(E.week_of_year(d)) == [1, 1, 2, 52, 52]
    d = pd.to_datetime(["2020-04-30", "2020-05-01", "2020-09-15", "2020-11-15", "2020-11-16"])
    assert list(E.in_season(d)) == [False, True, True, True, False]


def test_share_table():
    t = E.share_table([1, 0, 0, 0, 1, 1], ["a", "a", "a", "a", "b", "b"])
    assert t.loc["a", "n"] == 4 and t.loc["a", "k"] == 1 and t.loc["a", "share"] == 0.25
    assert t.loc["b", "lo"] < 1.0 <= t.loc["b", "hi"] + 1e-12


def test_observed_expected_flags_the_good_year():
    rng = np.random.default_rng(0)
    rows = []
    for yr in range(2010, 2020):
        for wk in range(30, 40):
            p = 0.10 * (2.0 if yr == 2015 else 1.0)
            n = 400
            rows.append(pd.DataFrame({"y": yr, "w": wk, "j": rng.random(n) < p}))
    d = pd.concat(rows)
    oe = E.observed_expected(d["j"], d["y"], d["w"])
    assert oe.loc[2015, "ratio"] > 1.6 and oe.loc[2015, "lo"] > 1.3
    assert oe.drop(2015)["ratio"].between(0.7, 1.15).all()
    assert (oe["lo"] <= oe["ratio"]).all() and (oe["ratio"] <= oe["hi"]).all()


def test_seasonal_shape_does_not_make_a_good_year():
    """A year with only late-season records must not look good just because late weeks have a high share."""
    rng = np.random.default_rng(1)
    rows = []
    for yr in range(2010, 2018):
        for wk in range(30, 40):
            n = 600 if (yr == 2012 and wk >= 36) else (50 if yr == 2012 else 300)
            rows.append(pd.DataFrame({"y": yr, "w": wk, "j": rng.random(n) < (0.02 if wk < 36 else 0.3)}))
    d = pd.concat(rows)
    oe = E.observed_expected(d["j"], d["y"], d["w"])
    assert 0.8 < oe.loc[2012, "ratio"] < 1.2


def test_spearman_boot():
    x = np.arange(15.0)
    r = E.spearman_boot(x, x ** 2, n_boot=200)
    assert r["rho"] == pytest.approx(1.0) and r["n"] == 15 and r["lo"] > 0.9
    rng = np.random.default_rng(0)
    r = E.spearman_boot(rng.normal(size=16), rng.normal(size=16), n_boot=300)
    assert r["lo"] < 0 < r["hi"]
    assert np.isnan(E.spearman_boot([1, 2], [2, 1])["rho"])


def test_multiple_testing_adjustments():
    p = np.array([0.001, 0.01, 0.03, 0.5])
    assert E.holm(p) == pytest.approx([0.004, 0.03, 0.06, 0.5])
    assert E.benjamini_hochberg(p) == pytest.approx([0.004, 0.02, 0.04, 0.5])
    assert (E.holm(p) >= p).all() and (E.benjamini_hochberg(p) >= p).all()


def test_auc_score_matches_pairs_and_weights():
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.4])
    y = np.array([0, 0, 1, 1, 1], bool)
    pos, neg = s[y], s[~y]
    expect = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert E.auc_score(s, y) == pytest.approx(expect)
    # integer weights == repeated rows
    w = np.array([1, 2, 1, 3, 1])
    rep = np.repeat(np.arange(5), w)
    assert E.auc_score(s, y, w) == pytest.approx(E.auc_score(s[rep], y[rep]))
    assert np.isnan(E.auc_score([1, 2], [True, True]))


def test_stratified_auc_removes_stratum_effect():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, 4000)
    y = rng.random(4000) < np.where(g == 1, 0.5, 0.05)
    s = g + 0.0 * rng.random(4000)               # score = stratum only
    assert E.auc_score(s, y) > 0.7
    assert E.stratified_auc(s, y, g) == pytest.approx(0.5, abs=0.02)
    s2 = g + y * 0.5 + rng.normal(0, 0.1, 4000)  # real signal inside strata
    assert E.stratified_auc(s2, y, g) > 0.9


def test_year_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(2010, 2020), 100)
    s = rng.normal(size=len(years))
    y = rng.random(len(years)) < 1 / (1 + np.exp(-s))
    est, lo, hi = E.year_bootstrap(lambda i: E.auc_score(s[i], y[i]), years, n_boot=100)
    assert lo <= est <= hi and 0.55 < est < 0.75


# --- calendar -----------------------------------------------------------------------------
def test_moon_phase_known_dates():
    ph = E.moon_phase(pd.to_datetime(["2000-01-06", "2000-01-21", "2024-04-08", "2024-04-23", "2024-04-16"]))
    circ = lambda x, c: min(abs(x - c), 1 - abs(x - c))
    assert circ(ph[0], 0.0) < 0.03          # new moon
    assert circ(ph[1], 0.5) < 0.03          # full moon
    assert circ(ph[2], 0.0) < 0.03          # new moon 2024-04-08
    assert circ(ph[3], 0.5) < 0.03          # full moon 2024-04-23
    assert circ(ph[4], 0.25) < 0.04         # first quarter
    assert E.moon_illumination(0.5) == pytest.approx(1.0) and E.moon_illumination(0.0) == pytest.approx(0.0)


def test_daylength():
    dl = E.daylength_hours([80, 172, 355], 46.0)
    assert dl[0] == pytest.approx(12.0, abs=0.3)
    assert dl[1] == pytest.approx(15.7, abs=0.3)
    assert dl[2] == pytest.approx(8.5, abs=0.4)
    assert E.daylength_hours(172, 89.0) == pytest.approx(24.0, abs=0.5)


# --- weather features ------------------------------------------------------------------
def _weather(T=120, N=3, seed=0):
    rng = np.random.default_rng(seed)
    base = 12 + 6 * np.sin(np.arange(T) / T * 2 * np.pi)[:, None]
    w = {"tmean": base + rng.normal(0, 1.5, (T, N)), "precip": rng.gamma(0.3, 8, (T, N)),
         "swvl1": 0.3 + 0.05 * rng.random((T, N)), "stl1": base + rng.normal(0, 0.5, (T, N)),
         "snow_depth": np.zeros((T, N))}
    w["tmin"], w["tmax"] = w["tmean"] - 4, w["tmean"] + 5
    return w


def test_wet_runs_and_days_since():
    pr = np.array([[0.0], [2.0], [3.0], [0.0], [5.0], [0.5]])
    tot, ln = E._wet_runs(pr)
    assert tot[:, 0].tolist() == [0, 2, 5, 0, 5, 0] and ln[:, 0].tolist() == [0, 1, 2, 0, 1, 0]
    ds = E._days_since(pr >= 1.0)
    assert ds[:, 0].tolist()[1:] == [0, 0, 1, 0, 1] and ds[0, 0] > 5


def test_cold_snap_feature_sees_a_drop_and_nothing_looks_ahead():
    w = _weather()
    w["tmean"][:] = 18.0
    w["tmean"][60:] = 8.0                       # 10 degree step down on day 60
    dates = pd.date_range("2020-05-01", periods=120)
    f = E.extra_features(w, dates)
    assert f["drop5"][59, 0] == pytest.approx(0.0)
    assert f["drop5"][62, 0] < -5                # last 5 days include the step
    assert f["coldsnap5_21"][62, 0] == 1 and f["coldsnap5_21"][59, 0] == 0
    # a change after day t must not move any feature up to day t
    w2 = {k: v.copy() for k, v in w.items()}
    w2["precip"][90:] += 50
    w2["tmean"][90:] += 10
    f2 = E.extra_features(w2, dates)
    for k, v in f.items():
        assert np.allclose(v[:90], f2[k][:90]), k


def test_rain_event_features_separate_size_and_duration():
    T = 60
    w = _weather(T, 2)
    w["precip"][:] = 0.0
    w["precip"][10:13, 0] = 5.0                  # slow event: 3 days x 5 mm = 15 mm
    w["precip"][10, 1] = 15.0                    # same total in one day
    f = E.extra_features(w, pd.date_range("2020-08-01", periods=T))
    assert f["event_mm30"][20, 0] == pytest.approx(15.0) and f["event_mm30"][20, 1] == pytest.approx(15.0)
    assert f["event_len30"][20, 0] == 3 and f["event_len30"][20, 1] == 1
    assert f["event_int30"][20, 1] == pytest.approx(15.0) and f["event_int30"][20, 0] == pytest.approx(5.0)
    assert f["maxday26"][20, 1] == 15.0 and f["maxday26"][20, 0] == 5.0
    assert f["rain_days26"][20, 0] == 3 and f["rain_days26"][20, 1] == 1
    assert f["days_since_10mm"][20, 0] == 7 and f["days_since_10mm"][20, 1] == 8
    assert f["intensity26"][20, 1] > f["intensity26"][20, 0]


def test_calendar_features_broadcast():
    w = _weather(30, 2)
    dates = pd.date_range("2020-06-10", periods=30)
    f = E.extra_features(w, dates, lat=46.0)
    assert f["daylength"].shape == (30, 2) and f["falling_light"][0, 0] == 0 and f["falling_light"][-1, 0] == 1
    assert np.allclose(f["moon_cos"] ** 2 + f["moon_sin"] ** 2, 1.0)
    assert f["near_full"].sum() > 0 and f["near_new"].sum() > 0
    assert np.all(f["doy"][:, 0] == dates.day_of_year)


def test_cell_weather_interpolates_and_applies_lapse():
    t = pd.date_range("2020-01-01", periods=10)
    lat, lon = np.array([45.5, 45.75, 46.0]), np.array([14.0, 14.25, 14.5])
    shape = (10, 3, 3)
    base = dict(tmean=10.0, tmin=5.0, tmax=15.0, precip=2.0, swvl1=0.3, stl1=10.0,
                dewpoint=5.0, rh=70.0, snow_depth=0.0)
    ds = xr.Dataset({k: (("time", "lat", "lon"), np.full(shape, v, "float32")) for k, v in base.items()},
                    coords={"time": t, "lat": lat, "lon": lon})
    ds["orography"] = (("lat", "lon"), np.full((3, 3), 300.0, "float32"))
    times, w = E.cell_weather([14.1, 14.3], [45.6, 45.9], [300.0, 1300.0], ds)
    assert len(times) == 10 and w["tmean"].shape == (10, 2)
    assert w["tmean"][0, 0] == pytest.approx(10.0, abs=1e-4)
    assert w["tmean"][0, 1] == pytest.approx(10.0 - 6.5, abs=1e-3)      # 1000 m higher
    assert w["precip"][0, 1] == pytest.approx(2.0)
    assert set(features.WEATHER_VARS) <= set(w)


# --- logistic regression ---------------------------------------------------------------
def test_fit_logit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 2))
    y = rng.random(20000) < 1 / (1 + np.exp(-(-1.0 + 0.8 * X[:, 0] - 0.4 * X[:, 1])))
    b = E.fit_logit(X, y)
    assert b == pytest.approx([-1.0, 0.8, -0.4], abs=0.06)
    p = E.predict_logit(b, X)
    assert abs(p.mean() - y.mean()) < 0.01
    assert E.logloss_each(y, p).mean() < E.logloss_each(y, np.full(len(y), y.mean())).mean()


def test_loyo_predict_is_out_of_sample():
    rng = np.random.default_rng(0)
    n = 3000
    years = rng.integers(2010, 2016, n)
    X = rng.normal(size=(n, 1))
    y = rng.random(n) < 1 / (1 + np.exp(-(X[:, 0])))
    p = E.loyo_predict(X, y, years)
    assert np.isfinite(p).all() and E.auc_score(p, y) > 0.65
    noise = rng.normal(size=(n, 1))
    p0 = E.loyo_predict(noise, y, years)
    assert abs(E.auc_score(p0, y) - 0.5) < 0.05


def test_cluster_wald_detects_signal_and_not_noise():
    rng = np.random.default_rng(1)
    n = 8000
    years = rng.integers(2010, 2026, n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.5 * x1)))
    r = E.cluster_wald(np.column_stack([x1, x2]), y, years, cols=[0])
    assert r["coef"][0] == pytest.approx(0.5, abs=0.1) and r["p"] < 1e-3
    r = E.cluster_wald(np.column_stack([x1, x2]), y, years, cols=[1])
    assert r["p"] > 0.01 and abs(r["coef"][0]) < 0.1
    both = E.cluster_wald(np.column_stack([x1, x2]), y, years, cols=[0, 1])
    assert len(both["coef"]) == 2 and both["p"] < 1e-3


# --- cells, groups -----------------------------------------------------------------------
def _fungi():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-09-01"] * 4 + ["2020-09-02", "2021-10-01"]),
        "lon": [14.50, 14.51, 14.50, 14.52, 14.50, 15.60],
        "lat": [46.05, 46.05, 46.06, 46.05, 46.05, 46.50],
        "is_jurcki": [True, False, False, False, False, True],
        "class": pd.Categorical(["Agaricomycetes", "Agaricomycetes", "Lecanoromycetes", "Agaricomycetes",
                                 "Agaricomycetes", "Agaricomycetes"]),
        "family": pd.Categorical(["Boletaceae", "Russulaceae", "Parmeliaceae", "Agaricaceae", "Boletaceae",
                                  "Boletaceae"]),
        "order": pd.Categorical(["Boletales", "Russulales", "Lecanorales", "Agaricales", "Boletales", "Boletales"]),
        "datasetkey": [E.INAT, E.INAT, E.OBSORG, "x", E.INAT, "x"],
        "countrycode": pd.Categorical(["SI"] * 6),
    })


def test_target_masks():
    d = _fungi()
    assert E.target_mask(d, "fungi").all()
    assert E.target_mask(d, "macro").tolist() == [True, True, False, True, True, True]
    assert E.target_mask(d, "myco").tolist() == [True, True, False, False, True, True]
    assert E.target_mask(d, "bolete").tolist() == [True, False, False, False, True, True]
    with pytest.raises(ValueError):
        E.target_mask(d, "nope")


def test_cell_day_table_counts_and_label():
    d = _fungi()
    d["source"] = np.where(d["datasetkey"] == E.INAT, "inat", "other")
    cid = E.assign_cells(d["lon"], d["lat"], 25)
    assert (cid >= 0).all() and cid[0] == cid[1] == cid[2] == cid[3] and cid[0] != cid[5]
    cd = E.cell_day_table(d, 25)
    day1 = cd[(cd["date"] == "2020-09-01") & (cd["cell"] == cid[0])].iloc[0]
    assert day1["n"] == 4 and day1["k"] == 1 and bool(day1["label"]) and day1["n_inat"] == 2
    day2 = cd[cd["date"] == "2020-09-02"].iloc[0]
    assert day2["n"] == 1 and not day2["label"]
    assert len(cd) == 3
    assert E.assign_cells([0.0], [0.0])[0] == -1       # outside the grid


def test_cellday_scores_and_evaluation_on_synthetic_signal():
    """Labels drawn from the trigger itself must give AUC > 0.5, season-only less than the trigger."""
    from jurcki.model import prior
    rng = np.random.default_rng(0)
    T, N = 400, 6
    times = pd.date_range("2018-01-01", periods=T)
    w = _weather(T, N, seed=3)
    w["tmean"] += 5
    w["tmin"], w["tmax"] = w["tmean"] - 4, w["tmean"] + 5
    cells = pd.DataFrame({"lon": 14.5, "lat": 46.0, "elev": 500.0}, index=np.arange(N) + 100)
    ti = rng.integers(130, 330, 4000)
    ci = rng.integers(0, N, 4000)
    p = prior.PriorParams()
    trig = features.dynamic_feature_arrays(w, p)["trigger"]
    lab = rng.random(4000) < 0.05 + 0.9 * trig[ti, ci]
    cd = pd.DataFrame({"cell": cells.index[ci], "date": times[ti], "n": 1 + rng.integers(0, 3, 4000), "label": lab})
    cd["year"] = cd["date"].dt.year
    sc = E.cellday_scores(cd, cells, times, w, {"literature": p, "tuned": p}, {2018: p, 2019: p})
    assert set(sc.columns) >= {"season", "literature", "tuned", "tuned_loyo", "trig_literature"}
    assert np.allclose(sc["literature"], sc["tuned"]) and sc["tuned_loyo"].notna().all()
    ev = E.evaluate_scores(cd, sc[["season", "trig_literature", "literature"]], n_boot=30)
    assert ev.loc["trig_literature", "auc"] > 0.65
    assert ev.loc["trig_literature", "auc"] > ev.loc["season", "auc"]
    assert ev.loc["trig_literature", "lo"] <= ev.loc["trig_literature", "auc"] <= ev.loc["trig_literature", "hi"]


# --- elevation, moon, years, scan --------------------------------------------------------
def test_elevation_percentile_and_table():
    rng = np.random.default_rng(0)
    n = 4000
    d = pd.DataFrame({"date": pd.Timestamp("2020-06-01") + pd.to_timedelta(rng.integers(0, 150, n), unit="D"),
                      "z": rng.uniform(200, 1200, n)})
    d["year"] = d["date"].dt.year + rng.integers(0, 6, n)
    d["is_jurcki"] = False
    j = rng.random(n) < 0.05
    d.loc[j, "z"] = rng.uniform(900, 1200, j.sum())            # jurcki sit high
    d["is_jurcki"] = j
    pct = E.elevation_percentile(d["z"], d["date"].dt.month, d["is_jurcki"])
    assert len(pct) == j.sum() and pct.mean() > 0.7
    tab = E.elevation_by_month(d, n_boot=100)
    assert (tab["pct_mean"] > 0.6).all() and (tab["pct_lo"] <= tab["pct_mean"]).all()
    assert (tab["median_jurcki"] > tab["median_all"]).all()
    # records without elevation are ignored
    d.loc[:50, "z"] = np.nan
    assert len(E.elevation_by_month(d, n_boot=20)) > 0


def test_moon_table_bins_cover_all_records():
    dates = pd.date_range("2020-01-01", periods=400)
    t = E.moon_table(dates, np.zeros(400, bool))
    assert t["n"].sum() == 400 and len(t) == 8 and (t["k"] == 0).all()
    # records only at full moon fall in the middle bin
    full = pd.to_datetime(["2024-04-23", "2024-05-23", "2024-06-22"])
    t = E.moon_table(full, np.ones(3, bool))
    assert t["k"].idxmax() == 4


def test_year_level_table():
    years = np.arange(2010, 2027)
    oe = pd.DataFrame({"ratio": np.linspace(0.5, 2.0, len(years))}, index=years)
    x = pd.Series(np.linspace(0, 1, len(years)), index=years)
    t = E.year_level_table(oe, {"x": x}, max_year=2025, n_boot=200)
    assert t.loc["x", "rho"] == pytest.approx(1.0) and t.loc["x", "n"] == 16
    assert t.loc["calendar year", "rho"] == pytest.approx(1.0)
    assert t.loc["previous year's ratio", "n"] == 15 or t.loc["previous year's ratio", "n"] == 16


def test_yearly_anomaly_removes_the_weekly_mean():
    dates = pd.to_datetime(["2020-08-03", "2020-08-04", "2021-08-03", "2021-08-04", "2020-10-05", "2021-10-05"])
    cd = pd.DataFrame({"date": dates, "year": dates.year, "n": [1, 1, 1, 1, 1, 1]})
    score = np.array([0.2, 0.2, 0.6, 0.6, 0.9, 0.9])   # summer is higher in 2021, October is the same
    a = E.yearly_anomaly(cd, score)
    assert a[2020] < 0 < a[2021] and a.sum() == pytest.approx(0.0, abs=1e-12)
    # a season effect alone (October always higher than August) gives no yearly anomaly
    flat = E.yearly_anomaly(cd.iloc[[0, 1, 4, 2, 3, 5]], np.array([0.2, 0.2, 0.9, 0.2, 0.2, 0.9]))
    assert abs(flat[2020] - flat[2021]) < 1e-12


def test_to_markdown():
    t = E.to_markdown(pd.DataFrame({"a": [1.23456, np.nan], "b": ["x", "y"]}, index=pd.Index([1, 2], name="i")))
    lines = t.splitlines()
    assert lines[0] == "| i | a | b |" and lines[2] == "| 1 | 1.235 | x |" and lines[3] == "| 2 |  | y |"


def _scan_data(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    years = rng.integers(2010, 2022, n)
    base = rng.normal(size=(n, 2))
    signal = rng.normal(size=(n, 1))
    noise = rng.normal(size=(n, 1))
    y = rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.4 * base[:, 0] + 0.5 * signal[:, 0])))
    return base, {"signal": signal, "noise": noise, "signal_sq": np.column_stack([signal[:, 0], signal[:, 0] ** 2])}, y, years


def test_scan_candidates_finds_signal_and_not_noise():
    base, cands, y, years = _scan_data()
    r = E.scan_candidates(base, cands, y, years, n_boot=200)
    assert set(r.index) == {"signal", "noise", "signal_sq"}
    assert r.loc["signal", "gain"] > 0.005 and r.loc["signal", "gain_lo"] > 0 and r.loc["signal", "rep"]
    assert r.loc["signal", "p_wald"] < 1e-3 and r.loc["signal", "or_lo"] > 1
    assert r.loc["noise", "p_wald"] > 0.01 and r.loc["noise", "gain"] < 0.001
    assert r.loc["signal_sq", "k"] == 2
    assert (r["p_holm"] >= r["p_wald"]).all() and (r["q_bh"] >= r["p_wald"] - 1e-12).all()


def test_effect_span_single_column():
    x = np.random.default_rng(0).normal(size=(5000, 1))
    orr, lo, hi = E.effect_span(x, np.array([0.5]), np.array([[0.01]]))
    q = np.quantile(x[:, 0], [0.1, 0.9])
    assert orr == pytest.approx(np.exp(0.5 * (x[x[:, 0] >= q[1], 0].mean() - x[x[:, 0] <= q[0], 0].mean())), rel=1e-6)
    assert lo < orr < hi


def test_pseudo_moon_null_is_uniform_without_a_lunar_effect():
    rng = np.random.default_rng(5)
    n = 3000
    dates = pd.Timestamp("2018-05-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    y = rng.random(n) < 0.1
    base = rng.normal(size=(n, 1))
    ph = E.moon_phase(dates)
    X0 = E.standardize(base)
    obs = E.deviance_gain(y, X0, np.column_stack([X0, np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)]))
    r = E.pseudo_moon_null(base, dates, y, dates.year, obs, n_null=60)
    assert r["p"] > 0.05 and r["null_p95"] >= r["null_median"] >= 0
    # a real lunar effect is detected
    y2 = rng.random(n) < 0.04 + 0.12 * (np.cos(2 * np.pi * ph) > 0.5)
    obs2 = E.deviance_gain(y2, X0, np.column_stack([X0, np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)]))
    assert E.pseudo_moon_null(base, dates, y2, dates.year, obs2, n_null=60)["p"] < 0.05


def test_design_matrices_shapes_and_controls():
    rng = np.random.default_rng(0)
    T, N = 400, 5
    times = pd.date_range("2018-01-01", periods=T)
    w = _weather(T, N)
    cells = pd.DataFrame({"lon": 14.5, "lat": 46.0, "elev": np.linspace(300, 1300, N),
                          "country": ["SI", "AT", "SI", "HR", "AT"]}, index=np.arange(N) + 7)
    m = 600
    ti, ci = rng.integers(130, 330, m), rng.integers(0, N, m)
    cd = pd.DataFrame({"cell": cells.index[ci], "date": times[ti], "n": 1 + rng.integers(0, 4, m)})
    cd["n_inat"] = rng.integers(0, 2, m) * cd["n"]
    cd["src"] = np.where(cd["n_inat"] == cd["n"], "inat", np.where(cd["n_inat"] == 0, "other", "mixed"))
    cd["year"] = cd["date"].dt.year
    score = rng.random(m)
    base, c = E.design_matrices(cd, cells, times, w, score, {2018: 1.2}, noise=4)
    assert base.shape == (m, 12)
    assert base[:, 0] == pytest.approx(score)
    base0, _ = E.design_matrices(cd, cells, times, w, score, None, noise=2, with_trigger=False)
    assert base0.shape[1] == base.shape[1] - 1
    for name in ("moon_phase", "elev", "stl1_7"):
        assert c[name].shape == (m, 2)
    for name in ("drop5", "rain_days26", "weekend", "frac_inat", "elev_x_doy", "jj_precip_anom", "prev_year_oe"):
        assert c[name].shape == (m, 1) and np.isfinite(c[name]).all()
    assert sum(k.startswith("noise_date") for k in c) == 4 and sum(k.startswith("noise_cellyear") for k in c) == 3
    # the date-level controls are identical for the same date
    nd = c["noise_date_0"][:, 0]
    for dte, g in cd.groupby("date").groups.items():
        assert len(set(nd[cd.index.get_indexer(g)])) == 1
    assert set(E.CANDIDATES) <= set(c)


def test_yearwise_stratified_auc_and_nested_choice():
    rng = np.random.default_rng(0)
    n = 3000
    year = rng.integers(2010, 2016, n)
    strata = rng.integers(0, 3, n)
    good = rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-(good - 2.0)))
    scores = np.stack([good, rng.normal(size=n), -good])      # candidate 0 informative, 2 anti-informative
    ys, num, den = E.yearwise_stratified_auc(good, y, strata, year)
    assert list(ys) == list(range(2010, 2016)) and (num / den).min() > 0.6
    r = E.nested_loyo_choice(scores, y, strata, year)
    assert set(r["chosen"].values()) == {0} and r["heldout_auc"] > 0.6
    assert r["in_sample_best"] == 0 and r["per_candidate_auc"][2] < 0.4
    assert r["train_auc"] > 0.6


def test_candidate_scores_shape_and_literature_row():
    from jurcki.model import prior
    T, N = 300, 3
    times = pd.date_range("2019-01-01", periods=T)
    w = _weather(T, N)
    cells = pd.DataFrame({"lon": 14.5, "lat": 46.0, "elev": 500.0}, index=[5, 6, 7])
    cd = pd.DataFrame({"cell": [5, 7, 6], "date": times[[150, 200, 250]]})
    pa, pb = prior.PriorParams(), prior.PriorParams(t_opt=18.0)
    S = E.candidate_scores(cd, cells, times, w, [pa, pb])
    assert S.shape == (2, 3) and S.dtype == np.float32 and not np.allclose(S[0], S[1])
    ref = features.dynamic_feature_arrays(w, pa)["trigger"]
    assert S[0, 1] == pytest.approx(ref[200, 2], abs=1e-6)


def _write_fungi(path):
    rng = np.random.default_rng(0)
    rows = []
    def add(ds, n, basis, cls, n_rec, ctry="SI", jur=0):
        for i in range(n):
            rows.append(dict(gbifid=len(rows), species="Boletus edulis" if i < jur else "Foo bar", genus="G", family="F",
                             order="O", **{"class": cls}, lat=46.0, lon=14.5, coordinateuncertaintyinmeters=10.0,
                             date=pd.Timestamp("2021-09-01") + pd.Timedelta(days=int(rng.integers(0, 40))),
                             datasetkey=ds, basisofrecord=basis, countrycode=ctry, recorder_id=f"r{i % n_rec}",
                             is_jurcki=i < jur))
    add("crowd", 100, "HUMAN_OBSERVATION", "Agaricomycetes", 40, jur=5)
    add("small_society", 50, "HUMAN_OBSERVATION", "Agaricomycetes", 5)
    add("survey", 200, "OBSERVATION", "Sordariomycetes", 50)
    add("samples", 30, "MATERIAL_SAMPLE", "Agaricomycetes", 40)
    add("lichens", 30, "HUMAN_OBSERVATION", "Lecanoromycetes", 40)
    pd.DataFrame(rows).to_parquet(path)


def test_load_fungi_filters_to_opportunistic_macrofungi(tmp_path):
    p = tmp_path / "f.parquet"
    _write_fungi(p)
    d = E.load_fungi(p)
    assert set(d["datasetkey"].astype(str)) == {"crowd"} and len(d) == 100 and d["is_jurcki"].sum() == 5
    assert {"year", "doy", "week", "source"} <= set(d.columns)
    # recorder minimum off: the small society comes back, the survey and the lichens still do not
    assert set(E.load_fungi(p, min_recorders=0)["datasetkey"].astype(str)) == {"crowd", "small_society"}
    # basis and group switched off: everything is there
    assert len(E.load_fungi(p, group="fungi", basis=None, min_recorders=0)) == 410
    assert set(E.load_fungi(p, drop_datasets=("crowd",), min_recorders=0)["datasetkey"].astype(str)) == {"small_society"}
    assert len(E.load_fungi(p, countries=("AT",))) == 0


def test_baseline_by_year_shows_each_filter(tmp_path):
    p = tmp_path / "f.parquet"
    _write_fungi(p)
    t = E.baseline_by_year(p)
    assert t.loc[2021, "all_fungi"] == 410 and t.loc[2021, "human_obs"] == 180
    assert t.loc[2021, "agaricomycetes_human_obs"] == 150 and t.loc[2021, "baseline"] == 100
    assert t.loc[2021, "baseline_SI"] == 100 and t.loc[2021, "jurcki_in_baseline"] == 5


def test_elev_season_flag_in_trigger_is_off_by_default_and_has_the_right_shape():
    from jurcki.model import trigger
    f = trigger.elev_season_factor
    spec = trigger.ELEV_SEASON
    assert f(spec["elev_mean"], spec["doy_mean"]) == pytest.approx(1.0)
    assert f(1200, 280) > 1 > f(1200, 160)        # high places: better late than early
    assert f(250, 160) > 1 > f(250, 280)          # low places: better early than late
    assert f(5000, 360) == spec["clip"][1] and f(-5000, 360) == spec["clip"][0]
    # panel: default unchanged, flag only rescales
    from tests.test_trigger import make_panel
    panel = make_panel()
    p = __import__("jurcki.model.prior", fromlist=["PriorParams"]).PriorParams()
    base = trigger.trigger_scores(panel, p)
    flagged = trigger.trigger_scores(panel, p, elev_season=True)
    assert base.shape == flagged.shape and not np.allclose(base, flagged)
    ratio = flagged[base > 0] / base[base > 0]
    assert ratio.min() >= spec["clip"][0] - 1e-9 and ratio.max() <= spec["clip"][1] + 1e-9


def test_elev_season_heldout_detects_a_planted_interaction():
    from jurcki.model import trigger
    spec = trigger.ELEV_SEASON
    rng = np.random.default_rng(0)
    n = 12000
    cells = pd.DataFrame({"lon": 14.5, "lat": 46.0, "elev": rng.uniform(200, 1300, 40), "country": rng.choice(["SI", "AT"], 40)},
                         index=np.arange(40))
    cd = pd.DataFrame({"cell": rng.integers(0, 40, n), "date": pd.Timestamp("2018-05-01") + pd.to_timedelta(rng.integers(0, 190, n), unit="D")})
    cd["year"] = cd["date"].dt.year + rng.integers(0, 6, n)
    cd["n"] = 1
    elev = cells["elev"].to_numpy()[cd["cell"]]
    doy = cd["date"].dt.day_of_year.to_numpy()
    prod = ((elev - spec["elev_mean"]) / spec["elev_sd"]) * ((doy - spec["doy_mean"]) / spec["doy_sd"])
    cd["label"] = rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.8 * prod)))
    score = rng.random(n)
    base = np.column_stack([score, rng.normal(size=n)])
    strata = cd["date"].dt.month.to_numpy()
    r = E.elev_season_heldout(cd, cells, score, base, strata, n_boot=60)
    assert r["coef_all_years"] == pytest.approx(0.8, abs=0.15) and r["with"]["pooled"] > r["without"]["pooled"] + 0.05
    assert r["paired_pooled"][1] > 0 and r["leave_country_out_logloss"]["with factor"] < r["leave_country_out_logloss"]["baseline"]
    assert set(r["auc_by_country"]) == {"AT", "SI"}
