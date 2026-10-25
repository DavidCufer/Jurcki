"""Offline tests for jurcki.model.backtest_reports (synthetic reports, weather and grids; no post text, no network)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from jurcki import config, grid, predict as P
from jurcki.model import backtest_reports as B
from jurcki.model import prior

LAT = np.array([45.5, 45.75, 46.0, 46.25])
LON = np.array([14.0, 14.25, 14.5, 14.75])


# ---------------------------------------------------------------- reports
def _reports(rows):
    base = dict(source="gobe_forum", report_kind="outing", date_uncertainty_days=0, loc_lon=np.nan, loc_lat=np.nan,
                loc_radius_m=np.nan, loc_ellipse_a_m=np.nan, loc_ellipse_b_m=np.nan, loc_ellipse_az_deg=np.nan,
                loc_name=None, region="")
    return pd.DataFrame([{**base, **r} for r in rows])


def test_load_reports_filters_and_maps(tmp_path):
    df = _reports([
        dict(post_id="a", date_observed="2009-09-01", jurcki_outcome="found"),      # before ERA5
        dict(post_id="b", date_observed="2015-09-01", jurcki_outcome="many"),
        dict(post_id="c", date_observed="2015-09-02", jurcki_outcome="none"),
        dict(post_id="d", date_observed="2015-09-03", jurcki_outcome="unclear"),    # dropped
        dict(post_id="e", date_observed="2015-09-03", jurcki_outcome="not_mentioned"),
    ])
    df["date_observed"] = pd.to_datetime(df["date_observed"])
    p = tmp_path / "r.parquet"
    df.to_parquet(p)
    posts = pd.DataFrame({"id": ["b", "c"], "author_hash": ["x1", "x1"]})
    posts.to_parquet(tmp_path / "posts.parquet")
    d = B.load_reports(p, posts_path=tmp_path / "posts.parquet")
    assert list(d["post_id"]) == ["b", "c"] and list(d["y"]) == [1, 0]
    assert d.attrs["n_before"] == 1
    assert list(d["author"]) == ["x1", "x1"] and d["yw"].iat[0] == 201536
    d2 = B.load_reports(p, posts_path=tmp_path / "missing.parquet")
    assert (d2["author"] == "").all()


def test_dedupe_cross_posts():
    d = _reports([
        dict(post_id="1", date_observed="2015-09-01", jurcki_outcome="found", author="a", loc_name="Pohorje"),
        dict(post_id="2", date_observed="2015-09-02", jurcki_outcome="found", author="a", loc_name="Pohorje"),     # dup (+1 d)
        dict(post_id="3", date_observed="2015-09-02", jurcki_outcome="none", author="a", loc_name="Pohorje"),      # other outcome
        dict(post_id="4", date_observed="2015-09-02", jurcki_outcome="found", author="b", loc_name="Pohorje"),     # other author
        dict(post_id="5", date_observed="2015-09-05", jurcki_outcome="found", author="a", loc_name="Pohorje"),     # 4 days later
        dict(post_id="6", date_observed="2015-09-01", jurcki_outcome="found", author="a", loc_name="Snežnik"),     # other place
        dict(post_id="7", date_observed="2015-09-01", jurcki_outcome="found", author="a", loc_name=None),          # unlocated, same author
        dict(post_id="8", date_observed="2015-09-01", jurcki_outcome="none", author="", loc_name=None),
        dict(post_id="9", date_observed="2015-09-01", jurcki_outcome="none", author="", loc_name=None),            # cannot say
    ])
    d["date"] = pd.to_datetime(d["date_observed"])
    out = B.dedupe(d)
    assert set(out["post_id"]) == {"1", "3", "4", "5", "6", "8", "9"}
    assert out.attrs["n_duplicates"] == 2


def test_dedupe_without_author_uses_place_name():
    d = _reports([
        dict(post_id="1", date_observed="2015-09-01", jurcki_outcome="none", loc_name="Rogla"),
        dict(post_id="2", date_observed="2015-09-02", jurcki_outcome="none", loc_name="Rogla"),
        dict(post_id="3", date_observed="2015-09-02", jurcki_outcome="none", loc_name="Areh"),
    ])
    d["author"] = ""
    d["date"] = pd.to_datetime(d["date_observed"])
    assert set(B.dedupe(d)["post_id"]) == {"1", "3"}


def test_author_week_weights():
    d = pd.DataFrame({"author": ["a", "a", "b", ""], "yw": [201536, 201536, 201536, 201536], "post_id": list("wxyz")})
    np.testing.assert_allclose(B.author_week_weights(d), [0.5, 0.5, 1.0, 1.0])


# ---------------------------------------------------------------- synthetic grid / forest
def _forest(rows=40, cols=60, hab=0.8):
    h = np.full((rows, cols), hab, "float32")
    h[:, :5] = 0.0                                 # a non-forest strip on the west
    mask = np.ones((rows, cols), bool)
    x = config.GRID_ORIGIN[0] + (np.arange(cols) + 0.5) * config.GRID_RES
    y = config.GRID_ORIGIN[1] - (np.arange(rows) + 0.5) * config.GRID_RES
    X, Y = np.meshgrid(x, y)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(config.CRS, config.CRS_WGS84, always_xy=True).transform(X, Y)
    elev = np.full((rows, cols), 600.0, "float32")
    return B.Forest(habitat=h, mask=mask, elev=elev, lonlat=(lon, lat)), (lon, lat)


def test_forest_and_ellipse_members():
    f, (lon, lat) = _forest()
    assert len(f.flat) == 40 * 55                        # habitat <= 0.05 strip removed
    r, c = 20, 30
    n_circle = len(f.ellipse_members(lon[r, c], lat[r, c], 2000))
    assert abs(n_circle - np.pi * 2000 ** 2 / config.GRID_RES ** 2) < 0.1 * n_circle
    # an ellipse with the major axis east-west (azimuth 90) is wider than tall
    m = f.ellipse_members(lon[r, c], lat[r, c], 4000, 1000, 90)
    assert np.ptp(f.x[m]) > 2.5 * np.ptp(f.y[m])
    m2 = f.ellipse_members(lon[r, c], lat[r, c], 4000, 1000, 0)
    assert np.ptp(f.y[m2]) > 2.5 * np.ptp(f.x[m2])
    assert len(f.ellipse_members(lon[r, c] + 5, lat[r, c], 2000)) == 0


def test_sample_units_unlocated_and_located():
    f, (lon, lat) = _forest()
    d = _reports([dict(post_id="a", loc_lon=lon[20, 30], loc_lat=lat[20, 30], loc_radius_m=1500.0),
                  dict(post_id="b"),
                  dict(post_id="c", loc_lon=lon[20, 30] + 5, loc_lat=lat[20, 30], loc_radius_m=1500.0)])   # outside the grid
    cells, ru, rg, nu, located = B.sample_units(f, d, k=20, k_region=30, k_national=50)
    assert list(located) == [True, False, False]
    assert ru.pos.shape[0] == 1 and ru.valid.sum() == 20
    assert nu.valid.sum() == 50 and len(cells) >= 50
    assert (cells.hab > B.HAB_MIN).all()
    # unit cells are inside the ellipse
    got = cells.flat[ru.pos[0][ru.valid[0]]]
    assert set(got) <= set(f.flat[f.ellipse_members(lon[20, 30], lat[20, 30], 1500)])


def test_report_ellipses_prefers_ellipse_else_circle():
    d = _reports([dict(post_id="a", loc_radius_m=3000.0), dict(post_id="b", loc_radius_m=3000.0, loc_ellipse_a_m=5000.0,
                                                               loc_ellipse_b_m=2000.0, loc_ellipse_az_deg=45.0)])
    e = B.report_ellipses(d)
    assert list(e["a_m"]) == [3000.0, 5000.0] and list(e["b_m"]) == [3000.0, 2000.0] and list(e["az"]) == [0.0, 45.0]


# ---------------------------------------------------------------- engine vs predict
def _dataset(first="2020-06-01", last="2020-10-15", seed=0):
    t = pd.date_range(first, last, freq="D")
    rng = np.random.default_rng(seed)
    shape = (len(t), len(LAT), len(LON))
    tmean = 12 + 8 * np.sin(np.linspace(0, np.pi, len(t)))[:, None, None] + rng.normal(0, 1.5, shape)
    precip = np.where(rng.random(shape) < 0.25, rng.gamma(2.0, 6.0, shape), 0.0)
    ds = xr.Dataset({"tmean": (("time", "lat", "lon"), tmean), "tmin": (("time", "lat", "lon"), tmean - 6),
                     "tmax": (("time", "lat", "lon"), tmean + 6), "precip": (("time", "lat", "lon"), precip),
                     "swvl1": (("time", "lat", "lon"), np.full(shape, 0.3)), "stl1": (("time", "lat", "lon"), tmean),
                     "dewpoint": (("time", "lat", "lon"), np.full(shape, 5.0)), "rh": (("time", "lat", "lon"), np.full(shape, 70.0)),
                     "snow_depth": (("time", "lat", "lon"), np.zeros(shape))},
                    coords={"time": t, "lat": LAT, "lon": LON})
    ds["orography"] = (("lat", "lon"), np.full((len(LAT), len(LON)), 400.0))
    return ds.astype("float32")


class _Fake:
    name = "fake"

    def __init__(self, ds):
        self.ds = ds

    def load(self, a, b):
        return self.ds.sel(time=slice(pd.Timestamp(a), pd.Timestamp(b)))


def _engine_and_cells():
    n = 40
    rng = np.random.default_rng(1)
    cells = B.CellTable(np.arange(n), rng.uniform(14.1, 14.6, n), rng.uniform(45.6, 46.1, n),
                        rng.uniform(300, 1000, n).astype("float32"), rng.uniform(0.2, 1.0, n).astype("float32"))
    ds = _dataset()
    return B.Engine(P.prepare_weather(ds), cells), cells, ds


def test_engine_trigger_matches_predict():
    eng, cells, ds = _engine_and_cells()
    p = prior.PriorParams(t_opt=15.0, t_sd=6.0, p_half=90.0)
    day = pd.Timestamp("2020-09-10")
    fg = P.FineGrid(cells.lon, cells.lat, cells.elev)
    ref = list(P.iter_trigger([day.date()], p, _Fake(ds), fg))[0][1]
    pos = np.arange(10, 25)[None, :]
    tq = eng.day_index(np.array([[day.to_datetime64()]]))
    got = eng.trigger(eng.coarse_features(p), p, tq, pos)[0, 0]
    np.testing.assert_allclose(got, ref[10:25], atol=1e-5)


def test_evaluate_tolerance_season_and_hab():
    eng, cells, _ = _engine_and_cells()
    units = B.UnitCells(np.tile(np.arange(10), (3, 1)), np.ones((3, 10), bool))
    days = pd.DatetimeIndex(["2020-09-10", "2020-09-30", "2020-10-01"])
    params = {"t": prior.PriorParams()}
    out = eng.evaluate(days, units, params, ["season", "hab", "hab_season", "trig_t", "comb_t", "full_t"])
    np.testing.assert_allclose(out[("season", 0)], [1.0, 1.0, 0.9])
    # max over +-3 days of the season (Sep 30 + 3 d = Oct 3: 0.9 vs Sep 1.0) keeps the larger
    np.testing.assert_allclose(out[("season", 3)], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out[("hab", 0)], cells.hab[:10].mean())
    np.testing.assert_allclose(out[("hab_season", 0)], out[("season", 0)] * cells.hab[:10].mean(), rtol=1e-6)
    for k in ("trig_t", "comb_t", "full_t"):
        assert (out[(k, 3)] >= out[(k, 1)] - 1e-7).all() and (out[(k, 1)] >= out[(k, 0)] - 1e-7).all()
    assert (out[("full_t", 0)] <= out[("comb_t", 0)] + 1e-7).all()


def test_evaluate_multiplier_and_padding():
    eng, cells, _ = _engine_and_cells()
    pos = np.array([[0, 1, 2, 0], [3, 4, 0, 0]])
    valid = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], bool)
    units = B.UnitCells(pos, valid)
    days = pd.DatetimeIndex(["2020-09-10", "2020-09-10"])
    mult = {"two": lambda elev, doy: np.full(np.broadcast_shapes(elev.shape, doy.shape), 2.0)}
    out = eng.evaluate(days, units, {"t": prior.PriorParams()}, ["comb_t", "comb_t@two", "hab"], mults=mult, offsets=(0,), tols=(0,))
    np.testing.assert_allclose(out[("comb_t@two", 0)], 2 * out[("comb_t", 0)], rtol=1e-6)
    np.testing.assert_allclose(out[("hab", 0)], [cells.hab[[0, 1, 2]].mean(), cells.hab[[3, 4]].mean()], rtol=1e-6)


def test_eval_pairs_dedupes_and_aligns():
    eng, cells, _ = _engine_and_cells()
    units = B.UnitCells(np.array([[0, 1], [2, 3]]), np.ones((2, 2), bool))
    days = pd.DatetimeIndex(["2020-09-10", "2020-09-10", "2020-09-10", "2020-09-12"])
    ids = np.array([0, 0, 1, 1])
    plan = B.PairPlan(days, ids, units)
    assert len(plan.days) == 3
    out = plan.run(eng, {"t": prior.PriorParams()}, ["comb_t"], offsets=(0,), tols=(0,))[("comb_t", 0)]
    assert out[0] == out[1] and out[0] != out[2]


def test_subsample_units():
    u = B.UnitCells(np.arange(20)[None, :], np.ones((1, 20), bool))
    s = B.subsample_units(u, 5)
    assert s.valid.sum() == 5 and len(set(s.pos[0][s.valid[0]])) == 5
    s = B.subsample_units(u, 50)
    assert s.valid.sum() == 20


# ---------------------------------------------------------------- statistics
def test_auc_and_stratified_auc_match_brute_force():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=120), 1)
    y = rng.random(120) < 0.4
    g = rng.integers(0, 5, 120)

    def brute(idx):
        num = den = 0.0
        for i in idx:
            for j in idx:
                if y[i] and not y[j]:
                    den += 1
                    num += (s[i] > s[j]) + 0.5 * (s[i] == s[j])
        return num / den

    assert B.auc(s, y) == pytest.approx(brute(range(120)))
    num = den = 0.0
    for k in range(5):
        idx = np.flatnonzero(g == k)
        n1, n0 = y[idx].sum(), (~y[idx]).sum()
        num += brute(idx) * n1 * n0
        den += n1 * n0
    assert B.stratified_auc(s, y, g) == pytest.approx(num / den)
    parts = B.stratum_parts(s, y, g)
    assert parts["u"].sum() / parts["p"].sum() == pytest.approx(B.stratified_auc(s, y, g))


def test_rank_auc_rows_matches_auc():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(6, 80))
    y = rng.random(80) < 0.5
    cols = np.arange(80) % 3 != 0
    got = B.rank_auc_rows(M, y, cols)
    want = [B.auc(M[i, cols], y[cols]) for i in range(6)]
    np.testing.assert_allclose(got, want)


def test_cluster_bootstrap_and_pooled_table():
    rng = np.random.default_rng(2)
    n = 300
    y = rng.random(n) < 0.5
    good = y + rng.normal(0, 0.8, n)
    bad = rng.normal(size=n)
    clusters = rng.integers(0, 30, n)
    S = pd.DataFrame({"good": good, "bad": bad})
    t = B.pooled_table(S, y, {"week": clusters}, n_boot=200)
    assert t.loc["good", "auc"] > 0.7 > t.loc["bad", "auc"] - 0.1
    assert t.loc["good", "lo_week"] < t.loc["good", "auc"] < t.loc["good", "hi_week"]
    assert t.loc["good", "n_pos"] == y.sum()
    d = B.paired_diffs(S, y, [("good", "bad")], clusters, n_boot=200)
    assert d["lo"].iat[0] > 0
    d2 = B.paired_diffs(S, y, [("good", "bad")], clusters, n_boot=100, strata=clusters % 3)
    assert d2["diff"].iat[0] > 0


def test_calibration_monotone_for_informative_score():
    rng = np.random.default_rng(3)
    s = rng.random(2000)
    y = rng.random(2000) < s
    rep = pd.DataFrame({"y": y.astype(int)})
    g = B.calibration(rep, s)
    assert len(g) == 10 and g["n"].sum() == 2000
    assert g["share_pos"].iat[-1] > g["share_pos"].iat[0] + 0.5
    assert ((g["lo"] <= g["share_pos"]) & (g["share_pos"] <= g["hi"])).all()


def test_year_table_within_table_dose_and_quality():
    rng = np.random.default_rng(4)
    n = 600
    rep = pd.DataFrame({"year": rng.choice([2013, 2014, 2015], n), "date": pd.Timestamp("2015-08-01") + pd.to_timedelta(rng.integers(0, 80, n), unit="D")})
    rep["yw"] = rep["date"].dt.isocalendar().week.to_numpy() + 100 * rep["year"]
    score = rng.random(n)
    rep["y"] = (rng.random(n) < 0.2 + 0.6 * score).astype(int)
    rep["jurcki_outcome"] = np.where(rep["y"] == 0, "none", rng.choice(["few", "found", "many"], n))
    rep["region"] = rng.choice(["A", "B"], n)
    S = pd.DataFrame({"score": score, "noise": rng.random(n)}, index=rep.index)
    yt = B.year_table(rep, S)
    assert list(yt.index) == [2013, 2014, 2015] and (yt["score"] > yt["noise"]).all()
    w = B.within_table(rep, S, n_boot=50)
    assert set(w["kind"]) == {"week", "year", "region", "region_year"}
    a = w[(w.score == "score") & (w.kind == "year")].iloc[0]
    assert a["lo"] <= a["auc"] <= a["hi"] and a["auc"] > 0.6
    dr = B.dose_response(rep, S)
    assert set(dr.index) == {"few", "found", "many"}
    yq = B.year_quality(rep, S, ["score"], min_n=10, n_boot=50)
    assert yq["n_years"] == 3 and "score" in yq["spearman"]


def test_sensitivity_runs_on_scoreset_like_object():
    rng = np.random.default_rng(5)
    n = 200
    d = pd.DataFrame({"y": rng.integers(0, 2, n), "year": rng.choice([2014, 2015, 2016], n), "report_kind": "outing",
                      "date_uncertainty_days": 0, "author": rng.choice(["a", "b", "c", "d", "e", "f", "g"], n),
                      "yw": 201500 + rng.integers(30, 40, n), "post_id": [str(i) for i in range(n)],
                      "date": pd.Timestamp("2015-09-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
                      "region": rng.choice(["Podravska", "Gorenjska"], n)})
    cols = pd.MultiIndex.from_product([["season", "comb_lit", "comb_tuned", "full_tuned"], [0, 1, 3]], names=["score", "tol"])
    nat = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
    ss = B.ScoreSet(d, np.ones(n, bool), nat, nat.copy(), nat.copy(), {}, None, None, None, {})
    t = B.sensitivity(ss, scope="cell", top_authors=2)
    assert "outings only" in t.index and "without Podravska (Pohorje / Maribor)" in t.index
    assert (t["n_pos"] + t["n_none"] <= n).all()


# ---------------------------------------------------------------- retuning
def test_nested_loyo_picks_planted_candidate_and_extra_objective():
    rng = np.random.default_rng(6)
    n = 400
    years = rng.choice([2014, 2015, 2016, 2017], n)
    y = rng.random(n) < 0.5
    M = rng.normal(size=(5, n))
    M[3] = y + rng.normal(0, 0.7, n)                    # candidate 3 is informative
    r = B.nested_loyo(M, y, years)
    assert set(r["chosen"].values()) == {3}
    assert B.auc(r["held"], y) > 0.7
    # a second objective that prefers candidate 1 with a huge weight overrides
    extra = np.zeros((5, 50))
    extra[1] = 5.0
    r2 = B.nested_loyo(M, y, years, extra=extra, extra_years=np.repeat([2014, 2015, 2016, 2017], 13)[:50], weight_extra=0.99)
    assert set(r2["chosen"].values()) == {1}
    r3 = B.nested_loyo(M, y, years, extra=lambda yr: np.array([0, 5.0, 0, 0, 0]), weight_extra=0.99)
    assert set(r3["chosen"].values()) == {1}


def test_candidate_matrix_scores_every_candidate():
    eng, cells, _ = _engine_and_cells()
    d = _reports([dict(post_id=str(i), date_observed="2020-09-01") for i in range(6)])
    d["date"] = pd.to_datetime(d["date_observed"]) + pd.to_timedelta(np.arange(6) * 5, unit="D")
    d["y"] = [1, 0, 1, 0, 1, 0]
    loc = np.array([True, True, True, False, False, False])
    ru = B.UnitCells(np.tile(np.arange(8), (3, 1)), np.ones((3, 8), bool))
    nu = B.UnitCells(np.arange(20)[None, :], np.ones((1, 20), bool))
    td = B.make_tune_data(d, ru, nu, loc, k_cell=5, k_nat=10)
    cands = [prior.PriorParams(), prior.PriorParams(t_opt=18.0, p_half=150.0)]
    M = B.candidate_matrix(eng, td, cands)
    assert M.shape == (2, 6) and np.isfinite(M).all() and not np.allclose(M[0], M[1])
    assert (M >= 0).all() and (M <= 1).all()


def test_elevation_factors():
    f = B.elev_optimum_factor(np.array([300.0, 900.0, 1500.0]), 250, o0=900, o1=0, sd=300, floor=0.2)
    assert f[1] == pytest.approx(1.0) and f[0] == pytest.approx(f[2]) and 0.2 < f[0] < 1
    # the optimum moves up with the day of year
    early = B.elev_optimum_factor(1100.0, 180, o0=800, o1=6, sd=300)
    late = B.elev_optimum_factor(1100.0, 290, o0=800, o1=6, sd=300)
    assert late > early
    assert B.elev_optimum_factor(500.0, 250, o0=0, o1=0, sd=1.0, floor=1.0) == 1.0
    r = B.rule_table_factor(np.array([1000.0, 1000.0]), np.array([232, 305]))        # 20 Aug, 1 Nov
    assert r[0] > r[1] and r[0] == pytest.approx(1.0)
    base = np.ones((3, 4))
    valid = np.ones((3, 4), bool)
    E = B.elevation_candidates(base, valid, np.full((3, 4), 900.0), np.full(3, 250), B.ELEV_GRID[:5])
    assert E.shape == (5, 3)


def test_recommend_rule():
    def entry(lo_w, lo_y, diff_comb, diff_strat):
        return {"vs_production_week": {"lo": lo_w}, "vs_production_year": {"lo": lo_y},
                "tg_heldout": {"comb": {"diff": diff_comb}, "strat": {"diff": diff_strat}}}

    rt = {"procedures": {"good": entry(0.01, 0.02, 0.0, 0.01), "no_gain": entry(-0.01, 0.02, 0.0, 0.0),
                         "hurts": entry(0.01, 0.02, -0.05, 0.0)}}
    tg = pd.DataFrame({"diff_comb": [0.0, 0.0, -0.05], "diff_strat": [0.0, 0.0, 0.0]},
                      index=["pick_good", "pick_no_gain", "pick_hurts"])
    rec = B.recommend(rt, tg)
    assert rec["good"]["recommend"] and not rec["no_gain"]["recommend"] and not rec["hurts"]["recommend"]


def test_national_year_signal_and_weekly_year():
    idx = pd.date_range("2010-05-01", "2025-11-15", freq="D")
    daily = pd.DataFrame({"season": 1.0, "comb_lit": np.sin(idx.day_of_year / 60), "comb_tuned": 0.5, "full_tuned": 0.3}, index=idx)
    z = B.national_year_signal(daily)
    assert z.shape[0] == 16 and abs(z["comb_tuned"].sum()) < 1e-9
    d = pd.DataFrame({"year": [2015, 2015, 2014], "y": [1, 0, 1],
                      "date": pd.to_datetime(["2015-09-10", "2015-09-11", "2014-09-10"])})
    w = B.weekly_year(d, daily, 2015)
    assert int(w["n"].sum()) == 2 and "comb_lit_year" in w and "comb_lit_clim" in w


def test_records_keep_named_index():
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.RangeIndex(2010, 2012, name="year"))
    assert B._records(df)[0]["year"] == 2010
