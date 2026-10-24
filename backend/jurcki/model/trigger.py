"""Validate and tune the "when" part of the model: the fruiting trigger.

Ground truth is the dated GBIF finds (2010+). For every find the weather of
its own point (ERA5, lapse-corrected to the point elevation) is turned into
a daily score, and the find day is ranked among all days of the same point
in the same year's season window (May 1 - Nov 15): the temporal percentile
of ``evaluate.temporal_percentiles``. Its mean is the AUC for "when".

Scores compared (the app uses ``season x trigger``):
* ``season``   month factor only (``prior.season_factor``)
* ``trigger``  ``prior.trigger_score`` alone
* ``comb``     season x trigger

A find is one day of a multi-day flush and is logged with a lag, so every
score is evaluated twice: on the exact day, and tolerant (+-3 days: the
score is the max over find day +-3, the background is smoothed the same way
and the neighbouring days are left out of it). The tolerant variant is the
primary metric and the tuning objective; the exact-day one is reported too.

Tuning is a random search over ``PriorParams`` (the literature defaults are
candidate 0) scored by the mean ``comb`` percentile. The averaging windows
are kept at their literature values (freeing them overfits, see
``sample_params``); the free-window search is run alongside for comparison. Candidates are fixed
before looking at the data and every candidate's percentile for every find
is kept, so leave-one-year-out CV is a nested selection: for each held-out
year the best candidate on the *other* years is chosen and scored on the
held-out year. A small LightGBM / logistic model on the dynamic features
(find day vs the other season days of the same point) is cross-validated
the same way.

Run:  python -m jurcki.model.trigger [--candidates 800] [--no-save]
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, fields, replace
from datetime import date

import numpy as np
import pandas as pd
from dateutil.easter import easter
from scipy.ndimage import maximum_filter1d

from .. import config, features
from . import evaluate, prior

log = logging.getLogger(__name__)

DATED = config.PROCESSED / "gbif_boletus.parquet"
PARAMS_PATH = config.PROCESSED / "trigger_params.json"
MODEL_DIR = config.INTERIM / "model"
ML_PATH = MODEL_DIR / "trigger_lgbm.txt"

COVER = (13.25, 45.25, 16.75, 47.0)   # ERA5 window (lon0, lat0, lon1, lat1)
MAX_UNCERTAINTY_M = 10_000
SEASON = ((5, 1), (11, 15))           # comparison window within each year
WARMUP = (3, 1)                       # weather starts here (windows need history)
T = 260                               # days from WARMUP to Nov 15 (same in leap years)
TOL = 3                               # days, tolerant variant
MIN_GAIN = 0.005                      # tuned params are recommended only if LOYO gains this much
TRIGGER_FIELDS = ["t_opt", "t_sd", "t_window", "p_window", "p_half", "pulse_min",
                  "pulse_lag", "heat_tmax", "heat_window", "frost_tmin", "frost_window"]
SI_HOLIDAYS = [(1, 1), (1, 2), (2, 8), (4, 27), (5, 1), (5, 2), (6, 25), (8, 15),
               (10, 31), (11, 1), (12, 25), (12, 26)]


# --- finds and panel --------------------------------------------------------
def load_finds(path=DATED, min_year: int = 2010) -> pd.DataFrame:
    """Dated jurcki finds inside the weather window, with point elevation.

    Drops coarse coordinates (> 10 km) and same-day duplicates at one place.
    ``elev_src`` is "dem" or "era5" (ERA5 orography outside the DEM grid).
    """
    d = pd.read_parquet(path)
    d = d[d["is_jurcki"] & d["date"].notna() & (d["date"] >= f"{min_year}-01-01")].copy()
    d = d[~(d["coordinateuncertaintyinmeters"] > MAX_UNCERTAINTY_M)]
    lon0, lat0, lon1, lat1 = COVER
    d = d[d["decimallongitude"].between(lon0, lon1) & d["decimallatitude"].between(lat0, lat1)]
    out = pd.DataFrame({"date": d["date"].dt.normalize(), "lon": d["decimallongitude"],
                        "lat": d["decimallatitude"], "country": d["countrycode"]})
    out["lon"], out["lat"] = out["lon"].round(3), out["lat"].round(3)   # ~100 m
    out = out.drop_duplicates(["date", "lon", "lat"]).reset_index(drop=True)
    ez = [features.point_elevation(x, y) for x, y in zip(out["lon"], out["lat"])]
    out["elev"], out["elev_src"] = [z[0] for z in ez], [z[1] for z in ez]
    out["year"] = out["date"].dt.year
    return out


def is_off_day(dates) -> np.ndarray:
    """True on weekends and Slovenian public holidays (observer effort peaks)."""
    d = pd.DatetimeIndex(np.ravel(dates))
    off = np.asarray(d.day_of_week >= 5)
    fixed = {(m, day) for m, day in SI_HOLIDAYS}
    off |= np.fromiter(((x.month, x.day) in fixed for x in d), bool, len(d))
    easters = {y: pd.Timestamp(easter(y)) + pd.Timedelta(days=1) for y in d.year.unique()}
    off |= np.fromiter((x.normalize() == easters[x.year] for x in d), bool, len(d))
    return off.reshape(np.shape(dates))


@dataclass
class Panel:
    """Daily weather of every (point, year) series that has finds.

    Arrays are (T, N): days from March 1 (warm-up) to Nov 15, series last.
    ``season`` marks the comparison days; ``finds`` carries ``sid`` (series
    index) and ``t`` (day index) for each find inside the window.
    """
    series: pd.DataFrame
    dates: np.ndarray
    weather: dict
    valid: np.ndarray
    season: np.ndarray
    off_day: np.ndarray
    finds: pd.DataFrame


def assemble(series: pd.DataFrame, weather: dict, valid: np.ndarray, finds: pd.DataFrame) -> Panel:
    """Build a Panel from series (lon, lat, elev, year), weather arrays and finds (sid, date)."""
    start = np.array([np.datetime64(f"{y}-{WARMUP[0]:02d}-{WARMUP[1]:02d}") for y in series["year"]])
    dates = start[None, :] + np.arange(T)[:, None]
    lo = np.array([np.datetime64(f"{y}-{SEASON[0][0]:02d}-{SEASON[0][1]:02d}") for y in series["year"]])
    hi = np.array([np.datetime64(f"{y}-{SEASON[1][0]:02d}-{SEASON[1][1]:02d}") for y in series["year"]])
    season = valid & (dates >= lo[None, :]) & (dates <= hi[None, :])
    f = finds.copy()
    f["t"] = (f["date"].to_numpy("datetime64[D]") - start[f["sid"].to_numpy()]).astype(int)
    keep = (f["t"] >= 0) & (f["t"] < T)
    f = f[keep]
    f = f[season[f["t"].to_numpy(), f["sid"].to_numpy()]].reset_index(drop=True)
    return Panel(series.reset_index(drop=True), dates, weather, valid, season,
                 is_off_day(dates), f)


def build_panel(finds: pd.DataFrame) -> Panel:
    """Fetch the point weather for every (point, year) of ``finds`` (ERA5, no network)."""
    key = finds[["lon", "lat", "year"]].drop_duplicates().reset_index(drop=True)
    elev = finds.groupby(["lon", "lat"])["elev"].first()
    key["elev"] = [elev[(x, y)] for x, y in zip(key["lon"], key["lat"])]
    n = len(key)
    w = {k: np.zeros((T, n)) for k in features.WEATHER_VARS}
    valid = np.zeros((T, n), bool)
    for i, r in key.iterrows():
        df = features.point_weather(r.lon, r.lat, r.elev, f"{int(r.year)}-03-01", f"{int(r.year)}-11-15")
        valid[:len(df), i] = True   # the history may end before Nov 15
        df = df.reindex(pd.date_range(f"{int(r.year)}-03-01", periods=T)).ffill()
        for k in w:
            w[k][:, i] = df[k].to_numpy()
    idx = {(a, b, c): i for i, (a, b, c) in enumerate(zip(key["lon"], key["lat"], key["year"]))}
    f = finds.copy()
    f["sid"] = [idx[(a, b, c)] for a, b, c in zip(f["lon"], f["lat"], f["year"])]
    return assemble(key, w, valid, f)


# --- scoring ----------------------------------------------------------------
def season_scores(panel: Panel) -> np.ndarray:
    """``prior.season_factor`` by month, (T, N)."""
    months = panel.dates.astype("datetime64[M]").astype(int) % 12 + 1
    table = np.array([prior.season_factor(m) for m in range(1, 13)])
    return table[months - 1]


def trigger_scores(panel: Panel, p: prior.PriorParams) -> np.ndarray:
    """``prior.trigger_score`` of the panel's weather, (T, N)."""
    w = panel.weather
    return prior.trigger_score(prior.weather_features(w["tmean"], w["tmin"], w["tmax"], w["precip"], p), p)


def percentiles(panel: Panel, score: np.ndarray, tol: int = 0, matched: bool = False,
                rows=None, lag: int = 0) -> np.ndarray:
    """Temporal percentile of each find's day (vectorised ``evaluate.temporal_percentiles``).

    score: (T, N). tol > 0: tolerant variant (see module doc). matched: the
    background only holds days of the same kind (weekend/holiday or workday)
    as the find day. rows: subset of find indices. lag: rank the day ``lag``
    days before the find instead (diagnostic for timing shifts).
    """
    f = panel.finds if rows is None else panel.finds.iloc[rows]
    sid, t = f["sid"].to_numpy(), np.clip(f["t"].to_numpy() - lag, 0, T - 1)
    s = maximum_filter1d(score, 2 * tol + 1, axis=0, mode="nearest") if tol else score
    S = s[:, sid]
    k = np.arange(len(f))
    own = S[t, k]
    M = panel.season[:, sid] & (np.abs(np.arange(T)[:, None] - t[None, :]) > tol)
    if matched:
        M &= panel.off_day[:, sid] == panel.off_day[t, sid][None, :]
    below = ((S < own) & M).sum(0)
    ties = ((S == own) & M).sum(0)
    pop = M.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pop > 0, (below + 0.5 * ties) / pop, np.nan)


def percentiles_ref(panel: Panel, score: np.ndarray, tol: int = 0, matched: bool = False) -> np.ndarray:
    """Same as ``percentiles`` but through ``evaluate.temporal_percentiles`` (slow, for checks)."""
    f = panel.finds
    s = maximum_filter1d(score, 2 * tol + 1, axis=0, mode="nearest") if tol else score
    season_days = np.unique(panel.dates[panel.season])

    def series_by_cell(i, _):
        sid, t = f["sid"].iat[i], f["t"].iat[i]
        sc = s[:, sid].astype(float).copy()
        sc[~panel.season[:, sid]] = np.nan
        near = np.abs(np.arange(T) - t) <= tol
        near[t] = False
        sc[near] = np.nan
        if matched:
            sc[panel.off_day[:, sid] != panel.off_day[t, sid]] = np.nan
        return panel.dates[:, sid], sc

    items = [(np.datetime64(d, "D"), i, 0) for i, d in enumerate(f["date"])]
    return evaluate.temporal_percentiles(series_by_cell, items, season_days)


VARIANTS = {  # name -> (score kind, tol, weekday-matched background)
    "comb_tol": ("comb", TOL, False), "comb_exact": ("comb", 0, False),
    "trig_tol": ("trig", TOL, False), "trig_exact": ("trig", 0, False),
    "comb_tol_wk": ("comb", TOL, True), "comb_exact_wk": ("comb", 0, True),
}
PRIMARY = "comb_tol"


def all_variants(panel: Panel, trig: np.ndarray | None, season: np.ndarray, rows=None) -> dict:
    """Percentile vectors for every variant; ``trig=None`` means the season-only baseline."""
    comb = season if trig is None else season * trig
    out = {}
    for name, (kind, tol, wk) in VARIANTS.items():
        if kind == "trig" and trig is None:
            continue
        out[name] = percentiles(panel, trig if kind == "trig" else comb, tol, wk, rows)
    return out


# --- parameter search -------------------------------------------------------
def sample_params(n: int, seed: int = 0, free_windows: bool = False) -> list[prior.PriorParams]:
    """Candidate ``PriorParams``: the literature defaults first, then random draws.

    By default the averaging windows (t_window, p_window, heat_window,
    frost_window) stay at their literature values: with ~270 finds, freeing
    them let the search drift to the edge of the grid (a 60-day "temperature"
    is just the season) and did not generalise. ``free_windows`` draws them too.
    """
    rng = np.random.default_rng(seed)
    base = prior.PriorParams()
    out = [base]
    for _ in range(n - 1):
        lo = int(rng.choice([3, 5, 7, 10, 14]))
        win = dict(t_window=int(rng.choice([7, 10, 14, 20, 26, 30, 40, 50, 60])),
                   p_window=int(rng.choice([7, 10, 14, 20, 26, 35, 42, 56, 70])),
                   heat_window=int(rng.choice([3, 5, 10, 14, 21, 28])),
                   frost_window=int(rng.choice([3, 7, 14])))
        out.append(replace(
            base, **(win if free_windows else {}),
            t_opt=float(rng.uniform(8, 20)), t_sd=float(rng.uniform(2, 9)),
            p_half=float(np.exp(rng.uniform(np.log(15), np.log(250)))),
            pulse_min=float(rng.uniform(10, 40)),
            pulse_lag=(lo, lo + int(rng.choice([3, 5, 7, 9, 12]))),
            heat_tmax=float(rng.uniform(25, 34)), frost_tmin=float(rng.uniform(-4, 2))))
    return out


def search(panel: Panel, candidates: list) -> dict:
    """Percentiles of every find for every candidate: {variant: (n_candidates, n_finds)}."""
    season = season_scores(panel)
    mats = {k: np.empty((len(candidates), len(panel.finds))) for k in VARIANTS}
    for i, p in enumerate(candidates):
        for k, v in all_variants(panel, trigger_scores(panel, p), season).items():
            mats[k][i] = v
    return mats


def loyo_select(mats: dict, years: np.ndarray, objective: str = PRIMARY) -> dict:
    """Nested leave-one-year-out selection.

    For each year the candidate with the best mean ``objective`` on the other
    years is chosen. Returns the chosen index per year, its training score,
    and for every variant the held-out percentile of each find.
    """
    obj = mats[objective]
    chosen, train = {}, {}
    held = {k: np.full(len(years), np.nan) for k in mats}
    for y in np.unique(years):
        tr, te = years != y, years == y
        sc = np.nanmean(obj[:, tr], axis=1)
        c = int(np.argmax(sc))
        chosen[int(y)], train[int(y)] = c, float(sc[c])
        for k, m in mats.items():
            held[k][te] = m[c, te]
    return {"chosen": chosen, "train_auc": train, "held": held}


def year_bootstrap(years: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple:
    """Mean of ``a - b`` per find with a 90 % interval from resampling whole years."""
    d = a - b
    ys = np.unique(years)
    groups = [d[years == y] for y in ys]
    rng = np.random.default_rng(seed)
    means = [np.nanmean(np.concatenate([groups[i] for i in rng.integers(0, len(ys), len(ys))]))
             for _ in range(n)]
    return float(np.nanmean(d)), float(np.percentile(means, 5)), float(np.percentile(means, 95))


# --- ML on dynamic features -------------------------------------------------
ML_FEATURES = features.DYNAMIC_NAMES + ["doy", "elev"]
LGBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=4, max_depth=3,
                   min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                   bagging_freq=1, lambda_l2=20.0, verbose=-1, seed=1, num_threads=4)
LGBM_ROUNDS = 150


def ml_matrix(panel: Panel) -> np.ndarray:
    """Dynamic features (literature windows) + day of year + elevation: (T, N, K)."""
    f = features.dynamic_feature_arrays(panel.weather)
    years = panel.series["year"].to_numpy()
    doy = (panel.dates - np.array([np.datetime64(f"{y}-01-01") for y in years])[None, :]).astype(float) + 1
    f["doy"] = doy
    f["elev"] = np.broadcast_to(panel.series["elev"].to_numpy(float)[None, :], doy.shape)
    return np.stack([f[k] for k in ML_FEATURES], axis=-1)


def _labels(panel: Panel):
    """Positive count per (T, N) cell and sample weights (background sums to the positives per series)."""
    y = np.zeros(panel.season.shape)
    np.add.at(y, (panel.finds["t"].to_numpy(), panel.finds["sid"].to_numpy()), 1.0)
    bg = panel.season & (y == 0)
    npos = y.sum(0)
    w = np.where(y > 0, y, bg * (npos / np.maximum(bg.sum(0), 1))[None, :])
    return y, w


def _design(X: np.ndarray, kind: str, names: list) -> np.ndarray:
    """Model matrix; the logistic model gets squares of the features with an optimum."""
    if kind == "logit":
        ix = [names.index(k) for k in ("t20", "stl1_20", "doy", "elev") if k in names]
        X = np.concatenate([X, X[..., ix] ** 2], axis=-1)
    return X


def fit_ml(panel: Panel, X: np.ndarray, cols: np.ndarray, kind: str, names=ML_FEATURES):
    """Fit on the series in ``cols`` using the feature subset ``names`` of ``X``.

    Returns (model, predict) where predict(X) gives a score array shaped like
    X without its last axis.
    """
    y, w = _labels(panel)
    ix = [ML_FEATURES.index(k) for k in names]
    X = X[..., ix]
    sel = panel.season[:, cols]
    A = _design(X[:, cols], kind, list(names))[sel]
    yy, ww = (y[:, cols][sel] > 0).astype(float), w[:, cols][sel]
    ok = ww > 0
    if kind == "lgbm":
        import lightgbm as lgb
        model = lgb.train(LGBM_PARAMS, lgb.Dataset(A[ok], yy[ok], weight=ww[ok]), LGBM_ROUNDS)
        return model, lambda Z: model.predict(_design(Z[..., ix], kind, list(names)).reshape(-1, A.shape[-1])).reshape(Z.shape[:-1])
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
    model.fit(A[ok], yy[ok], logisticregression__sample_weight=ww[ok])
    return model, lambda Z: model.decision_function(_design(Z[..., ix], kind, list(names)).reshape(-1, A.shape[-1])).reshape(Z.shape[:-1])


def ml_loyo(panel: Panel, X: np.ndarray, kind: str, names=ML_FEATURES) -> dict:
    """Leave-one-year-out held-out percentiles of the ML score for every variant."""
    years = panel.series["year"].to_numpy()
    fy = panel.finds["sid"].map(lambda s: years[s]).to_numpy()
    held = {k: np.full(len(fy), np.nan) for k in VARIANTS if k.startswith("comb")}
    for y in np.unique(fy):
        _, predict = fit_ml(panel, X, np.flatnonzero(years != y), kind, names)
        rows = np.flatnonzero(fy == y)
        sc = predict(X)
        for k in held:
            _, tol, wk = VARIANTS[k]
            held[k][rows] = percentiles(panel, sc, tol, wk, rows)
    return held


# --- elevation, effort ------------------------------------------------------
BANDS = [0, 400, 700, 1000, 1300, 3000]
BAND_LABELS = ["<400", "400-700", "700-1000", "1000-1300", ">1300"]


def elevation_table(finds: pd.DataFrame) -> dict:
    """Find month x elevation band counts, median day of year by band, Spearman rho."""
    from scipy.stats import spearmanr
    d = finds.copy()
    d["band"] = pd.cut(d["elev"], BANDS, labels=BAND_LABELS, right=False)
    d["doy"] = d["date"].dt.day_of_year
    d["month"] = d["date"].dt.month
    counts = pd.crosstab(d["month"], d["band"]).reindex(columns=BAND_LABELS, fill_value=0)
    by_band = d.groupby("band", observed=False).agg(n=("doy", "size"), median_doy=("doy", "median"),
                                                    mean_doy=("doy", "mean")).round(1)
    within = d["doy"] - d.groupby("year")["doy"].transform("mean")   # remove the year effect
    rho_raw = spearmanr(d["elev"], d["doy"])
    rho_year = spearmanr(d["elev"], within)
    return {"counts": counts, "by_band": by_band,
            "spearman_doy": (float(rho_raw.statistic), float(rho_raw.pvalue)),
            "spearman_doy_within_year": (float(rho_year.statistic), float(rho_year.pvalue))}


def lag_scan(panel: Panel, score: np.ndarray, lags=range(-10, 21, 2)) -> pd.DataFrame:
    """Mean exact-day percentile per elevation band when each find is moved ``lag`` days earlier."""
    band = pd.cut(panel.finds["sid"].map(panel.series["elev"]), BANDS, labels=BAND_LABELS, right=False)
    rows = {lag: pd.Series(percentiles(panel, score, 0, False, None, lag)).groupby(band.to_numpy(), observed=False).mean()
            for lag in lags}
    return pd.DataFrame(rows).T.round(3)


def effort_check(panel: Panel) -> dict:
    """Share of finds on weekends/holidays vs the share of such days in the season window."""
    from scipy.stats import binomtest
    f = panel.finds
    off = panel.off_day[f["t"].to_numpy(), f["sid"].to_numpy()]
    base = float(panel.off_day[:, f["sid"].to_numpy()][panel.season[:, f["sid"].to_numpy()]].mean())
    k, n = int(off.sum()), len(off)
    return {"n": n, "off_day_share": k / n, "baseline_share": base,
            "p_value": float(binomtest(k, n, base, alternative="greater").pvalue)}


# --- params file ------------------------------------------------------------
def params_to_dict(p: prior.PriorParams) -> dict:
    """Trigger fields of ``p`` as a JSON-friendly dict."""
    d = {k: getattr(p, k) for k in TRIGGER_FIELDS}
    d["pulse_lag"] = list(d["pulse_lag"])
    return d


def params_from_dict(d: dict) -> prior.PriorParams:
    """Inverse of ``params_to_dict``; habitat fields keep their defaults."""
    d = {k: d[k] for k in TRIGGER_FIELDS if k in d}
    if "pulse_lag" in d:
        d["pulse_lag"] = tuple(d["pulse_lag"])
    return prior.PriorParams(**d)


def load_trigger_params(path=PARAMS_PATH) -> prior.PriorParams:
    """Recommended trigger parameters from ``trigger_params.json`` (literature defaults if absent)."""
    try:
        with open(path) as f:
            return params_from_dict(json.load(f)["params"])
    except FileNotFoundError:
        log.warning("%s not found, using literature defaults", path)
        return prior.PriorParams()


def load_trigger_model(path=ML_PATH):
    """The learned trigger model (LightGBM booster) if one was saved, else None."""
    if not path.exists():
        return None
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path))


# --- driver -----------------------------------------------------------------
def summ(p) -> dict:
    s = evaluate.summarize(p, top=(0.10, 0.20))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}


def block(held: dict) -> dict:
    """{variant: summary} for a dict of percentile vectors."""
    return {k: summ(v) for k, v in held.items()}


def per_year(years: np.ndarray, vecs: dict, key: str) -> pd.DataFrame:
    """Mean percentile (AUC) per year for each named vector."""
    df = pd.DataFrame({name: v[key] for name, v in vecs.items()})
    df["year"] = years
    out = df.groupby("year").mean().round(3)
    out.insert(0, "n", df.groupby("year").size())
    return out


def run(n_candidates: int = 800, seed: int = 0, ml: bool = True, finds: pd.DataFrame | None = None,
        panel: Panel | None = None) -> dict:
    """Evaluate, tune and cross-validate; returns a report dict (see ``format_report``)."""
    if panel is None:
        finds = load_finds() if finds is None else finds
        panel = build_panel(finds)
    f = panel.finds
    years = panel.series["year"].to_numpy()[f["sid"].to_numpy()]
    season = season_scores(panel)
    cands = sample_params(n_candidates, seed)
    rep = {"n_finds": len(f), "n_series": len(panel.series), "years": sorted(set(map(int, years))),
           "n_candidates": n_candidates, "seed": seed}

    # literature prior and baselines (no fitting: every find is out of sample)
    lit_trig = trigger_scores(panel, cands[0])
    lit = all_variants(panel, lit_trig, season)
    base = all_variants(panel, None, season)
    chk = np.nanmax(np.abs(percentiles(panel, season * lit_trig, TOL) - percentiles_ref(panel, season * lit_trig, TOL)))
    rep["fast_vs_ref_max_diff"] = float(chk)

    # tuning
    mats = search(panel, cands)
    cv = loyo_select(mats, years)
    best = int(np.argmax(np.nanmean(mats[PRIMARY], axis=1)))
    cv_wk = loyo_select(mats, years, "comb_tol_wk")
    free = sample_params(n_candidates, seed, free_windows=True)
    mats_free = search(panel, free)
    cv_free = loyo_select(mats_free, years)
    free_best = int(np.argmax(np.nanmean(mats_free[PRIMARY], axis=1)))
    tuned_in = {k: m[best] for k, m in mats.items()}
    chosen = pd.DataFrame([params_to_dict(cands[c]) for c in cv["chosen"].values()],
                          index=list(cv["chosen"].keys()))
    rep.update(literature=block(lit), season_only=block(base), tuned_loyo=block(cv["held"]),
               tuned_in_sample=block(tuned_in), tuned_loyo_wk_objective=block(cv_wk["held"]),
               free_windows_loyo=block(cv_free["held"]), free_windows_in_sample=block({k: m[free_best] for k, m in mats_free.items()}),
               free_windows_best=params_to_dict(free[free_best]),
               train_auc_mean=float(np.mean(list(cv["train_auc"].values()))),
               best=params_to_dict(cands[best]), best_idx=best, fold_params=chosen,
               n_distinct_fold_choices=int(len(set(cv["chosen"].values()))))
    vecs = {"season": base, "literature": lit, "tuned_loyo": cv["held"]}

    # ML
    if ml:
        X = ml_matrix(panel)
        for kind in ("lgbm", "logit"):
            held = ml_loyo(panel, X, kind)
            rep[f"{kind}_loyo"] = block(held)
            vecs[kind] = held
        # ablations: how much comes from the weather, how much from the calendar and elevation
        for key, names in (("lgbm_calendar", ["doy", "elev"]),
                           ("lgbm_weather", [n for n in ML_FEATURES if n not in ("doy", "elev")])):
            held = ml_loyo(panel, X, "lgbm", names)
            rep[f"{key}_loyo"] = block(held)
            vecs[key] = held
        rep["ml_best"] = max(("lgbm", "logit"), key=lambda k: rep[f"{k}_loyo"][PRIMARY]["auc"])

    gain = rep["tuned_loyo"][PRIMARY]["auc"] - rep["literature"][PRIMARY]["auc"]
    rep["tuned_gain_loyo"] = round(gain, 4)
    rep["recommended"] = "tuned" if gain >= MIN_GAIN else "literature"
    rep["gain_ci"] = {"tuned_minus_literature": year_bootstrap(years, cv["held"][PRIMARY], lit[PRIMARY]),
                      "tuned_minus_season": year_bootstrap(years, cv["held"][PRIMARY], base[PRIMARY]),
                      "literature_minus_season": year_bootstrap(years, lit[PRIMARY], base[PRIMARY])}
    if ml:
        rep["gain_ci"]["ml_minus_tuned"] = year_bootstrap(years, vecs[rep["ml_best"]][PRIMARY], cv["held"][PRIMARY])
    rep["ml_wins"] = bool(ml and rep[f"{rep['ml_best']}_loyo"][PRIMARY]["auc"]
                          >= rep["tuned_loyo"][PRIMARY]["auc"] + 0.01)

    # per year, per country, per elevation band (held-out)
    rep["per_year_tol"] = per_year(years, vecs, PRIMARY)
    rep["per_year_exact"] = per_year(years, vecs, "comb_exact")
    if "country" in f:
        d = pd.DataFrame({"c": np.where(f["country"] == "SI", "SI", "other"),
                          "tuned": cv["held"][PRIMARY], "lit": lit[PRIMARY]})
        rep["per_country"] = d.groupby("c").agg(n=("tuned", "size"), literature=("lit", "mean"),
                                                tuned_loyo=("tuned", "mean")).round(3)

    # elevation, effort
    fe = f.assign(elev=f["sid"].map(panel.series["elev"]), year=years)
    rep["elevation"] = elevation_table(fe)
    tuned_p = cands[best]
    comb = season * trigger_scores(panel, tuned_p)
    rep["lag_scan_comb"] = lag_scan(panel, comb)
    rep["lag_scan_trigger"] = lag_scan(panel, trigger_scores(panel, tuned_p))
    band = pd.cut(fe["elev"], BANDS, labels=BAND_LABELS, right=False).to_numpy()
    rep["per_band_heldout"] = (pd.DataFrame({"b": band, "literature": lit[PRIMARY], "tuned_loyo": cv["held"][PRIMARY]})
                               .groupby("b", observed=False).agg(n=("tuned_loyo", "size"), literature=("literature", "mean"),
                                                                 tuned_loyo=("tuned_loyo", "mean")).round(3))
    rep["effort"] = effort_check(panel)
    rep["panel"], rep["cands"], rep["cv"] = panel, cands, cv
    if ml:
        rep["X"] = X
    return rep


def save(rep: dict, path=PARAMS_PATH, model_dir=MODEL_DIR) -> None:
    """Write trigger_params.json (and the learned model if it wins)."""
    cands, panel = rep["cands"], rep["panel"]
    tuned, lit = params_to_dict(cands[rep["best_idx"]]), params_to_dict(cands[0])
    out = {"generated": date.today().isoformat(), "recommended": rep["recommended"],
           "params": tuned if rep["recommended"] == "tuned" else lit,
           "tuned": tuned, "literature": lit,
           "primary_metric": f"{PRIMARY}: mean temporal percentile of season x trigger, "
                             f"+-{TOL} day tolerant, leave-one-year-out",
           "metrics": {k: rep[k] for k in ("literature", "season_only", "tuned_loyo", "tuned_in_sample",
                                           "tuned_loyo_wk_objective", "free_windows_loyo",
                                           "free_windows_in_sample") if k in rep},
           "free_windows_best": rep["free_windows_best"],
           "n_finds": rep["n_finds"], "years": rep["years"], "n_candidates": rep["n_candidates"],
           "seed": rep["seed"], "tuned_gain_loyo": rep["tuned_gain_loyo"]}
    for k in ("lgbm_loyo", "logit_loyo", "lgbm_calendar_loyo", "lgbm_weather_loyo"):
        if k in rep:
            out["metrics"][k] = rep[k]
    out["ml_wins"] = rep["ml_wins"]
    if rep["ml_wins"]:
        kind = rep["ml_best"]
        model, _ = fit_ml(panel, rep["X"], np.arange(len(panel.series)), kind)
        if kind == "lgbm":
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_dir / ML_PATH.name))
            out["ml_model"] = {"file": ML_PATH.name, "features": ML_FEATURES, "kind": kind}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def format_report(rep: dict) -> str:
    """Plain-text report of a ``run`` result."""
    lines = []
    add = lines.append
    add(f"finds {rep['n_finds']} in {rep['n_series']} point-years, years {rep['years'][0]}-{rep['years'][-1]}; "
        f"{rep['n_candidates']} candidates; primary = {PRIMARY}")
    add(f"fast vs evaluate.temporal_percentiles max diff: {rep['fast_vs_ref_max_diff']:.2e}")
    rows = []
    for name, key in [("season only", "season_only"), ("literature trigger", "literature"),
                      ("tuned (in-sample)", "tuned_in_sample"), ("tuned (LOYO)", "tuned_loyo"),
                      ("tuned (LOYO, wk-matched obj.)", "tuned_loyo_wk_objective"),
                      ("tuned, free windows (in-sample)", "free_windows_in_sample"),
                      ("tuned, free windows (LOYO)", "free_windows_loyo"),
                      ("lgbm (LOYO)", "lgbm_loyo"), ("logit (LOYO)", "logit_loyo"),
                      ("lgbm doy+elev only (LOYO)", "lgbm_calendar_loyo"),
                      ("lgbm weather only (LOYO)", "lgbm_weather_loyo")]:
        if key not in rep:
            continue
        for v, label in [("comb_tol", "comb +-3d"), ("comb_exact", "comb exact"),
                         ("trig_tol", "trigger +-3d"), ("trig_exact", "trigger exact"),
                         ("comb_tol_wk", "comb +-3d wk-matched")]:
            s = rep[key].get(v)
            if s:
                rows.append({"model": name, "metric": label, "n": s["n"], "auc": s["auc"],
                             "top10": round(s["hit_top10"], 3), "top20": round(s["hit_top20"], 3)})
    add(pd.DataFrame(rows).to_string(index=False))
    add(f"\nmean train AUC of the selected candidate over folds: {rep['train_auc_mean']:.3f} "
        f"(held-out {rep['tuned_loyo'][PRIMARY]['auc']:.3f}); distinct choices over folds: {rep['n_distinct_fold_choices']}")
    add(f"recommended: {rep['recommended']} (LOYO gain {rep['tuned_gain_loyo']:+.4f}); ml wins: {rep['ml_wins']}")
    lit = params_to_dict(rep["cands"][0])
    best = rep["best"]
    fp = rep["fold_params"]
    pr = pd.DataFrame({"literature": pd.Series(lit), "tuned (all years)": pd.Series(best),
                       "free-window best": pd.Series(rep["free_windows_best"])})
    num = fp.drop(columns="pulse_lag").astype(float)
    pr["fold median"] = pd.concat([num.median(), pd.Series({"pulse_lag": ""})])
    pr["fold min"] = pd.concat([num.min(), pd.Series({"pulse_lag": ""})])
    pr["fold max"] = pd.concat([num.max(), pd.Series({"pulse_lag": ""})])
    add("\nAUC differences (comb +-3d, mean, 90% interval from resampling years)")
    for k, (m, lo, hi) in rep["gain_ci"].items():
        add(f"  {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    add("\nparameters\n" + pr.to_string())
    add("\nheld-out AUC per year, comb +-3d\n" + rep["per_year_tol"].to_string())
    add("\nheld-out AUC per year, comb exact day\n" + rep["per_year_exact"].to_string())
    if "per_country" in rep:
        add("\nby country (held-out, comb +-3d)\n" + rep["per_country"].to_string())
    add("\nby elevation band (held-out, comb +-3d)\n" + rep["per_band_heldout"].to_string())
    e = rep["elevation"]
    add("\nfinds: month x elevation band\n" + e["counts"].to_string())
    add("\nby band\n" + e["by_band"].to_string())
    add(f"Spearman(elev, day of year) {e['spearman_doy'][0]:+.2f} (p={e['spearman_doy_within_year'][1]:.3g} within year: "
        f"{e['spearman_doy_within_year'][0]:+.2f})")
    add("\nlag scan, season x trigger (exact-day AUC if finds were moved `lag` days earlier)\n" + rep["lag_scan_comb"].to_string())
    add("\nlag scan, trigger only\n" + rep["lag_scan_trigger"].to_string())
    ef = rep["effort"]
    add(f"\nobserver effort: {ef['off_day_share']:.1%} of finds on weekends/holidays vs {ef['baseline_share']:.1%} of "
        f"season days (one-sided binomial p={ef['p_value']:.2g})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rep = run(args.candidates, args.seed, ml=not args.no_ml)
    print(format_report(rep))
    if not args.no_save:
        save(rep)
        print(f"\nwrote {PARAMS_PATH}")


if __name__ == "__main__":
    main()
