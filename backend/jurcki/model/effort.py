"""Observer-effort corrected analysis of jurcki finds (target-group approach).

Presence-only finds mix "where / when jurcki fruit" with "where / when people
look for mushrooms and report them". The other fungi records of GBIF
(``sources/gbif_effort``, data/processed/gbif_fungi.parquet) show the second
part: a day with many fungi records and no jurcki is a weak absence, a jurcki
share that rises above the effort baseline is a real signal.

Contents
* statistics   Wilson / Poisson intervals, observed / expected ratio by year,
               Spearman with bootstrap interval, Holm / BH adjustment, AUC
* cell-days    25 km cell x day units of the target group, label = any jurcki
* weather      ERA5 at the cells (same interpolation + lapse rate as the
               trigger) and the candidate "trend" features of ``extra_features``
* evaluation   trigger as predictor of P(jurcki | fungi reported), year-grouped
               CV; habitat model against a target-group background
* trends       exploratory scan of candidate features with held-out years

Run:  python -m jurcki.model.effort [--quick]      (writes docs/figures and
data/processed/effort_results.json; the report docs/TRENDS.md is written by hand
from those numbers)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .. import config

FUNGI = config.PROCESSED / "gbif_fungi.parquet"
SEASON = ((5, 1), (11, 15))        # same comparison window as model.trigger
CELL_KM = 25
MIN_YEAR = 2010

# target groups: which non-jurcki fungi records stand for "people looking at mushrooms"
MYCORRHIZAL_FAMILIES = (
    "Boletaceae", "Suillaceae", "Gomphidiaceae", "Paxillaceae", "Russulaceae",
    "Cantharellaceae", "Hydnaceae", "Amanitaceae", "Cortinariaceae", "Inocybaceae",
    "Gomphaceae", "Hygrophoraceae", "Albatrellaceae", "Bankeraceae", "Thelephoraceae",
)
TARGET_GROUPS = ("fungi", "macro", "myco", "bolete")


# =========================================================================== statistics
def wilson_ci(k, n, z: float = 1.96):
    """Wilson score interval of a binomial proportion (arrays allowed): (lo, hi)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    lo, hi = np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)
    return np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)


def poisson_ci(k, alpha: float = 0.05):
    """Exact (Garwood) interval of a Poisson count (arrays allowed): (lo, hi)."""
    k = np.asarray(k, float)
    lo = np.where(k > 0, stats.chi2.ppf(alpha / 2, 2 * np.maximum(k, 1)) / 2, 0.0)
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return lo, hi


def week_of_year(dates) -> np.ndarray:
    """1..52 from the day of year (days 365 / 366 join week 52)."""
    doy = pd.DatetimeIndex(np.ravel(dates)).day_of_year.to_numpy()
    return np.minimum((doy - 1) // 7 + 1, 52)


def in_season(dates, season=SEASON) -> np.ndarray:
    """True for dates within season[0] .. season[1] (month, day) of their year."""
    d = pd.DatetimeIndex(np.ravel(dates))
    key = d.month.to_numpy() * 100 + d.day.to_numpy()
    return (key >= season[0][0] * 100 + season[0][1]) & (key <= season[1][0] * 100 + season[1][1])


def share_table(is_jurcki, groups) -> pd.DataFrame:
    """Jurcki share among target-group records per group with Wilson 95 % interval."""
    df = pd.DataFrame({"g": np.asarray(groups), "j": np.asarray(is_jurcki, bool)})
    t = df.groupby("g").agg(n=("j", "size"), k=("j", "sum"))
    t["share"] = t["k"] / t["n"]
    t["lo"], t["hi"] = wilson_ci(t["k"], t["n"])
    return t


def observed_expected(is_jurcki, year, week, strata=None, loo: bool = True, shrink: float = 20.0) -> pd.DataFrame:
    """Per year: observed jurcki records vs the number expected from the baseline share.

    The baseline share of a (stratum, week) is the jurcki share of all records
    in it (of the *other* years when ``loo``), shrunk towards the stratum's
    overall share with weight ``shrink`` records so that thin cells do not give
    0 or 1. The ratio O/E says how much more (or less) of the records were jurcki
    than usual for the same weeks and the same kind of observer: it removes the
    seasonal shape, the number of records and, with ``strata`` (e.g. the dataset),
    changes in who reports. The interval is the exact Poisson one on O (E is
    treated as fixed).
    """
    st = np.zeros(len(np.asarray(year)), int) if strata is None else np.asarray(strata)
    df = pd.DataFrame({"y": np.asarray(year), "w": np.asarray(week), "s": st, "j": np.asarray(is_jurcki, float)})
    key = ["s", "w"]
    tot = df.groupby(key)["j"].agg(["sum", "size"])
    stot = df.groupby("s")["j"].agg(["sum", "size"])
    out = []
    for y, g in df.groupby("y"):
        yk = g.groupby(key)["j"].agg(["sum", "size"])
        base = tot.reindex(yk.index).fillna(0)
        sbase = stot.reindex(yk.index.get_level_values("s")).fillna(0).to_numpy()
        if loo:
            base = base - yk
            ys = g.groupby("s")["j"].agg(["sum", "size"]).reindex(yk.index.get_level_values("s")).to_numpy()
            sbase = sbase - ys
        prior_share = sbase[:, 0] / np.maximum(sbase[:, 1], 1)
        share = (base["sum"].to_numpy() + shrink * prior_share) / (base["size"].to_numpy() + shrink)
        e = float((share * yk["size"].to_numpy()).sum())
        o = float(yk["sum"].sum())
        lo, hi = poisson_ci(o)
        out.append({"year": y, "n": int(len(g)), "observed": o, "expected": e,
                    "ratio": o / e if e > 0 else np.nan,
                    "lo": float(lo) / e if e > 0 else np.nan, "hi": float(hi) / e if e > 0 else np.nan})
    return pd.DataFrame(out).set_index("year")


def spearman_boot(x, y, n_boot: int = 4000, seed: int = 0) -> dict:
    """Spearman rho with a percentile bootstrap interval (pairs resampled) and its p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4:
        return {"n": n, "rho": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    res = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        rhos.append(stats.spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {"n": n, "rho": float(res.statistic), "lo": float(lo), "hi": float(hi), "p": float(res.pvalue)}


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for rank, i in enumerate(order):
        run = min(run, p[i] * m / (m - rank))
        adj[i] = run
    return adj


def auc_score(score, label, weight=None) -> float:
    """ROC AUC (Mann-Whitney, ties count half); optional case weights. NaN if one class is missing."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    w = np.ones(len(s)) if weight is None else np.asarray(weight, float)
    ok = np.isfinite(s)
    s, y, w = s[ok], y[ok], w[ok]
    if not y.any() or y.all():
        return np.nan
    if weight is None:
        r = stats.rankdata(s)
        n1 = int(y.sum())
        return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    # weighted ranks with ties: cumulative negative weight below each distinct score
    uniq, start = np.unique(s, return_index=True)
    neg = np.where(y, 0.0, w)
    pos = np.where(y, w, 0.0)
    neg_u = np.add.reduceat(neg, start)
    pos_u = np.add.reduceat(pos, start)
    below = np.cumsum(neg_u) - neg_u
    num = float((pos_u * (below + 0.5 * neg_u)).sum())
    return num / float(pos_u.sum() * neg_u.sum())


def auc_by_group(score, label, group, min_pos: int = 1) -> pd.Series:
    """AUC within each group (NaN where a class is missing or fewer than ``min_pos`` positives)."""
    df = pd.DataFrame({"s": np.asarray(score, float), "y": np.asarray(label, bool), "g": np.asarray(group)})
    out = {}
    for g, d in df.groupby("g"):
        out[g] = auc_score(d["s"], d["y"]) if d["y"].sum() >= min_pos else np.nan
    return pd.Series(out)


def stratified_auc(score, label, strata) -> float:
    """AUC computed only within strata (pairs of cell-days in the same stratum), pooled by pair count.

    Removes anything the stratum variable explains, e.g. the number of
    records in a cell-day or the month.
    """
    df = pd.DataFrame({"s": np.asarray(score, float), "y": np.asarray(label, bool), "g": np.asarray(strata)})
    num = den = 0.0
    for _, d in df.groupby("g"):
        n1, n0 = int(d["y"].sum()), int((~d["y"]).sum())
        if n1 == 0 or n0 == 0:
            continue
        num += auc_score(d["s"], d["y"]) * n1 * n0
        den += n1 * n0
    return num / den if den else np.nan


def year_bootstrap(fn, years, n_boot: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, lo, hi) of ``fn(mask)``; ``mask`` selects rows of whole years resampled with replacement.

    ``fn`` receives an integer index array (rows may repeat).
    """
    years = np.asarray(years)
    ys = np.unique(years)
    idx = {y: np.flatnonzero(years == y) for y in ys}
    est = fn(np.arange(len(years)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ys, len(ys))
        v = fn(np.concatenate([idx[y] for y in pick]))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


# =========================================================================== calendar features
SYNODIC_MONTH = 29.530588853
_NEW_MOON_JD = 2451550.26          # 2000-01-06 18:14 UTC


def moon_phase(dates) -> np.ndarray:
    """Lunar phase as a fraction of the synodic month at noon of each date: 0 = new, 0.5 = full.

    Mean-motion approximation (reference new moon 2000-01-06 18:14 UTC); the true
    phase differs by up to ~0.6 day (0.02 of the cycle), which does not matter for
    a folk-belief test with +-3 day wide classes.
    """
    d = pd.DatetimeIndex(np.ravel(dates)).normalize()
    jd = d.to_julian_date().to_numpy() + 0.0     # midnight JD (pandas gives .5 steps at midnight)
    jd = jd + 0.5                                # noon
    return np.mod((jd - _NEW_MOON_JD) / SYNODIC_MONTH, 1.0)


def moon_illumination(phase) -> np.ndarray:
    """Illuminated fraction 0..1 from the phase fraction."""
    return 0.5 * (1 - np.cos(2 * np.pi * np.asarray(phase, float)))


def daylength_hours(doy, lat) -> np.ndarray:
    """Hours between sunrise and sunset (standard declination formula)."""
    doy, lat = np.asarray(doy, float), np.deg2rad(np.asarray(lat, float))
    dec = np.deg2rad(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.0)
    x = np.clip(-np.tan(lat) * np.tan(dec), -1, 1)
    return 24.0 / np.pi * np.arccos(x)


# =========================================================================== weather at cells
LAPSE_TEMP_VARS = ("tmean", "tmin", "tmax", "stl1")


def cell_weather(lon, lat, elev, ds=None) -> tuple[pd.DatetimeIndex, dict]:
    """ERA5 daily weather at points: (times, {var: (T, N) float32}).

    Same recipe as ``predict`` / ``features.point_weather``: bilinear
    interpolation of the 0.25 deg ERA5 history, temperatures shifted by the
    lapse rate from the interpolated ERA5 orography to ``elev`` (m); soil
    temperature ``stl1`` is shifted too. Gaps are filled like ``predict.prepare_weather``.
    """
    from .. import features, predict
    from ..sources import weather
    ds = predict._era5_history() if ds is None else ds
    co = predict.prepare_weather(ds)
    fg = predict.FineGrid(np.asarray(lon, float), np.asarray(lat, float), np.asarray(elev, float))
    w = predict._bilinear_weights(co.lat, co.lon, fg)
    dz = (fg.elev - predict.bilinear(co.orog, w)) * weather.LAPSE_RATE
    out = {}
    for v in features.WEATHER_VARS:
        a = predict.bilinear(co.w[v], w)
        if v in LAPSE_TEMP_VARS:
            a = a + dz.astype("float32")
        out[v] = a.astype("float32")
    return co.times, out


def _shift(x, k):
    """x delayed by k steps along time (first k rows repeat x[0])."""
    if k <= 0:
        return x
    out = np.empty_like(x)
    out[k:] = x[:-k]
    out[:k] = x[0]
    return out


def _wet_runs(precip, wet_mm: float = 1.0):
    """Total (mm) and length (days) of the wet spell that ends on each day (0 on dry days)."""
    wet = precip >= wet_mm
    tot = np.zeros(precip.shape, "float64")
    ln = np.zeros(precip.shape, "float64")
    cur = np.zeros(precip.shape[1:], "float64")
    cl = np.zeros(precip.shape[1:], "float64")
    for t in range(precip.shape[0]):
        cur = np.where(wet[t], cur + precip[t], 0.0)
        cl = np.where(wet[t], cl + 1.0, 0.0)
        tot[t], ln[t] = cur, cl
    return tot, ln


def _days_since(mask):
    """Days since ``mask`` was last true (0 on the day; capped at the length of the series)."""
    out = np.zeros(mask.shape, "float64")
    cur = np.full(mask.shape[1:], float(mask.shape[0]))
    for t in range(mask.shape[0]):
        cur = np.where(mask[t], 0.0, cur + 1.0)
        out[t] = cur
    return out


def extra_features(w: dict, dates, lat=46.0) -> dict:
    """Candidate trend features from daily point weather (time first, trailing windows).

    ``w`` as ``features.WEATHER_VARS`` arrays (T, N); ``dates`` (T,). Every feature
    is known on the day itself (nothing looks ahead). Names are documented in
    ``CANDIDATES``.
    """
    from . import prior
    rm, rs, rmax, rmin = prior._rolling_mean, prior._rolling_sum, prior._rolling_max, prior._rolling_min
    f64 = {k: np.asarray(v, "float64") for k, v in w.items()}
    tm, tmin, tmax, pr = f64["tmean"], f64["tmin"], f64["tmax"], f64["precip"]
    sw, st = f64["swvl1"], f64["stl1"]
    T, N = tm.shape
    f = {}
    # cold snap after a warm spell: mean of the last 5 days minus the mean of the 10 days before them
    drop5 = rm(tm, 5) - _shift(rm(tm, 10), 5)
    f["drop5"] = drop5
    f["drop5_min10"] = rmin(drop5, 10)
    f["drop5_min21"] = rmin(drop5, 21)
    for x in (3, 5):
        f[f"coldsnap{x}_21"] = (f["drop5_min21"] <= -x).astype("float64")
    f["t_trend"] = rm(tm, 7) - rm(tm, 20)
    f["tmin_min14"] = rmin(tmin, 14)
    f["dtr14"] = rm(tmax - tmin, 14)
    f["hot_days21"] = rs((tmax >= 28.0).astype("float64"), 21)
    # rain: how it fell rather than how much
    p26 = rs(pr, 26)
    wet = (pr >= 1.0).astype("float64")
    f["rain_days26"] = rs(wet, 26)
    f["intensity26"] = p26 / np.maximum(f["rain_days26"], 1.0)
    f["maxday26"] = rmax(pr, 26)
    tot, ln = _wet_runs(pr)
    best = np.zeros((T, N)); blen = np.zeros((T, N))
    win = 30
    for t in range(T):
        sl = slice(max(0, t - win + 1), t + 1)
        a = tot[sl]
        i = a.argmax(0)
        best[t] = a[i, np.arange(N)]
        blen[t] = ln[sl][i, np.arange(N)]
    f["event_mm30"] = best
    f["event_len30"] = blen
    f["event_int30"] = best / np.maximum(blen, 1.0)
    f["dry_run"] = _days_since(pr >= 1.0)
    f["days_since_10mm"] = np.minimum(_days_since(rs(pr, 3) >= 10.0), 40.0)
    # soil
    f["swvl1_7"] = rm(sw, 7)
    f["swvl1_7_l7"] = _shift(f["swvl1_7"], 7)
    f["swvl1_7_l14"] = _shift(f["swvl1_7"], 14)
    f["swvl_trend"] = f["swvl1_7"] - f["swvl1_7_l14"]
    f["stl1_7"] = rm(st, 7)
    f["stl1_7_l7"] = _shift(f["stl1_7"], 7)
    f["stl1_7_l14"] = _shift(f["stl1_7"], 14)
    # calendar
    d = pd.DatetimeIndex(np.ravel(dates))
    doy = d.day_of_year.to_numpy().astype("float64")
    dl = daylength_hours(doy, lat)
    ph = moon_phase(d)
    col = lambda a: np.broadcast_to(np.asarray(a, "float64")[:, None], (T, N))
    f["daylength"] = col(dl)
    f["falling_light"] = col((doy > 172).astype("float64"))
    f["doy"] = col(doy)
    f["moon_cos"], f["moon_sin"] = col(np.cos(2 * np.pi * ph)), col(np.sin(2 * np.pi * ph))
    cd = np.minimum(np.abs(ph - 0.5), 1 - np.abs(ph - 0.5))      # distance to full moon (cycle fraction)
    nd = np.minimum(ph, 1 - ph)                                  # distance to new moon
    f["near_full"] = col((cd * SYNODIC_MONTH <= 2.0).astype("float64"))
    f["near_new"] = col((nd * SYNODIC_MONTH <= 2.0).astype("float64"))
    return f


# =========================================================================== logistic regression
def fit_logit(X, y, ridge: float = 1e-4, n_iter: int = 30, w=None) -> np.ndarray:
    """Logistic regression by IRLS with a small ridge on the slopes; returns [intercept, slopes...]."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    sw = np.ones(len(y)) if w is None else np.asarray(w, float)
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(A.shape[1])
    p0 = np.clip((sw * y).sum() / sw.sum(), 1e-6, 1 - 1e-6)
    beta[0] = np.log(p0 / (1 - p0))
    pen = np.eye(A.shape[1]) * ridge
    pen[0, 0] = 0.0
    for _ in range(n_iter):
        eta = np.clip(A @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        g = A.T @ (sw * (y - p)) - pen @ beta
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + pen
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def predict_logit(beta, X) -> np.ndarray:
    """Probabilities from ``fit_logit`` coefficients."""
    A = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    return 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))


def logloss_each(y, p) -> np.ndarray:
    """Per-row negative log-likelihood."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def standardize(X) -> np.ndarray:
    """Zero mean, unit variance per column (constant columns become 0)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)


def loyo_predict(X, y, years, ridge: float = 1e-4, w=None) -> np.ndarray:
    """Leave-one-year-out held-out probabilities of a logistic regression."""
    X, y, years = np.asarray(X, float), np.asarray(y, float), np.asarray(years)
    out = np.full(len(y), np.nan)
    for yr in np.unique(years):
        te = years == yr
        beta = fit_logit(X[~te], y[~te], ridge, w=None if w is None else w[~te])
        out[te] = predict_logit(beta, X[te])
    return out


def cluster_wald(X, y, groups, cols, ridge: float = 1e-4) -> dict:
    """Coefficients of ``cols`` with a cluster-robust Wald test (clusters = years).

    Returns the slopes of ``cols`` (indices into X), their cluster-robust
    standard errors, the Wald statistic of the block and its p-value from an F
    distribution with (len(cols), G - 1) degrees of freedom.
    """
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    beta = fit_logit(X, y, ridge)
    A = np.column_stack([np.ones(len(y)), X])
    p = predict_logit(beta, X)
    pen = np.eye(A.shape[1]) * ridge
    pen[0, 0] = 0.0
    H = (A * (p * (1 - p))[:, None]).T @ A + pen
    Hinv = np.linalg.inv(H)
    r = y - p
    ug = np.unique(groups)
    S = np.zeros((len(ug), A.shape[1]))
    for i, g in enumerate(ug):
        m = groups == g
        S[i] = A[m].T @ r[m]
    G = len(ug)
    meat = S.T @ S * (G / (G - 1))
    V = Hinv @ meat @ Hinv
    idx = [c + 1 for c in cols]
    b, Vb = beta[idx], V[np.ix_(idx, idx)]
    try:
        wald = float(b @ np.linalg.solve(Vb, b))
    except np.linalg.LinAlgError:
        wald = np.nan
    k = len(idx)
    pval = float(stats.f.sf(wald / k, k, max(G - 1, 1))) if np.isfinite(wald) else np.nan
    return {"coef": b, "se": np.sqrt(np.diag(Vb)), "V": Vb, "wald": wald, "p": pval, "beta": beta}


def deviance_gain(y, X0, X1, ridge: float = 1e-4) -> float:
    """In-sample log-likelihood increase when ``X1`` replaces ``X0`` (both full fits)."""
    ll = lambda X: -logloss_each(y, predict_logit(fit_logit(X, y, ridge), X)).sum()
    return float(ll(X1) - ll(X0))


# =========================================================================== data
INAT = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
OBSORG = "8a863029-f435-446a-821e-275f4f641165"


def target_mask(d: pd.DataFrame, group: str = "macro") -> np.ndarray:
    """Which records stand for "someone was looking at / reporting mushrooms".

    ``fungi`` every fungus (also lichens, moulds, yeasts); ``macro`` the gill and
    pore fungi, boletes, chanterelles, puffballs... (class Agaricomycetes);
    ``myco`` families of mostly mycorrhizal (forest) mushrooms; ``bolete`` the
    order Boletales (the same hunters, the same habitat, the closest relatives).
    Jurcki records are always part of the group.
    """
    if group == "fungi":
        m = np.ones(len(d), bool)
    elif group == "macro":
        m = (d["class"].astype(str) == "Agaricomycetes").to_numpy()
    elif group == "myco":
        m = d["family"].astype(str).isin(MYCORRHIZAL_FAMILIES).to_numpy()
    elif group == "bolete":
        m = (d["order"].astype(str) == "Boletales").to_numpy()
    else:
        raise ValueError(f"unknown target group {group!r}")
    return m | d["is_jurcki"].to_numpy(bool)


BASIS = ("HUMAN_OBSERVATION",)
MIN_RECORDERS = 30
OMG = "71d34a18-4dad-4d2d-9cb3-c1e91317faad"       # Austrian Mycological Society mapping


def load_fungi(path=FUNGI, group: str = "macro", countries=None, min_year: int = MIN_YEAR,
               in_season_only: bool = False, basis=BASIS, min_recorders: int = MIN_RECORDERS,
               drop_datasets=()) -> pd.DataFrame:
    """The fungi table restricted to a target group, countries, years and basis of record.

    ``basis`` keeps people's observations (iNaturalist, mycological society
    mapping) and drops the specimen / sample / survey rows (forest-health
    monitoring of chestnut blight, ash dieback and elm disease alone are 45 k
    ``OBSERVATION`` rows and say nothing about mushroom hunters); ``None`` keeps all.
    ``min_recorders`` keeps only datasets with that many distinct (hashed) recorders among the
    selected records: opportunistic reporting by many people (iNaturalist, Observation.org, the
    Austrian Mycological Society's mapping) rather than one person's survey or a curated list
    (the other human-observation datasets in the box have 3-15 recorders, one of them 495 records
    and no jurcki at all). ``drop_datasets``: dataset keys to leave out (sensitivity runs).
    ``countries=None`` keeps every country of the bounding box (Slovenia has only
    ~70 dated jurcki since 2010, Austria ~190).
    Adds ``year``, ``doy``, ``week`` and ``source`` (inat / obsorg / other).
    """
    d = pd.read_parquet(path)
    d = d[target_mask(d, group)]
    if basis is not None:
        d = d[d["basisofrecord"].astype(str).isin(basis)]
    if min_recorders:
        nrec = d.groupby(d["datasetkey"].astype(str))["recorder_id"].nunique()
        d = d[d["datasetkey"].astype(str).isin(nrec[nrec >= min_recorders].index)]
    if len(drop_datasets):
        d = d[~d["datasetkey"].astype(str).isin(drop_datasets)]
    if countries is not None:
        d = d[d["countrycode"].astype(str).isin(countries)]
    d = d[d["date"].dt.year >= min_year].copy()
    if in_season_only:
        d = d[in_season(d["date"])]
    d["date"] = d["date"].dt.normalize()
    d["year"] = d["date"].dt.year
    d["doy"] = d["date"].dt.day_of_year
    d["week"] = week_of_year(d["date"])
    ds = d["datasetkey"].astype(str)
    d["source"] = np.where(ds == INAT, "inat", np.where(ds == OBSORG, "obsorg", "other"))
    d["lon"], d["lat"] = d["lon"].astype(float), d["lat"].astype(float)
    return d.reset_index(drop=True)


def assign_cells(lon, lat, km: float = CELL_KM) -> np.ndarray:
    """Id of the ``km`` wide square of the model grid (EPSG:3794) holding each point; -1 outside the grid."""
    from .. import grid
    rows, cols = grid.lonlat_to_rowcol(np.asarray(lon, float), np.asarray(lat, float))
    shape = grid.grid_shape()
    k = max(int(round(km * 1000 / config.GRID_RES)), 1)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    per_row = -(-shape[1] // k)
    return np.where(ok, (rows // k) * per_row + cols // k, -1)


def cell_day_table(d: pd.DataFrame, km: float = CELL_KM) -> pd.DataFrame:
    """One row per (cell, date) with target-group records: n records, k jurcki, label = any jurcki."""
    cid = assign_cells(d["lon"], d["lat"], km)
    t = d.assign(cell=cid)
    t = t[t["cell"] >= 0]
    g = t.groupby(["cell", "date"]).agg(n=("is_jurcki", "size"), k=("is_jurcki", "sum"),
                                         n_inat=("source", lambda s: int((s == "inat").sum()))).reset_index()
    g["label"] = g["k"] > 0
    g["year"] = g["date"].dt.year
    g["src"] = np.where(g["n_inat"] == g["n"], "inat", np.where(g["n_inat"] == 0, "other", "mixed"))
    return g


def cell_table(d: pd.DataFrame, km: float = CELL_KM) -> pd.DataFrame:
    """Location (mean of the records) and elevation (median DEM height of the records) per cell."""
    from .. import features
    cid = assign_cells(d["lon"], d["lat"], km)
    t = d.assign(cell=cid)
    t = t[t["cell"] >= 0].copy()
    z = features.sample_static(t["lon"].to_numpy(), t["lat"].to_numpy(), ["elev"])["elev"].to_numpy()
    t["z"] = z
    c = t.groupby("cell").agg(lon=("lon", "mean"), lat=("lat", "mean"), elev=("z", "median"),
                              n_records=("z", "size"),
                              country=("countrycode", lambda s: s.astype(str).mode().iat[0]))
    miss = ~np.isfinite(c["elev"])
    if miss.any():          # outside the DEM grid: ERA5 orography
        c.loc[miss, "elev"] = [features.era5_orography(x, y) for x, y in zip(c.loc[miss, "lon"], c.loc[miss, "lat"])]
    return c


SEASON_TABLE = None


def season_of(dates) -> np.ndarray:
    """``prior.season_factor`` of the month of each date."""
    from . import prior
    m = pd.DatetimeIndex(np.ravel(dates)).month.to_numpy()
    tab = np.array([prior.season_factor(i) for i in range(13)])
    return tab[m]


def gather(arr, times, dates, col) -> np.ndarray:
    """arr[(date index), col] for (T, N) ``arr`` and the time index ``times``."""
    ti = times.get_indexer(pd.DatetimeIndex(dates))
    if (ti < 0).any():
        raise ValueError("dates outside the weather history")
    return arr[ti, col]


# =========================================================================== trigger evaluation
SCORES = ("season", "literature", "tuned", "tuned_loyo", "trig_literature", "trig_tuned")


def cellday_scores(cd: pd.DataFrame, cells: pd.DataFrame, times, w: dict, params_by_name: dict,
                   loyo_params_by_year: dict | None = None) -> pd.DataFrame:
    """Season / trigger / season x trigger scores for every cell-day.

    ``params_by_name``: {"literature": PriorParams, "tuned": PriorParams};
    ``loyo_params_by_year``: {year: PriorParams chosen without that year}.
    """
    from .. import features
    col = cells.index.get_indexer(cd["cell"])
    out = pd.DataFrame(index=cd.index)
    season = season_of(cd["date"])
    out["season"] = season
    trig = {}
    for name, p in params_by_name.items():
        t = features.dynamic_feature_arrays(w, p)["trigger"]
        trig[name] = gather(t, times, cd["date"], col)
        out[f"trig_{name}"] = trig[name]
        out[name] = season * trig[name]
    if loyo_params_by_year:
        score = np.full(len(cd), np.nan)
        cache = {}
        for yr, p in loyo_params_by_year.items():
            m = (cd["year"] == yr).to_numpy()
            if not m.any():
                continue
            key = id(p)
            if key not in cache:
                cache[key] = features.dynamic_feature_arrays(w, p)["trigger"]
            score[m] = season[m] * gather(cache[key], times, cd.loc[m, "date"], col[m])
        out["tuned_loyo"] = score
    return out


def evaluate_scores(cd: pd.DataFrame, scores: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """AUC of each score for ``label`` (any jurcki in the cell-day).

    Columns: pooled AUC with a year-bootstrap interval, the mean of the per-year
    AUCs, the AUC within strata of the number of records (so that busy days are not
    simply more likely to hold a jurcki), within month, and within month x record count.
    """
    y = cd["label"].to_numpy(bool)
    nb = np.digitize(cd["n"], [2, 3, 6, 11])           # 1, 2, 3-5, 6-10, 11+
    month = cd["date"].dt.month.to_numpy()
    year = cd["year"].to_numpy()
    strat = month * 100 + nb * 10 + pd.Categorical(cd["src"]).codes if "src" in cd else month * 10 + nb
    cell = cd["cell"].to_numpy()
    rows = []
    for name in scores.columns:
        s = scores[name].to_numpy(float)
        ok = np.isfinite(s)
        est, lo, hi = year_bootstrap(lambda i: auc_score(s[ok][i], y[ok][i]), year[ok], n_boot) \
            if ok.any() else (np.nan,) * 3
        per_year = auc_by_group(s[ok], y[ok], year[ok], min_pos=3)
        rows.append({"score": name, "n": int(ok.sum()), "auc": est, "lo": lo, "hi": hi,
                     "auc_mean_year": float(per_year.mean()),
                     "auc_by_nrec": stratified_auc(s[ok], y[ok], nb[ok]),
                     "auc_within_month": stratified_auc(s[ok], y[ok], month[ok]),
                     "auc_month_nrec": stratified_auc(s[ok], y[ok], month[ok] * 10 + nb[ok]),
                     "auc_month_nrec_src": stratified_auc(s[ok], y[ok], strat[ok]),
                     "auc_within_cell": stratified_auc(s[ok], y[ok], cell[ok])})
    return pd.DataFrame(rows).set_index("score")


def paired_auc_diff(cd: pd.DataFrame, a, b, n_boot: int = 500, seed: int = 0, strata=None) -> tuple:
    """AUC(a) - AUC(b) with a year-bootstrap interval; ``strata`` -> stratified AUCs."""
    y = cd["label"].to_numpy(bool)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    f = (lambda s, i: auc_score(s[i], y[i])) if strata is None else \
        (lambda s, i: stratified_auc(s[i], y[i], strata[i]))
    idx = np.flatnonzero(ok)
    yr = cd["year"].to_numpy()[idx]
    return year_bootstrap(lambda i: f(a, idx[i]) - f(b, idx[i]), yr, n_boot, seed)


def baseline_by_year(path=FUNGI) -> pd.DataFrame:
    """Records per year (all seasons) as the filters are added, and the jurcki among the final ones.

    all fungi -> people's observations only -> Agaricomycetes -> datasets with >= ``MIN_RECORDERS``
    recorders (the baseline used everywhere else). Also the final baseline's Slovenian part.
    """
    raw = pd.read_parquet(path)
    y = raw["date"].dt.year
    obs = raw["basisofrecord"].astype(str).isin(BASIS)
    mac = raw["class"].astype(str) == "Agaricomycetes"
    fin = load_fungi(path)
    out = pd.DataFrame({"all_fungi": y.groupby(y).size(),
                        "human_obs": y[obs].groupby(y[obs]).size(),
                        "agaricomycetes_human_obs": y[obs & mac].groupby(y[obs & mac]).size(),
                        "baseline": fin.groupby("year").size(),
                        "baseline_SI": fin[fin["countrycode"].astype(str) == "SI"].groupby("year").size(),
                        "jurcki_in_baseline": fin[fin["is_jurcki"]].groupby("year").size()}).fillna(0).astype(int)
    out.index.name = "year"
    return out


# =========================================================================== yearly model signal
def forest_sample(n: int = 800, seed: int = 0):
    """Random Slovenian forest cells: (lon, lat, elev, habitat) arrays for the yearly model signal."""
    from .. import grid, predict
    land = grid.load_mask()
    elev = predict.load_elev()
    hab = np.nan_to_num(predict.load_habitat(), nan=0.0)
    from .. import features
    tc = features.load_static_stack(["treecover"])[1][0]
    ok = land & (tc > 0.3) & np.isfinite(elev)
    rr, cc = np.nonzero(ok)
    pick = np.random.default_rng(seed).choice(len(rr), min(n, len(rr)), replace=False)
    rr, cc = rr[pick], cc[pick]
    lon, lat = grid.cell_lonlat()
    return lon[rr, cc], lat[rr, cc], elev[rr, cc].astype(float), hab[rr, cc]


def yearly_model_signal(params, n: int = 800, seed: int = 0, ds=None) -> pd.DataFrame:
    """Per year: mean season x trigger (and habitat x season x trigger) over Slovenian forest.

    Averaged over May 1 - Nov 15 (``all``) and over Aug 15 - Oct 15 (``peak``,
    the weeks of the main flush); days after the end of the weather history are
    left out, so the current year is partial.
    """
    from .. import features
    lon, lat, elev, hab = forest_sample(n, seed)
    times, w = cell_weather(lon, lat, elev, ds)
    trig = features.dynamic_feature_arrays(w, params)["trigger"]
    season = season_of(times)[:, None]
    comb = season * trig
    mo = times.month.to_numpy() * 100 + times.day.to_numpy()
    out = {}
    for name, sel in (("all", in_season(times)), ("peak", (mo >= 815) & (mo <= 1015))):
        out[f"comb_{name}"] = pd.Series(comb[sel].mean(1), index=times[sel]).groupby(times[sel].year).mean()
        out[f"trigger_{name}"] = pd.Series(trig[sel].mean(1), index=times[sel]).groupby(times[sel].year).mean()
        out[f"scored_{name}"] = pd.Series((comb[sel] * hab[None, :]).mean(1), index=times[sel]).groupby(times[sel].year).mean()
    return pd.DataFrame(out)


# =========================================================================== habitat vs target-group background
def target_background(d: pd.DataFrame, sf, land, units: pd.DataFrame, forest_only: bool = True,
                      foreign_buffer_m: float | None = None) -> pd.DataFrame:
    """Background cells at the locations of non-jurcki target-group records (one row per cell).

    Rows have the columns of ``habitat.load_background`` (region, row, col,
    weight); the weight makes each region's background sum to the region's
    presence weight. ``forest_only`` keeps cells with tree cover (the model
    domain), otherwise parks, gardens and fields count as background too.
    Foreign records are kept within ``foreign_buffer_m`` of a foreign find.
    """
    from scipy import ndimage
    from .. import grid
    from . import habitat
    foreign_buffer_m = habitat.FOREIGN_BUFFER_M if foreign_buffer_m is None else foreign_buffer_m
    r, c = grid.lonlat_to_rowcol(d["lon"].to_numpy(), d["lat"].to_numpy())
    shape = sf.shape
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    cells = pd.DataFrame({"row": r[ok], "col": c[ok]}).drop_duplicates()
    if forest_only:
        cells = cells[sf.forest[cells.row.values, cells.col.values]]
    cells["region"] = np.where(land[cells.row.values, cells.col.values], "si", "foreign")
    near = np.zeros(shape, bool)
    fu = units[units.region == "foreign"]
    near[fu.row.values, fu.col.values] = True
    dist = ndimage.distance_transform_edt(~near) * config.GRID_RES
    keep = (cells.region == "si") | (dist[cells.row.values, cells.col.values] <= foreign_buffer_m)
    cells = cells[keep].reset_index(drop=True)
    for region in ("si", "foreign"):
        m = cells.region == region
        cells.loc[m, "weight"] = units.loc[units.region == region, "weight"].sum() / max(int(m.sum()), 1)
    return cells[["region", "row", "col", "weight"]]


def habitat_target_group(non_jurcki: pd.DataFrame, n_bg_si: int = 20000, n_bg_foreign: int = 8000,
                         block_km: float = 10.0, n_folds: int = 5, seed: int = 0,
                         forest_only: bool = True, retrain: bool = True) -> dict:
    """Spatial block CV of the habitat prior / model against a target-group background.

    Returns {"random_bg": ..., "tg_eval": ..., "tg_train_tg_eval": ..., "tg_train_random_eval": ...}
    (nested dicts of ``habitat.cross_validate``): the first is the standard
    test (random forest cells as background), the second keeps the models
    trained on random background but ranks the held-out finds against the
    held-out target-group locations; the last two retrain with the
    target-group locations as the training background.
    """
    from .. import grid
    from . import habitat
    rng = np.random.default_rng(seed)
    sf = habitat.Surfaces()
    land = grid.load_mask()
    units, rows = habitat.load_units(sf.forest, land, rng)
    bg = habitat.load_background(units, sf.forest, land, n_bg_si, n_bg_foreign, rng)
    tg = target_background(non_jurcki, sf, land, units, forest_only=forest_only)
    cells = int(round(block_km * 1000 / config.GRID_RES))
    out = {"n_units": units.groupby(["region", "kind"]).size().to_dict(),
           "n_tg_cells": tg.groupby("region").size().to_dict()}
    out["random_bg"] = habitat.cross_validate(sf, units, rows, bg, land, cells, n_folds, seed)
    out["tg_eval"] = habitat.cross_validate(sf, units, rows, bg, land, cells, n_folds, seed, bg_eval=tg)
    if retrain:
        out["tg_train_tg_eval"] = habitat.cross_validate(sf, units, rows, tg, land, cells, n_folds, seed, bg_eval=tg)
        out["tg_train_random_eval"] = habitat.cross_validate(sf, units, rows, tg, land, cells, n_folds, seed, bg_eval=bg)
    return out


# =========================================================================== trigger parameters without the held-out year
CACHE_DIR = config.INTERIM / "effort"


def loyo_trigger_params(n_candidates: int = 800, seed: int = 0, cache: bool = True) -> dict:
    """{year: PriorParams} chosen by ``trigger.loyo_select`` on the presence-only finds of the *other* years.

    This is the nested leave-one-year-out selection of ``model.trigger`` (same
    candidates, same objective), kept so that the target-group labels of a year
    are scored with parameters that never saw that year's finds. Years without
    finds fall back to the candidate that is best on all years.
    """
    import json
    from . import trigger
    path = CACHE_DIR / f"loyo_choice_{n_candidates}_{seed}.json"
    cands = trigger.sample_params(n_candidates, seed)
    if cache and path.exists():
        chosen = {int(k): int(v) for k, v in json.loads(path.read_text()).items()}
    else:
        finds = trigger.load_finds()
        panel = trigger.build_panel(finds)
        years = panel.series["year"].to_numpy()[panel.finds["sid"].to_numpy()]
        mats = trigger.search(panel, cands)
        cv = trigger.loyo_select(mats, years)
        chosen = {int(k): int(v) for k, v in cv["chosen"].items()}
        chosen[-1] = int(np.argmax(np.nanmean(mats[trigger.PRIMARY], axis=1)))
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(chosen))
    default = chosen.get(-1, 0)
    return {y: cands[chosen.get(y, default)] for y in range(MIN_YEAR, 2027)}


# =========================================================================== exploratory scan of candidate signals
def effect_span(X, cols_coef, V, partial=None, lo_q: float = 0.1, hi_q: float = 0.9):
    """Odds ratio between the 90th and 10th percentile of a candidate's fitted effect, with a delta-method 95 % interval.

    X: standardised candidate columns (n, k); cols_coef/V: their coefficients and covariance.
    For a single linear column this is exp(coef x (q90 - q10)).
    """
    X = np.asarray(X, float)
    b = np.asarray(cols_coef, float)
    part = X @ b
    lo_t, hi_t = np.quantile(part, [lo_q, hi_q])
    lo_m = X[part <= lo_t].mean(0)
    hi_m = X[part >= hi_t].mean(0)
    d = hi_m - lo_m
    est = float(d @ b)
    se = float(np.sqrt(max(d @ V @ d, 0)))
    return float(np.exp(est)), float(np.exp(est - 1.96 * se)), float(np.exp(est + 1.96 * se))


def half_split_gain(X0, X1, y, years, ridge: float = 1e-4) -> tuple[float, float]:
    """Mean held-out log-loss gain (per row) when fitting on even years and testing on odd years, and vice versa."""
    years = np.asarray(years)
    even = years % 2 == 0
    out = []
    for tr in (even, ~even):
        te = ~tr
        b0 = fit_logit(X0[tr], y[tr], ridge)
        b1 = fit_logit(X1[tr], y[tr], ridge)
        ll0 = logloss_each(y[te], predict_logit(b0, X0[te])).mean()
        ll1 = logloss_each(y[te], predict_logit(b1, X1[te])).mean()
        out.append(float(ll0 - ll1))
    return out[0], out[1]


def scan_candidates(base: np.ndarray, cands: dict, y, years, n_boot: int = 2000, seed: int = 0,
                    ridge: float = 1e-4) -> pd.DataFrame:
    """Add each candidate to the baseline logistic model and test it out of sample.

    ``base``: baseline design (n, b), e.g. [season x trigger score, log n records].
    ``cands``: {name: (n, k) array} (k = 1 linear, 2 for a quadratic or a cos / sin pair).
    For every candidate: held-out (leave-one-year-out) log-loss gain per row with a
    year-bootstrap 95 % interval and the share of bootstrap draws <= 0; the change in
    held-out pooled AUC; the cluster-robust (by year) Wald p-value; the odds ratio
    between the 90th and 10th percentile of the fitted effect; and the gain when
    fitting on even years and testing on odd years and vice versa (``rep``: both > 0).
    """
    y = np.asarray(y, float)
    years = np.asarray(years)
    base = standardize(base)
    p0 = loyo_predict(base, y, years, ridge)
    ll0 = logloss_each(y, p0)
    ys = np.unique(years)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ys), (n_boot, len(ys)))      # same year resamples for every candidate
    n_y = np.array([(years == v).sum() for v in ys], float)
    auc0 = auc_score(p0, y)
    rows = []
    for name, C in cands.items():
        C = standardize(C)
        X1 = np.column_stack([base, C])
        p1 = loyo_predict(X1, y, years, ridge)
        gain_y = np.array([(ll0 - logloss_each(y, p1))[years == v].sum() for v in ys])
        boot = gain_y[picks].sum(1) / n_y[picks].sum(1)
        w = cluster_wald(X1, y, years, list(range(base.shape[1], X1.shape[1])), ridge)
        orr, olo, ohi = effect_span(C, w["coef"], w["V"])
        ga, gb = half_split_gain(base, X1, y, years, ridge)
        rows.append({"candidate": name, "k": C.shape[1],
                     "gain": float(gain_y.sum() / n_y.sum()), "gain_lo": float(np.percentile(boot, 2.5)),
                     "gain_hi": float(np.percentile(boot, 97.5)), "p_boot": float((boot <= 0).mean()),
                     "d_auc": float(auc_score(p1, y) - auc0), "coef": float(w["coef"][0]), "se": float(w["se"][0]),
                     "p_wald": w["p"], "or_90_10": orr, "or_lo": olo, "or_hi": ohi,
                     "gain_even_to_odd": ga, "gain_odd_to_even": gb,
                     "rep": bool(ga > 0 and gb > 0)})
    out = pd.DataFrame(rows).set_index("candidate")
    out["p_holm"] = holm(out["p_wald"].to_numpy())
    out["q_bh"] = benjamini_hochberg(out["p_wald"].to_numpy())
    return out


def pseudo_moon_null(base: np.ndarray, dates, y, years, observed_stat: float, n_null: int = 300, seed: int = 0,
                     ridge: float = 1e-4) -> dict:
    """How unusual is the moon's effect compared with "moons" of other periods?

    The statistic is the in-sample log-likelihood gain of adding cos / sin of a
    cycle with a given period. The null distribution uses random periods
    (20-40 days, excluding 28.5-30.5) and random start dates, so it has the same
    smoothness and the same use of the calendar as the real lunar cycle. Returns the
    share of pseudo-moons with a gain at least as large as the real one.
    """
    base = standardize(base)
    y = np.asarray(y, float)
    d = pd.DatetimeIndex(np.ravel(dates))
    days = (d - pd.Timestamp("2000-01-06")).days.to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    X0 = base
    ll0 = -logloss_each(y, predict_logit(fit_logit(X0, y, ridge), X0)).sum()
    stats_ = []
    while len(stats_) < n_null:
        period = rng.uniform(20, 40)
        if 28.5 <= period <= 30.5:
            continue
        ph = np.mod((days - rng.uniform(0, period)) / period, 1.0)
        X1 = np.column_stack([base, np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)])
        stats_.append(-logloss_each(y, predict_logit(fit_logit(X1, y, ridge), X1)).sum() - ll0)
    stats_ = np.asarray(stats_)
    return {"null_median": float(np.median(stats_)), "null_p95": float(np.percentile(stats_, 95)),
            "p": float((1 + (stats_ >= observed_stat).sum()) / (1 + n_null))}


def yearly_anomaly(cd: pd.DataFrame, score, weight_col: str = "n") -> pd.Series:
    """Per year: mean of ``score`` minus its mean in the same week of all years, weighted by records.

    Evaluated on the cell-days where the target group was reported, so it is
    "how much better than usual for the date was the weather where and when
    people were looking" (the same effort-matched sampling as the O/E ratio).
    """
    wk = week_of_year(cd["date"])
    w = cd[weight_col].to_numpy(float)
    df = pd.DataFrame({"y": cd["year"].to_numpy(), "wk": wk, "s": np.asarray(score, float), "w": w})
    df["sw"] = df["s"] * df["w"]
    base = df.groupby("wk")[["sw", "w"]].sum()
    df["anom"] = df["s"] - (base["sw"] / base["w"]).reindex(df["wk"]).to_numpy()
    df["aw"] = df["anom"] * df["w"]
    g = df.groupby("y")[["aw", "w"]].sum()
    return g["aw"] / g["w"]


# =========================================================================== candidate design
YEAR_LEVEL = ("jj_precip_anom", "prev_year_oe")
QUADRATIC = ("swvl1_7", "swvl1_7_l7", "swvl1_7_l14", "stl1_7", "stl1_7_l7", "stl1_7_l14", "elev")
# candidate -> (group, features used)
CANDIDATES = {
    # cold snap after a warm spell
    "drop5": "cold", "drop5_min10": "cold", "drop5_min21": "cold", "coldsnap3_21": "cold",
    "coldsnap5_21": "cold", "t_trend": "cold", "tmin_min14": "cold", "dtr14": "cold", "hot_days21": "heat",
    # how the rain fell
    "rain_days26": "rain", "intensity26": "rain", "maxday26": "rain", "event_mm30": "rain",
    "event_len30": "rain", "event_int30": "rain", "dry_run": "rain", "days_since_10mm": "rain",
    # soil
    "swvl1_7": "soil", "swvl1_7_l7": "soil", "swvl1_7_l14": "soil", "swvl_trend": "soil",
    "stl1_7": "soil", "stl1_7_l7": "soil", "stl1_7_l14": "soil",
    # calendar
    "daylength": "calendar", "falling_light": "calendar",
    "moon_phase": "moon", "near_full": "moon", "near_new": "moon",
    # observers / place
    "weekend": "effort", "frac_inat": "effort", "elev": "place", "elev_x_doy": "place",
    # year level
    "jj_precip_anom": "year", "prev_year_oe": "year",
}


def design_matrices(cd: pd.DataFrame, cells: pd.DataFrame, times, w: dict, score,
                    oe_prev=None, noise: int = 12, seed: int = 0, with_trigger: bool = True):
    """Baseline design and the candidate dictionary for ``scan_candidates``.

    cd: cell-days (cell, date, n, n_inat, src); cells: location table (index = cell id,
    columns lon, lat, elev, country); w: cell weather for ``cells``; score: trigger score
    per cell-day (baseline; left out of the baseline when ``with_trigger`` is False). The
    baseline also holds month dummies, log(number of records), dataset mix and country,
    so the target-group share is judged against a flexible season and the
    observer mix, not the step-function ``season_factor``. ``oe_prev``: {year: O/E
    ratio} for the previous-year candidate. Adds ``noise`` per-date and 3 per-cell-year random features
    as negative controls.
    """
    from . import trigger
    col = cells.index.get_indexer(cd["cell"])
    feats = extra_features(w, times, float(cells["lat"].mean()))
    ti = times.get_indexer(pd.DatetimeIndex(cd["date"]))
    g = {k: v[ti, col] for k, v in feats.items()}
    n = cd["n"].to_numpy(float)
    ctry = cells["country"].to_numpy()[col]
    src = cd["src"].to_numpy()
    month = pd.DatetimeIndex(cd["date"]).month.to_numpy()
    cols = [np.log(n), src == "inat", src == "mixed", ctry == "SI", ctry == "HR"] + \
        [month == m for m in range(6, 12)]
    if with_trigger:
        cols = [np.asarray(score, float)] + cols
    base = np.column_stack(cols).astype(float)
    yr = cd["year"].to_numpy()
    cands = {k: g[k][:, None] for k in feats if k in CANDIDATES}
    cands["moon_phase"] = np.column_stack([g["moon_cos"], g["moon_sin"]])
    cands["weekend"] = trigger.is_off_day(cd["date"].to_numpy())[:, None].astype(float)
    cands["frac_inat"] = (cd["n_inat"].to_numpy(float) / n)[:, None]
    elev = cells["elev"].to_numpy()[col]
    cands["elev"] = elev[:, None]
    z = standardize(elev)[:, 0]
    cands["elev_x_doy"] = (z * standardize(g["doy"])[:, 0])[:, None]
    # June-July precipitation of the cell-year relative to its multi-year mean (z-score over years)
    pr = pd.DataFrame(w["precip"], index=times)
    jj = pr[(pr.index.month.isin([6, 7])) & (pr.index.year >= MIN_YEAR)].groupby(lambda t: t.year).sum()
    jj = (jj - jj.mean()) / jj.std()
    cands["jj_precip_anom"] = np.array([jj.at[y, c] for y, c in zip(yr, col)], float)[:, None] if len(jj) else np.zeros((len(yr), 1))
    cands["jj_precip_anom"] = np.nan_to_num(cands["jj_precip_anom"])
    prev = np.array([np.log(oe_prev.get(y - 1, 1.0)) if oe_prev else 0.0 for y in yr], float)
    cands["prev_year_oe"] = np.nan_to_num(np.clip(prev, -3, 3))[:, None]
    for k in QUADRATIC:
        x = standardize(cands[k])[:, 0]
        cands[k] = np.column_stack([x, x ** 2])
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(cd["date"])
    uniq = dates.unique()
    for i in range(noise):
        v = pd.Series(rng.normal(size=len(uniq)), index=uniq)
        cands[f"noise_date_{i}"] = v.reindex(dates).to_numpy()[:, None]
    cy = pd.Series(range(len(cd))).groupby([cd["cell"].to_numpy(), yr]).ngroup().to_numpy()
    for i in range(3):
        v = rng.normal(size=cy.max() + 1)
        cands[f"noise_cellyear_{i}"] = v[cy][:, None]
    return base, cands


# =========================================================================== elevation through the season, moon, years
def elevation_percentile(elev, month, is_jurcki) -> pd.Series:
    """Elevation of each jurcki record as a percentile of all target-group records of the same month.

    0.5 = as high as the typical record of the group; > 0.5 = higher. Returns the percentile of the jurcki rows
    (index = their positions), so means by month say where jurcki are found relative to where people record fungi.
    """
    df = pd.DataFrame({"z": np.asarray(elev, float), "m": np.asarray(month), "j": np.asarray(is_jurcki, bool)})
    out = pd.Series(np.nan, index=df.index)
    for m, g in df.groupby("m"):
        ref = np.sort(g["z"].dropna().to_numpy())
        jj = g[g["j"]]
        if len(ref) == 0 or jj.empty:
            continue
        lo = np.searchsorted(ref, jj["z"].to_numpy(), "left")
        hi = np.searchsorted(ref, jj["z"].to_numpy(), "right")
        out.loc[jj.index] = (lo + hi) / 2 / len(ref)
    return out[df["j"].to_numpy()]


def elevation_by_month(d: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per month: records, jurcki, median elevation of both, mean percentile of jurcki with a year-bootstrap interval.

    ``d`` needs ``z`` (elevation, m), ``date``, ``is_jurcki`` and ``year``.
    """
    d = d[np.isfinite(d["z"])].reset_index(drop=True)
    month = d["date"].dt.month.to_numpy()
    pct = elevation_percentile(d["z"], month, d["is_jurcki"])
    j = d[d["is_jurcki"]].assign(pct=pct.to_numpy())
    rng = np.random.default_rng(seed)
    years = np.unique(d["year"])
    rows = []
    for m in sorted(set(month)):
        a, b = d[month == m], j[j["date"].dt.month == m]
        if b.empty:
            continue
        boots = []
        by = {y: g["pct"].to_numpy() for y, g in b.groupby("year")}
        ys = np.array(list(by))
        for _ in range(n_boot):
            pick = rng.choice(ys, len(ys))
            v = np.concatenate([by[y] for y in pick])
            boots.append(v.mean())
        rows.append({"month": m, "n": len(a), "n_jurcki": len(b), "median_all": a["z"].median(),
                     "median_jurcki": b["z"].median(), "pct_mean": b["pct"].mean(),
                     "pct_lo": np.percentile(boots, 2.5), "pct_hi": np.percentile(boots, 97.5)})
    return pd.DataFrame(rows).set_index("month")


def moon_table(dates, is_jurcki, bins: int = 8) -> pd.DataFrame:
    """Jurcki share among target-group records by lunar phase bin (0 = new moon, bins/2 = full moon)."""
    ph = moon_phase(dates)
    b = np.minimum((((ph + 0.5 / bins) % 1.0) * bins).astype(int), bins - 1)    # bins centred on new, ..., full
    t = share_table(is_jurcki, b)
    t.index.name = "phase_bin"
    return t


def year_level_table(oe: pd.DataFrame, extra: dict, max_year: int = 2025, n_boot: int = 4000) -> pd.DataFrame:
    """Spearman of the effort-corrected yearly ratio (``oe['ratio']``) with year-level predictors.

    ``extra``: {name: Series indexed by year}; the previous-year ratio and the calendar year are added.
    """
    y = oe.index[oe.index <= max_year]
    r = oe.loc[y, "ratio"]
    items = dict(extra)
    items["previous year's ratio"] = oe["ratio"].shift(1).loc[y]
    items["calendar year"] = pd.Series(y, index=y, dtype=float)
    rows = []
    for name, s in items.items():
        res = spearman_boot(s.reindex(y).to_numpy(float), r.to_numpy(float), n_boot)
        res["predictor"] = name
        rows.append(res)
    return pd.DataFrame(rows).set_index("predictor")[["n", "rho", "lo", "hi", "p"]]


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = True) -> str:
    """Minimal GitHub-markdown table (no extra dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in zip(df.index, df.astype(object).itertuples(index=False)):
        cells = ([str(idx)] if index else []) + [fmt(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =========================================================================== figures
FIG_DIR = config.ROOT / "docs" / "figures"
BLUE, ORANGE, GRAY, INK = "#2f6db5", "#d9822b", "#8a8f98", "#2b2f36"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e8ec",
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white"})
    return plt


def _save(fig, name, fig_dir=FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / name
    fig.savefig(p, dpi=85, bbox_inches="tight")
    return p


def fig_weekly(weekly: pd.DataFrame, fig_dir=FIG_DIR):
    """Weekly jurcki share among target-group records (with Wilson interval) and the record counts."""
    plt = _mpl()
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    x = weekly.index.to_numpy()
    a.fill_between(x, weekly["lo"] * 100, weekly["hi"] * 100, color=BLUE, alpha=0.18, lw=0)
    a.plot(x, weekly["share"] * 100, color=BLUE, lw=1.6, marker="o", ms=3)
    a.set_ylabel("jurcki share of records (%)")
    a.set_title("Jurcki among macrofungi records, by week of year (95 % Wilson interval)", fontsize=9, loc="left")
    b.bar(x, weekly["n"], color=GRAY, width=0.8, label="all macrofungi records")
    b.bar(x, weekly["k"], color=ORANGE, width=0.8, label="jurcki records")
    b.set_ylabel("records"); b.set_xlabel("week of year"); b.legend(frameon=False, fontsize=7.5, loc="upper left")
    return _save(fig, "trends_weekly_share.png", fig_dir)


def fig_years(oe: pd.DataFrame, anomaly: pd.Series, fig_dir=FIG_DIR):
    """Effort-corrected yearly ratio O/E and the model's weather anomaly at the same places and dates."""
    plt = _mpl()
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    x = oe.index.to_numpy()
    a.axhline(1, color=GRAY, lw=1)
    a.errorbar(x, oe["ratio"], yerr=[oe["ratio"] - oe["lo"], oe["hi"] - oe["ratio"]], fmt="o", color=BLUE,
               ms=4, lw=1.2, capsize=2)
    a.set_ylabel("jurcki O/E (1 = usual)")
    a.set_title("Good and poor jurcki years relative to reporting effort (95 % interval)", fontsize=9, loc="left")
    a.set_yscale("log"); a.set_yticks([0.25, 0.5, 1, 2, 4]); a.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    b.axhline(0, color=GRAY, lw=1)
    b.bar(anomaly.index, anomaly.to_numpy(), color=ORANGE, width=0.7)
    b.set_ylabel("trigger anomaly"); b.set_xlabel("year")
    b.set_xticks(list(range(int(x.min()), int(x.max()) + 1, 2)))
    b.set_title("tuned trigger, mean anomaly vs the same week of other years, weighted by records", fontsize=8, loc="left")
    return _save(fig, "trends_year_ratio.png", fig_dir)


def fig_auc(ev: pd.DataFrame, fig_dir=FIG_DIR):
    """AUC of season-only / literature / tuned scores: pooled, and within month x record count x dataset."""
    plt = _mpl()
    names = [("season", "season only"), ("literature", "literature"), ("tuned", "tuned (all years)"),
             ("tuned_loyo", "tuned (leave-year-out)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.38
    xs = np.arange(len(names))
    pooled = [ev.loc[n, "auc"] for n, _ in names]
    err = [[ev.loc[n, "auc"] - ev.loc[n, "lo"] for n, _ in names], [ev.loc[n, "hi"] - ev.loc[n, "auc"] for n, _ in names]]
    ax.bar(xs - w / 2, pooled, w, yerr=err, color=BLUE, capsize=2, label="pooled (year-bootstrap 95 %)", ecolor=INK, error_kw={"lw": 0.8})
    ax.bar(xs + w / 2, [ev.loc[n, "auc_month_nrec_src"] for n, _ in names], w, color=ORANGE,
           label="within month x records x dataset")
    ax.axhline(0.5, color=GRAY, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in names]); ax.set_ylim(0.45, 0.68)
    ax.set_ylabel("AUC for 'any jurcki in cell-day'")
    ax.set_title("Trigger as predictor of P(jurcki | someone reported macrofungi)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return _save(fig, "trends_trigger_auc.png", fig_dir)


def fig_habitat(hab: dict, fig_dir=FIG_DIR):
    """Habitat AUC (Slovenian finds, block CV) against random forest cells vs target-group background."""
    plt = _mpl()
    rows = [("random_bg", "random forest cells"), ("tg_eval", "fungi record locations")]
    models = [("prior", "literature prior"), ("core", "model, core"), ("full", "model, full")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.38
    for i, (key, lab) in enumerate(rows):
        v = [hab[key][m]["si_all"]["auc"] for m, _ in models]
        ax.bar(np.arange(3) + (i - 0.5) * w, v, w, color=[BLUE, ORANGE][i], label=f"background: {lab}")
        for x, y in zip(np.arange(3) + (i - 0.5) * w, v):
            ax.text(x, y + 0.006, f"{y:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.5, color=GRAY, lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in models]); ax.set_ylim(0.4, 0.7)
    ax.set_ylabel("AUC (10 km block CV, Slovenian finds)")
    ax.set_title("Habitat model against two kinds of background", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return _save(fig, "trends_habitat_auc.png", fig_dir)


def fig_elevation(tab: pd.DataFrame, fig_dir=FIG_DIR):
    """Where in the elevation distribution of macrofungi records the jurcki sit, by month."""
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    x = tab.index.to_numpy()
    a.plot(x, tab["median_all"], color=GRAY, marker="o", ms=3, label="all macrofungi records")
    a.plot(x, tab["median_jurcki"], color=BLUE, marker="o", ms=3, label="jurcki records")
    a.set_xlabel("month"); a.set_ylabel("median elevation (m)"); a.legend(frameon=False, fontsize=7.5)
    b.axhline(0.5, color=GRAY, lw=1)
    b.errorbar(x, tab["pct_mean"], yerr=[tab["pct_mean"] - tab["pct_lo"], tab["pct_hi"] - tab["pct_mean"]],
               fmt="o", color=BLUE, ms=4, capsize=2, lw=1.2)
    b.set_xlabel("month"); b.set_ylabel("elevation percentile among macrofungi records"); b.set_ylim(0, 1)
    fig.suptitle("Fruiting elevation through the season, corrected for where people record", fontsize=9, x=0.02, ha="left")
    return _save(fig, "trends_elevation.png", fig_dir)


def fig_scan(res: pd.DataFrame, fig_dir=FIG_DIR):
    """Odds ratio (90th vs 10th percentile of the candidate) with 95 % interval; noise controls in grey."""
    plt = _mpl()
    r = res.sort_values("p_wald")
    r = r[~r.index.str.startswith("trigger_")]
    fig, ax = plt.subplots(figsize=(6.4, 0.16 * len(r) + 0.9))
    y = np.arange(len(r))[::-1]
    for yi, (name, row) in zip(y, r.iterrows()):
        noise = name.startswith("noise")
        col = GRAY if noise else (ORANGE if row["p_holm"] < 0.05 else BLUE)
        ax.plot([row["or_lo"], row["or_hi"]], [yi, yi], color=col, lw=1.1)
        ax.plot(row["or_90_10"], yi, "o", color=col, ms=3.5)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(r.index, fontsize=6.5)
    ax.set_xlabel("odds ratio, 90th vs 10th percentile of the candidate (95 % cluster-robust interval)")
    ax.set_title("Candidate signals beyond season, trigger and observer mix (grey = random noise controls,\n"
                 "orange = survives Holm correction)", fontsize=8, loc="left")
    return _save(fig, "trends_scan.png", fig_dir)


def lgbm_loyo(X, y, years, rounds: int = 150) -> np.ndarray:
    """Leave-one-year-out held-out probabilities of a small, strongly regularised LightGBM model."""
    import lightgbm as lgb
    from .trigger import LGBM_PARAMS
    X, y, years = np.asarray(X, float), np.asarray(y, float), np.asarray(years)
    out = np.zeros(len(y))
    for v in np.unique(years):
        te = years == v
        m = lgb.train(dict(LGBM_PARAMS), lgb.Dataset(X[~te], y[~te]), rounds)
        out[te] = m.predict(X[te])
    return out


def elev_season_heldout(cd: pd.DataFrame, cells: pd.DataFrame, score, base: np.ndarray, strata,
                        n_boot: int = 500) -> dict:
    """Held-out check of the elevation x day-of-year factor (``trigger.ELEV_SEASON``).

    The coefficient is refitted without each year (baseline design ``base`` + z_elev x z_doy, standardisation
    constants of ``trigger.ELEV_SEASON``) and the held-out year's score is multiplied by the clipped factor.
    Reports pooled / stratified / within-cell AUC with and without it, the paired differences with
    year-bootstrap intervals, the leave-country-out log-loss and the per-country AUC.
    """
    from . import trigger
    spec = trigger.ELEV_SEASON
    y = cd["label"].to_numpy(float)
    yr = cd["year"].to_numpy()
    col = cells.index.get_indexer(cd["cell"])
    elev = cells["elev"].to_numpy()[col]
    doy = pd.DatetimeIndex(cd["date"]).day_of_year.to_numpy()
    prod = ((elev - spec["elev_mean"]) / spec["elev_sd"]) * ((doy - spec["doy_mean"]) / spec["doy_sd"])
    Xb = standardize(base)
    X1 = np.column_stack([Xb, prod])
    score = np.asarray(score, float)
    fac = np.ones(len(y))
    coefs = {}
    for v in np.unique(yr):
        te = yr == v
        b = fit_logit(X1[~te], y[~te])[-1]
        coefs[int(v)] = float(b)
        fac[te] = np.clip(np.exp(b * prod[te]), *spec["clip"])
    new = score * fac
    cell = cd["cell"].to_numpy()
    out = {"coef_by_held_out_year": coefs, "coef_all_years": float(fit_logit(X1, y)[-1])}
    for nm, sc_ in (("without", score), ("with", new)):
        out[nm] = {"pooled": auc_score(sc_, y), "stratified": stratified_auc(sc_, y, strata),
                   "within_cell": stratified_auc(sc_, y, cell)}
    out["paired_pooled"] = paired_auc_diff(cd, new, score, n_boot)
    out["paired_stratified"] = paired_auc_diff(cd, new, score, n_boot, strata=np.asarray(strata))
    out["paired_within_cell"] = paired_auc_diff(cd, new, score, n_boot, strata=cell)
    ctry = cells["country"].to_numpy()[col]
    grp = np.where(ctry == "AT", "AT", np.where(ctry == "SI", "SI", "other"))
    out["leave_country_out_logloss"] = {
        nm: float(logloss_each(y, loyo_predict(X, y, grp)).mean()) for nm, X in (("baseline", Xb), ("with factor", X1))}
    out["auc_by_country"] = {c: {"without": auc_score(score[ctry == c], y[ctry == c]),
                                 "with": auc_score(new[ctry == c], y[ctry == c])} for c in ("AT", "SI")}
    return out


# =========================================================================== driver
RESULTS = config.PROCESSED / "effort_results.json"


def _records(df: pd.DataFrame) -> list:
    return json.loads(df.reset_index().to_json(orient="records", date_format="iso"))


def run_all(group: str = "macro", n_boot: int = 300, habitat: bool = True, figures: bool = True,
            fig_dir=FIG_DIR, out=RESULTS) -> dict:
    """Everything in the report: 2a effort-corrected seasonality, 2b trigger, 2c habitat, 2d trend scan."""
    from .. import features
    from . import trigger
    log = print
    tuned = trigger.load_trigger_params()
    with open(trigger.PARAMS_PATH) as f:
        lit = trigger.params_from_dict(json.load(f)["literature"])
    loyo = loyo_trigger_params()
    res: dict = {"group": group}

    # ---- data
    d = load_fungi(group=group, in_season_only=True)
    d["z"] = features.sample_static(d["lon"].to_numpy(), d["lat"].to_numpy(), ["elev"])["elev"].to_numpy()
    res["n_records"], res["n_jurcki"] = int(len(d)), int(d["is_jurcki"].sum())
    res["records_by_country"] = _records(d.groupby(d["countrycode"].astype(str)).agg(
        n=("year", "size"), jurcki=("is_jurcki", "sum")))
    res["records_by_source"] = _records(d.groupby("source").agg(n=("year", "size"), jurcki=("is_jurcki", "sum")))
    res["baseline_by_year"] = _records(baseline_by_year())

    # ---- 2a seasonality and years
    weekly = share_table(d["is_jurcki"], d["week"])
    monthly = share_table(d["is_jurcki"], d["date"].dt.month)
    oe = observed_expected(d["is_jurcki"], d["year"], d["week"], d["source"])
    aut = d[(d["week"] >= 33) & (d["week"] <= 44)]
    oe_aut = observed_expected(aut["is_jurcki"], aut["year"], aut["week"], aut["source"])
    cd = cell_day_table(d)
    cells = cell_table(d)
    cells = cells[cells.index.isin(cd["cell"].unique())]
    times, w = cell_weather(cells["lon"].to_numpy(), cells["lat"].to_numpy(), cells["elev"].to_numpy())
    sc = cellday_scores(cd, cells, times, w, {"literature": lit, "tuned": tuned}, loyo)
    anomaly = {k: yearly_anomaly(cd, sc[k]) for k in ("trig_tuned", "tuned", "trig_literature", "literature")}
    sig = yearly_model_signal(tuned)
    jj = pd.DataFrame(w["precip"], index=times)
    jj = jj[jj.index.month.isin([6, 7])].groupby(lambda t: t.year).sum()
    jj = (jj - jj.mean()) / jj.std()
    cell_col = cells.index.get_indexer(cd["cell"])
    jj_eff = pd.Series({y: float(np.average(jj.loc[y].to_numpy()[cell_col[(cd["year"] == y).to_numpy()]],
                                            weights=cd.loc[cd["year"] == y, "n"]))
                        for y in jj.index if (cd["year"] == y).any()})
    extra = {"model: tuned trigger anomaly at the places/dates of records": anomaly["trig_tuned"],
             "model: tuned season x trigger anomaly": anomaly["tuned"],
             "model: literature trigger anomaly": anomaly["trig_literature"],
             "model: mean season x trigger over Slovenian forest (May-Nov)": sig["comb_all"],
             "model: mean season x trigger over Slovenian forest (15 Aug-15 Oct)": sig["comb_peak"],
             "Jun-Jul rain anomaly (z) at the places of records": jj_eff}
    ytab = year_level_table(oe, extra)
    aut_tab = year_level_table(oe_aut, {"Jun-Jul rain anomaly (z) at the places of records": jj_eff,
                                        "model: tuned trigger anomaly": anomaly["trig_tuned"]})
    res.update(weekly=_records(weekly), monthly=_records(monthly), oe=_records(oe), oe_autumn=_records(oe_aut),
               year_level=_records(ytab), year_level_autumn=_records(aut_tab),
               model_signal=_records(sig))
    log("2a done")

    # ---- 2b trigger
    ev = evaluate_scores(cd, sc, n_boot)
    paired = {f"{a} - {b}": paired_auc_diff(cd, sc[a], sc[b], n_boot)
              for a, b in (("tuned_loyo", "season"), ("literature", "season"), ("tuned_loyo", "literature"))}
    paired_strat = {f"{a} - {b}": paired_auc_diff(cd, sc[a], sc[b], n_boot,
                                                  strata=(cd["date"].dt.month.to_numpy() * 100
                                                          + np.digitize(cd["n"], [2, 3, 6, 11]) * 10
                                                          + pd.Categorical(cd["src"]).codes))
                    for a, b in (("tuned_loyo", "season"), ("literature", "season"), ("tuned_loyo", "literature"))}
    res.update(trigger_eval=_records(ev), paired_auc=paired, paired_auc_stratified=paired_strat,
               n_cellday=int(len(cd)), n_cellday_pos=int(cd["label"].sum()), n_cells=int(len(cells)))
    variants = {}
    for name, kw in {"macro, Slovenia + Austria": dict(countries=("SI", "AT")), "macro, Slovenia only": dict(countries=("SI",)),
                     "mycorrhizal families": dict(group="myco"), "boletes (Boletales)": dict(group="bolete"),
                     "iNaturalist only": dict(src="inat"), "without the Austrian OMG dataset": dict(drop_datasets=(OMG,)),
                     "contrast: all fungi, all records (unfiltered)": dict(group="fungi", basis=None, min_recorders=0),
                     "contrast: macro, no recorder minimum": dict(min_recorders=0)}.items():
        kw = dict(kw)
        src = kw.pop("src", None)
        dv = load_fungi(in_season_only=True, **{"group": group, **kw})
        if src:
            dv = dv[dv["source"] == src]
        cdv = cell_day_table(dv)
        cv = cell_table(dv)
        cv = cv[cv.index.isin(cdv["cell"].unique())]
        tv, wv = cell_weather(cv["lon"].to_numpy(), cv["lat"].to_numpy(), cv["elev"].to_numpy())
        sv = cellday_scores(cdv, cv, tv, wv, {"literature": lit, "tuned": tuned}, loyo)
        e = evaluate_scores(cdv, sv[["season", "literature", "tuned", "tuned_loyo"]], max(n_boot // 2, 50))
        e.insert(0, "variant", name)
        e.insert(1, "n_pos", int(cdv["label"].sum()))
        variants[name] = _records(e)
    res["trigger_variants"] = variants
    log("2b: variants done")

    # re-tuning the trigger on the target-group labels (weather signal only: AUC within month x records x dataset)
    strata = (cd["date"].dt.month.to_numpy() * 100 + np.digitize(cd["n"], [2, 3, 6, 11]) * 10
              + pd.Categorical(cd["src"]).codes)
    cands = trigger.sample_params(400, 1)
    cands[1] = tuned
    S = candidate_scores(cd, cells, times, w, cands)
    nest = nested_loyo_choice(S, cd["label"].to_numpy(), strata, cd["year"].to_numpy())
    ys_, num_, den_ = yearwise_stratified_auc(sc["tuned_loyo"] / sc["season"], cd["label"].to_numpy(), strata,
                                              cd["year"].to_numpy())
    res["retune"] = {"n_candidates": len(cands), "heldout_auc": nest["heldout_auc"], "train_auc": nest["train_auc"],
                     "in_sample_best_auc": nest["in_sample_auc"], "literature_auc": float(nest["per_candidate_auc"][0]),
                     "tuned_all_years_auc": float(nest["per_candidate_auc"][1]),
                     "find_based_loyo_auc": float(num_.sum() / den_.sum()),
                     "distinct_choices": len(set(nest["chosen"].values()))}
    # how much weather can a flexible model add to the effort variables (leave-one-year-out)?
    col_ = cells.index.get_indexer(cd["cell"])
    ti_ = times.get_indexer(pd.DatetimeIndex(cd["date"]))
    dyn = features.dynamic_feature_arrays(w, tuned)
    names_ = [k for k in features.DYNAMIC_NAMES if k != "trigger"]
    D = np.column_stack([dyn[k][ti_, col_] for k in names_])
    eff = np.column_stack([np.log(cd["n"]), cd["src"] == "inat", cd["src"] == "mixed",
                           cells["country"].to_numpy()[col_] == "SI", cells["country"].to_numpy()[col_] == "AT",
                           pd.DatetimeIndex(cd["date"]).day_of_year, cells["elev"].to_numpy()[col_]]).astype(float)
    yl, yy = cd["label"].to_numpy(), cd["year"].to_numpy()
    gb = {}
    for nm, X in (("effort + day of year + elevation", eff), ("+ dynamic weather features", np.column_stack([eff, D]))):
        p_ = lgbm_loyo(X, yl, yy)
        gb[nm] = {"logloss": float(logloss_each(yl, p_).mean()), "auc": auc_score(p_, yl),
                  "auc_within_strata": stratified_auc(p_, yl, strata)}
    res["lgbm_ceiling"] = gb
    log("2b done")

    # ---- 2c habitat
    hab = None
    if habitat:
        nj = load_fungi(group=group)
        nj = nj[~nj["is_jurcki"]]
        hab = habitat_target_group(nj)
        res["habitat"] = {k: {m: {g: v for g, v in s.items() if g in ("si_all", "si_dated", "si_sfi", "foreign_dated")}
                              for m, s in hab[k].items()} for k in
                          ("random_bg", "tg_eval", "tg_train_tg_eval", "tg_train_random_eval")}
        res["habitat"]["n_tg_cells"] = {k: int(v) for k, v in hab["n_tg_cells"].items()}
        nj2 = habitat_target_group(nj, forest_only=False, retrain=False)
        res["habitat_any_cover"] = {m: nj2["tg_eval"][m]["si_all"] for m in nj2["tg_eval"]}
        log("2c done")

    # ---- 2d trends
    trig_loyo = sc["tuned_loyo"] / sc["season"]
    oe_prev = oe["ratio"].to_dict()
    y_lab, years = cd["label"].to_numpy(), cd["year"].to_numpy()
    base_a, cands = design_matrices(cd, cells, times, w, trig_loyo, oe_prev, with_trigger=False)
    cands["trigger_tuned_loyo"] = trig_loyo.to_numpy()[:, None]
    cands["trigger_tuned_all"] = sc["trig_tuned"].to_numpy()[:, None]
    cands["trigger_literature"] = sc["trig_literature"].to_numpy()[:, None]
    groups = {**CANDIDATES, "trigger_tuned_loyo": "trigger", "trigger_tuned_all": "trigger", "trigger_literature": "trigger"}
    scan_a = scan_candidates(base_a, cands, y_lab, years)
    base_b, cands_b = design_matrices(cd, cells, times, w, trig_loyo, oe_prev, with_trigger=True)
    scan_b = scan_candidates(base_b, cands_b, y_lab, years)
    for t in (scan_a, scan_b):
        t.insert(0, "group", [groups.get(k, "noise") for k in t.index])
    k_real = int((~scan_b.index.str.startswith("noise")).sum())
    ph = moon_phase(cd["date"])
    obs = deviance_gain(y_lab, standardize(base_b),
                        np.column_stack([standardize(base_b), np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)]))
    res["elev_season"] = elev_season_heldout(cd, cells, sc["tuned_loyo"], base_b, strata)
    res.update(scan_without_trigger=_records(scan_a), scan_with_trigger=_records(scan_b), n_candidates=k_real,
               n_noise_controls=int(scan_b.index.str.startswith("noise").sum()),
               moon_table=_records(moon_table(d["date"], d["is_jurcki"])),
               moon_pseudo=pseudo_moon_null(base_b, cd["date"], y_lab, years, obs))
    elev_tab = elevation_by_month(d)
    res["elevation"] = _records(elev_tab)
    res["elevation_SI"] = _records(elevation_by_month(d[d["countrycode"].astype(str) == "SI"]))
    log("2d done")

    if figures:
        fig_weekly(weekly, fig_dir); fig_years(oe, anomaly["trig_tuned"], fig_dir); fig_auc(ev, fig_dir)
        fig_elevation(elev_tab, fig_dir); fig_scan(scan_b, fig_dir)
        if hab is not None:
            fig_habitat({k: hab[k] for k in ("random_bg", "tg_eval")}, fig_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    res["_frames"] = dict(weekly=weekly, monthly=monthly, oe=oe, oe_autumn=oe_aut, ytab=ytab, aut_tab=aut_tab, ev=ev,
                          scan_a=scan_a, scan_b=scan_b, elev=elev_tab)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group", default="macro", choices=TARGET_GROUPS)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--no-habitat", action="store_true", help="skip the (slow) habitat block CV")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    r = run_all(a.group, a.n_boot, not a.no_habitat, not a.no_figures)
    print(f"wrote {RESULTS}")
    return r


# =========================================================================== re-tuning the trigger on target-group labels
def candidate_scores(cd: pd.DataFrame, cells: pd.DataFrame, times, w: dict, cands: list) -> np.ndarray:
    """Trigger score of every parameter set in ``cands`` at every cell-day: (n_cands, n_rows) float32."""
    from .. import features
    col = cells.index.get_indexer(cd["cell"])
    out = np.empty((len(cands), len(cd)), "float32")
    for i, p in enumerate(cands):
        out[i] = gather(features.dynamic_feature_arrays(w, p)["trigger"], times, cd["date"], col)
    return out


def yearwise_stratified_auc(score, y, strata, year) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(years, AUC x pairs, pairs) with the AUC taken within (year, stratum) cells and pooled by pair count."""
    score, y, strata, year = (np.asarray(a) for a in (score, y, strata, year))
    ys = np.unique(year)
    num, den = np.zeros(len(ys)), np.zeros(len(ys))
    for j, v in enumerate(ys):
        m = year == v
        df = pd.DataFrame({"s": score[m].astype(float), "y": y[m].astype(bool), "g": strata[m]})
        for _, g in df.groupby("g"):
            n1, n0 = int(g["y"].sum()), int((~g["y"]).sum())
            if n1 and n0:
                num[j] += auc_score(g["s"], g["y"]) * n1 * n0
                den[j] += n1 * n0
    return ys, num, den


def nested_loyo_choice(scores: np.ndarray, y, strata, year) -> dict:
    """Leave-one-year-out selection of the candidate with the best (year x stratum) AUC on the other years.

    Returns the chosen candidate per held-out year, the pooled held-out AUC of those
    choices, the mean training AUC of the chosen ones and the best candidate in sample.
    """
    cols = [yearwise_stratified_auc(s, y, strata, year) for s in scores]
    ys = cols[0][0]
    num = np.stack([c[1] for c in cols])
    den = cols[0][2]
    chosen, hn, hd, train = {}, 0.0, 0.0, []
    for j, v in enumerate(ys):
        keep = np.arange(len(ys)) != j
        tr = num[:, keep].sum(1) / max(den[keep].sum(), 1e-9)
        c = int(np.argmax(tr))
        chosen[int(v)] = c
        train.append(float(tr[c]))
        hn += num[c, j]
        hd += den[j]
    return {"chosen": chosen, "heldout_auc": float(hn / hd), "train_auc": float(np.mean(train)),
            "in_sample_best": int(np.argmax(num.sum(1))), "in_sample_auc": float(num.sum(1).max() / den.sum()),
            "per_candidate_auc": num.sum(1) / den.sum()}


if __name__ == "__main__":
    main()
