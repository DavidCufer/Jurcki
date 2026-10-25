"""Presence / ABSENCE backtest of the score against forum field reports.

Ground truth: ``data/processed/field_reports.parquet`` (gobe.si forum, labelled by a single LLM labeler).
A report says what a person found on an outing, including "nothing" (``jurcki_outcome == "none"``), which
GBIF presence-only finds never do.  Positive = found / few / many, negative = none; ``unclear`` and
``not_mentioned`` are dropped.

Unit = report.  Cross-posts are removed first (``dedupe``: same author, same outcome, within +-1 day, same
place -- the author hash is read from the local ``web_posts.parquet`` when it exists, otherwise reports are
only compared by place name).  Reports before the first day of the ERA5 history (2010-01-01) cannot be scored.

What is scored
--------------
For every report the model is evaluated on the 250 m grid with the production machinery
(``predict``: ERA5 daily weather -> coarse features -> bilinear interpolation to the cell + lapse-rate shift ->
``prior.trigger_score``) and averaged over the sampled forest cells (``mask & habitat > 0.05``) inside the
report's location ellipse / circle.  Three scopes are available:

* ``cell``      the report's own ellipse (located reports only);
* ``region``    the statistical region (nearest centre, as in ``reports.region_of``) of a located report;
* ``national``  the mean over a random sample of Slovenian forest cells on that date -- works for *every*
  report, also the ~40 % without coordinates.

Scores (all multiplicative as in ``predict.raw_score``)
* ``season``       month step table only
* ``comb_lit``     season x literature trigger     * ``comb_tuned``  season x tuned trigger (production)
* ``hab``          habitat alone                   * ``hab_season``  habitat x season (no weather at all)
* ``full_tuned``   habitat x season x tuned trigger = the production score (before the fixed rescaling)
* ``full_lit``     same with the literature trigger
* ``*_es``         the same times ``trigger.elev_season_factor`` (experimental, off in production)

Tolerance ``tol`` = 0, 1 or 3 days: the cell score is the maximum over the day +-tol (as in ``model.trigger``),
then averaged over the cells.

Metrics (``auc``, cluster bootstrap over year-weeks / authors / years): pooled AUC positive-vs-none, per-year
AUC, within-week AUC (pairs of reports of the same ISO week: tests the *spatial* part), within-place AUC (pairs
of the same statistical region: tests the *timing* part), calibration by score decile.

Retuning (``retune``): the ``trigger.sample_params`` search on the forum presence/absence data with
leave-one-year-out selection, alone and jointly with the GBIF finds, and a monthly-elevation-optimum habitat
factor.

CLI::

    python -m jurcki.model.backtest_reports [--quick] [--no-retune] [--no-figures]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .. import config, grid
from . import prior

REPORTS = config.PROCESSED / "field_reports.parquet"
POSTS = config.PROCESSED / "web_posts.parquet"
RESULTS = config.PROCESSED / "backtest_reports_results.json"
PARAMS_FORUM = config.PROCESSED / "trigger_params_forum.json"
CACHE_DIR = config.INTERIM / "backtest"
FIG_DIR = config.ROOT / "docs" / "figures"

OUTCOME_Y = {"found": 1, "few": 1, "many": 1, "none": 0}
MIN_YEAR = 2010
OFFSETS = tuple(range(-3, 4))
TOLS = (0, 1, 3)
HAB_MIN = 0.05                  # forest cells = mask & habitat > HAB_MIN (as in the app / predict.build_reference)
TRIGGER_FEATURES = ("t20", "p26", "pulse", "tmax10", "tmin7")
TEMP_FEATURES = ("t20", "tmax10", "tmin7")


# ===================================================================================================
# reports
# ===================================================================================================
def load_reports(path=REPORTS, min_year: int = MIN_YEAR, last_day=None, kinds=None,
                  posts_path=POSTS) -> pd.DataFrame:
    """Reports with a presence / absence outcome in the scorable period.

    Adds ``y`` (1 positive, 0 none), ``date``, ``year``, ``yw`` (ISO year-week id), ``author`` (hashed forum
    author from the local post table, '' when unknown).  ``attrs['n_before']`` counts the usable reports
    dropped because they are older than ``min_year`` (no ERA5 weather), ``attrs['n_after']`` those after ``last_day``.
    """
    d = pd.read_parquet(path)
    d = d[d["jurcki_outcome"].isin(OUTCOME_Y)].copy()
    if kinds is not None:
        d = d[d["report_kind"].isin(kinds)]
    d["y"] = d["jurcki_outcome"].map(OUTCOME_Y).astype(int)
    d["date"] = pd.to_datetime(d["date_observed"]).dt.normalize()
    before = int((d["date"] < pd.Timestamp(f"{min_year}-01-01")).sum())
    after = 0
    keep = d["date"] >= pd.Timestamp(f"{min_year}-01-01")
    if last_day is not None:
        after = int((d["date"] > pd.Timestamp(last_day)).sum())
        keep &= d["date"] <= pd.Timestamp(last_day)
    d = d[keep].reset_index(drop=True)
    d["year"] = d["date"].dt.year
    iso = d["date"].dt.isocalendar()
    d["yw"] = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    d["author"] = ""
    if posts_path is not None and Path(posts_path).exists():
        p = pd.read_parquet(posts_path, columns=["id", "author_hash"])
        m = dict(zip(p["id"], p["author_hash"].fillna("")))
        d["author"] = d["post_id"].map(m).fillna("").astype(str)
    d.attrs.update(n_before=before, n_after=after)
    return d


def dedupe(d: pd.DataFrame) -> pd.DataFrame:
    """Drop cross-posts: same author, same outcome, same place, within one day of the report already kept.

    Without an author id, only reports with a *named* location are compared (same name, same outcome,
    +-1 day): two unlocated posts of the same date cannot be told apart from two people.  ``place`` is the
    location name, unlocated reports match any place of the same author.
    """
    d = d.sort_values(["date", "post_id"]).reset_index(drop=True)
    place = d["loc_name"].fillna("")
    author = d["author"] if "author" in d else pd.Series("", index=d.index)
    keep = np.ones(len(d), bool)
    kept: dict[tuple, list] = {}
    for i in range(len(d)):
        a = author.iat[i]
        pl = place.iat[i]
        if a == "" and pl == "":
            continue                                    # cannot say
        key = (a, d["jurcki_outcome"].iat[i]) if a != "" else ("", d["jurcki_outcome"].iat[i], pl)
        day = d["date"].iat[i]
        dup = False
        for (day0, pl0) in kept.get(key, ()):
            if (day - day0).days > 1:
                continue
            if a != "" and not (pl == "" or pl0 == "" or pl == pl0):
                continue
            dup = True
            break
        if dup:
            keep[i] = False
        else:
            kept.setdefault(key, []).append((day, pl))
    out = d[keep].reset_index(drop=True)
    out.attrs.update(d.attrs)
    out.attrs["n_duplicates"] = int((~keep).sum())
    return out


def author_week_weights(d: pd.DataFrame) -> np.ndarray:
    """1 / (number of reports of the same author in the same ISO week): every author-week counts once.

    Unknown authors (``''``) keep weight 1."""
    a = d["author"].to_numpy(str)
    n = d.groupby([d["author"].where(d["author"] != "", d["post_id"]), d["yw"]])["post_id"].transform("size").to_numpy()
    return np.where(a == "", 1.0, 1.0 / n)


# ===================================================================================================
# cells
# ===================================================================================================
REGION_CENTRES = None


def _region_names_centres():
    from .reports import REGION_CENTRES as rc
    names = list(rc)
    return names, np.array([rc[n] for n in names])      # (lat, lon)


def nearest_region(lon, lat) -> np.ndarray:
    """Statistical region (index into ``_region_names_centres``) by the same nearest-centre rule as ``reports.region_of``."""
    names, c = _region_names_centres()
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    dd = (lat[..., None] - c[:, 0]) ** 2 + ((lon[..., None] - c[:, 1]) * 0.7) ** 2
    return dd.argmin(-1)


@dataclass
class CellTable:
    """The grid cells that are scored: flat index into the 250 m grid and their static attributes."""
    flat: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    elev: np.ndarray
    hab: np.ndarray

    def __len__(self):
        return len(self.flat)


@dataclass
class UnitCells:
    """Sampled cells of each unit (report / region / national): positions into a ``CellTable`` (padded)."""
    pos: np.ndarray        # (U, K) int
    valid: np.ndarray      # (U, K) bool


class Forest:
    """Forest cells of the grid (mask & habitat > HAB_MIN) with coordinates in EPSG:3794 and WGS84."""

    def __init__(self, habitat=None, mask=None, elev=None, lonlat=None):
        from .. import predict
        h = np.nan_to_num(predict.load_habitat() if habitat is None else habitat, nan=0.0)
        m = predict.load_mask() if mask is None else mask
        self.shape = h.shape
        elev = predict.load_elev() if elev is None else elev
        lon, lat = predict._cell_lonlat() if lonlat is None else lonlat
        ok = m & (h > HAB_MIN)
        self.flat = np.flatnonzero(ok.ravel())
        self.hab = h.ravel()[self.flat].astype("float32")
        self.elev = np.nan_to_num(np.asarray(elev, "float32").ravel()[self.flat], nan=0.0)
        self.lon, self.lat = lon.ravel()[self.flat], lat.ravel()[self.flat]
        x, y = grid.cell_xy() if lonlat is None else self._xy(h.shape)
        self.x, self.y = x.ravel()[self.flat], y.ravel()[self.flat]
        self.region = nearest_region(self.lon, self.lat)

    @staticmethod
    def _xy(shape):       # synthetic grids: unit cells of GRID_RES
        r, c = shape
        x = config.GRID_ORIGIN[0] + (np.arange(c) + 0.5) * config.GRID_RES
        y = config.GRID_ORIGIN[1] - (np.arange(r) + 0.5) * config.GRID_RES
        return np.meshgrid(x, y)

    def ellipse_members(self, lon, lat, a_m, b_m=None, az_deg=0.0) -> np.ndarray:
        """Forest-cell numbers inside the ellipse (centre lon/lat, semi-axes in metres, azimuth of the major axis
        clockwise from north); ``b_m=None`` -> circle of radius ``a_m``."""
        from pyproj import Transformer
        tr = Transformer.from_crs(config.CRS_WGS84, config.CRS, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        return self.members_xy(cx, cy, a_m, b_m, az_deg)

    def members_xy(self, cx, cy, a_m, b_m=None, az_deg=0.0) -> np.ndarray:
        """As ``ellipse_members`` with the centre already in EPSG:3794 (x, y)."""
        b_m = a_m if b_m is None or not np.isfinite(b_m) else b_m
        dx, dy = self.x - cx, self.y - cy
        near = (np.abs(dx) <= a_m + 300) & (np.abs(dy) <= a_m + 300)
        idx = np.flatnonzero(near)
        az = math.radians(az_deg if np.isfinite(az_deg) else 0.0)
        east, north = dx[idx], dy[idx]
        u = east * math.sin(az) + north * math.cos(az)
        v = east * math.cos(az) - north * math.sin(az)
        return idx[(u / a_m) ** 2 + (v / b_m) ** 2 <= 1.0]


def report_ellipses(d: pd.DataFrame) -> pd.DataFrame:
    """Per located report: a_m, b_m, az (ellipse if the matcher gave one, else a circle of ``loc_radius_m``)."""
    a = d["loc_ellipse_a_m"].to_numpy(float)
    b = d["loc_ellipse_b_m"].to_numpy(float)
    az = d["loc_ellipse_az_deg"].to_numpy(float)
    r = d["loc_radius_m"].to_numpy(float)
    circ = ~(np.isfinite(a) & np.isfinite(b))
    a = np.where(circ, r, a)
    b = np.where(circ, r, b)
    az = np.where(circ | ~np.isfinite(az), 0.0, az)
    return pd.DataFrame({"a_m": a, "b_m": b, "az": az}, index=d.index)


def sample_units(forest: Forest, d: pd.DataFrame, k: int = 100, k_region: int = 500, k_national: int = 3000,
                 seed: int = 0):
    """Sample cells for every located report (its ellipse), every region and the nation.

    Returns ``(cells, report_units, region_units, national_units, located)`` where ``located`` is a boolean
    array over ``d`` (True: the report has an ellipse containing forest cells); the unit rows of
    ``report_units`` are those of ``d[located]`` in order.  Reports with a location but no forest cell inside
    use the forest cells nearest to the centre (within 2 radii, else they are marked unlocated).
    """
    rng = np.random.default_rng(seed)
    ell = report_ellipses(d)
    has = d["loc_lon"].notna().to_numpy() & np.isfinite(ell["a_m"].to_numpy())
    members, located = [], np.zeros(len(d), bool)
    cx_all, cy_all = np.full(len(d), np.nan), np.full(len(d), np.nan)
    if has.any():
        from pyproj import Transformer
        tr = Transformer.from_crs(config.CRS_WGS84, config.CRS, always_xy=True)
        cx_all[has], cy_all[has] = tr.transform(d["loc_lon"].to_numpy(float)[has], d["loc_lat"].to_numpy(float)[has])
    for i in np.flatnonzero(has):
        e = ell.iloc[i]
        cx, cy = cx_all[i], cy_all[i]
        mem = forest.members_xy(cx, cy, e["a_m"], e["b_m"], e["az"])
        if len(mem) == 0:                                    # no forest in the ellipse: nearest forest cells
            mem = forest.members_xy(cx, cy, 2 * e["a_m"], 2 * e["b_m"], e["az"])
            if len(mem):
                mem = mem[np.argsort((forest.x[mem] - cx) ** 2 + (forest.y[mem] - cy) ** 2)[:30]]
        if len(mem) == 0:
            members.append(None)
            continue
        located[i] = True
        members.append(mem if len(mem) <= k else rng.choice(mem, k, replace=False))
    rep_members = [m for m in members if m is not None]
    reg_members = []
    for g in range(len(_region_names_centres()[0])):
        mem = np.flatnonzero(forest.region == g)
        reg_members.append(rng.choice(mem, min(k_region, len(mem)), replace=False) if len(mem) else mem)
    nat = rng.choice(len(forest.flat), min(k_national, len(forest.flat)), replace=False)
    allmem = np.unique(np.concatenate(rep_members + reg_members + [nat])) if rep_members else np.unique(np.concatenate(reg_members + [nat]))
    cells = CellTable(forest.flat[allmem], forest.lon[allmem], forest.lat[allmem], forest.elev[allmem], forest.hab[allmem])

    def pack(lists):
        K = max((len(m) for m in lists), default=1)
        pos = np.zeros((len(lists), max(K, 1)), np.intp)
        valid = np.zeros_like(pos, bool)
        for u, m in enumerate(lists):
            p = np.searchsorted(allmem, m)
            pos[u, :len(m)], valid[u, :len(m)] = p, True
        return UnitCells(pos, valid)

    return cells, pack(rep_members), pack(reg_members), pack([nat]), located


# ===================================================================================================
# engine: weather -> trigger at (query, day offset, cell)
# ===================================================================================================
class Engine:
    """Production trigger evaluated at arbitrary (day, cell) combinations.

    ``co``: ``predict.Coarse`` (gap-free coarse daily weather, see ``predict.prepare_weather``); ``cells``: a ``CellTable``.
    The interpolation (bilinear on the coarse grid, lapse-rate shift of the temperature features) is that of
    ``predict.iter_fine_features``.
    """

    def __init__(self, co, cells: CellTable):
        from .. import predict
        from ..sources import weather
        self.co, self.cells = co, cells
        fg = predict.FineGrid(cells.lon, cells.lat, cells.elev)
        self.i0, self.j0, self.di, self.dj = predict._bilinear_weights(co.lat, co.lon, fg)
        self.dz = ((cells.elev - predict.bilinear(co.orog, (self.i0, self.j0, self.di, self.dj)))
                   * weather.LAPSE_RATE).astype("float32")
        self.times = co.times

    def coarse_features(self, p: prior.PriorParams) -> dict:
        w = self.co.w
        f = prior.weather_features(w["tmean"], w["tmin"], w["tmax"], w["precip"], p)
        return {k: np.asarray(f[k], "float32") for k in TRIGGER_FEATURES}

    def day_index(self, days) -> np.ndarray:
        """Index into the weather times; days outside the history are clipped to its ends."""
        di = pd.DatetimeIndex(np.ravel(days))
        ti = self.times.get_indexer(di)
        ti = np.where(ti < 0, np.where(di < self.times[0], 0, len(self.times) - 1), ti)
        return ti.reshape(np.shape(days))

    def _gather(self, F, tq, pos):
        """F[t, lat, lon] bilinear at the cells ``pos`` (Q, K) for the time indices ``tq`` (Q, O) -> (Q, O, K)."""
        t = tq[:, :, None]
        i0, j0 = self.i0[pos][:, None, :], self.j0[pos][:, None, :]
        di, dj = self.di[pos][:, None, :], self.dj[pos][:, None, :]
        return (F[t, i0, j0] * (1 - di) * (1 - dj) + F[t, i0, j0 + 1] * ((1 - di) * dj)
                + F[t, i0 + 1, j0] * (di * (1 - dj)) + F[t, i0 + 1, j0 + 1] * (di * dj))

    def trigger(self, feats: dict, p: prior.PriorParams, tq, pos) -> np.ndarray:
        """Fruiting trigger (0..1) at (Q, O, K)."""
        f = {}
        for n in TRIGGER_FEATURES:
            v = self._gather(feats[n], tq, pos)
            if n in TEMP_FEATURES:
                v = v + self.dz[pos][:, None, :]
            f[n] = v
        return np.clip(prior.trigger_score(f, p), 0, 1).astype("float32")

    def evaluate(self, days, units: UnitCells, params: dict, want: list, mults: dict | None = None,
                 offsets=OFFSETS, tols=TOLS, keep0: list | None = None, chunk_cells: int = 3_000_000,
                 feats: dict | None = None) -> dict:
        """Scores of the units (one query per row of ``units``) on ``days`` (datetime64, one per unit).

        ``params``: {key: PriorParams}; ``want`` names (``tols`` variants of each are returned as ``(name, tol)``):
        ``season``, ``hab``, ``hab_season`` and, for each params key k, ``trig_k``, ``comb_k`` (season x trigger),
        ``full_k`` (habitat x season x trigger).  ``mults``: {m: f(elev (Q,1,K), doy (Q,O,1)) -> factor};
        ``comb_k@m`` / ``full_k@m`` are the products with the factor.  ``keep0``: names whose cell-level values
        at offset 0 are kept (returned as ``('cell0', name)`` -> (Q, K)).
        """
        mults = mults or {}
        offsets = np.asarray(offsets)
        i_zero = int(np.flatnonzero(offsets == 0)[0])
        cf = feats or {k: self.coarse_features(p) for k, p in params.items()}
        days = pd.DatetimeIndex(days)
        n = len(days)
        out = {(w, t): np.full(n, np.nan) for w in want for t in tols}
        K = units.pos.shape[1]
        step = max(1, chunk_cells // (len(offsets) * K))
        table = np.array([prior.season_factor(m) for m in range(13)], "float32")
        for lo in range(0, n, step):
            sl = slice(lo, min(n, lo + step))
            dd = (days.values[sl].astype("datetime64[D]")[:, None]
                  + offsets.astype("timedelta64[D]")[None, :])                         # (Q, O)
            ddi = pd.DatetimeIndex(dd.ravel())
            tq = self.day_index(dd)
            pos, valid = units.pos[sl], units.valid[sl]
            season = table[ddi.month.to_numpy()].reshape(dd.shape).astype("float32")  # (Q, O)
            hab = self.cells.hab[pos]                                                  # (Q, K)
            elev = self.cells.elev[pos]
            doy = ddi.day_of_year.to_numpy().reshape(dd.shape)
            trig = {}
            vals = {}
            vals["season"] = lambda: np.broadcast_to(season[:, :, None], (len(season), len(offsets), K))
            vals["hab"] = lambda: np.broadcast_to(hab[:, None, :], (len(season), len(offsets), K))
            vals["hab_season"] = lambda: season[:, :, None] * hab[:, None, :]

            def T(k):
                if k not in trig:
                    trig[k] = self.trigger(cf[k], params[k], tq, pos)
                return trig[k]

            def get(name):
                if name in vals and callable(vals[name]):
                    return vals[name]()
                base, _, m = name.partition("@")
                kind, _, k = base.partition("_")
                if base in ("season", "hab", "hab_season"):
                    x = vals[base]()
                elif kind == "trig":
                    x = T(k)
                elif kind == "comb":
                    x = season[:, :, None] * T(k)
                elif kind == "full":
                    x = season[:, :, None] * T(k) * hab[:, None, :]
                else:
                    raise KeyError(name)
                if m:
                    x = x * mults[m](elev[:, None, :], doy[:, :, None])
                return x

            vm = valid.astype("float32")
            nv = np.maximum(vm.sum(1), 1)
            for w in want:
                x = np.asarray(get(w), "float32")
                if keep0 and w in keep0:
                    out.setdefault(("cell0", w), np.full((n, K), np.nan, "float32"))[sl] = x[:, i_zero, :]
                for t in tols:
                    sel = np.abs(offsets) <= t
                    m = x[:, sel, :].max(1)
                    out[(w, t)][sl] = (m * vm).sum(1) / nv
        return out


class PairPlan:
    """Distinct (day, unit) pairs of a set of queries, so that repeated evaluations (e.g. many parameter sets) share the bookkeeping."""

    def __init__(self, days, unit_ids, units: UnitCells):
        days = pd.DatetimeIndex(days).values.astype("datetime64[D]")
        key = pd.MultiIndex.from_arrays([days, np.asarray(unit_ids)])
        self.codes, uniq = pd.factorize(key)
        self.days = np.array([u[0] for u in uniq], dtype="datetime64[D]")
        uu = np.array([u[1] for u in uniq], dtype=np.intp)
        self.units = UnitCells(units.pos[uu], units.valid[uu])

    def run(self, eng: Engine, params: dict, want: list, **kw) -> dict:
        res = eng.evaluate(self.days, self.units, params, want, **kw)
        return {k: v[self.codes] for k, v in res.items()}


def eval_pairs(eng: Engine, days, unit_ids, units: UnitCells, params: dict, want: list, **kw) -> dict:
    """``Engine.evaluate`` for (day, unit) pairs, computing every distinct pair once.

    ``days`` / ``unit_ids`` are aligned per query; ``units`` holds the cells of each unit id.  Returns the dict of
    ``Engine.evaluate`` aligned with the queries."""
    return PairPlan(days, unit_ids, units).run(eng, params, want, **kw)


def subsample_units(units: UnitCells, k: int, seed: int = 0) -> UnitCells:
    """At most ``k`` random valid cells per unit (fewer if the unit has fewer)."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((len(units.pos), k), np.intp)
    valid = np.zeros((len(units.pos), k), bool)
    for u in range(len(units.pos)):
        cols = np.flatnonzero(units.valid[u])
        if len(cols) > k:
            cols = rng.choice(cols, k, replace=False)
        pos[u, :len(cols)], valid[u, :len(cols)] = units.pos[u, cols], True
    return UnitCells(pos, valid)


def load_engine(cells: CellTable, source=None, first_day="2010-01-01", last_day=None) -> Engine:
    """Engine on the local ERA5 history (``predict.Era5Source``)."""
    from .. import predict
    src = predict.make_source(source or "era5")
    ds = src.load(pd.Timestamp(first_day), pd.Timestamp(last_day) if last_day is not None else
                  pd.Timestamp(predict._era5_history().time.values[-1]))
    return Engine(predict.prepare_weather(ds), cells)


# ===================================================================================================
# statistics
# ===================================================================================================
def auc(score, y, weight=None) -> float:
    """ROC AUC with ties counting half (``effort.auc_score``)."""
    from .effort import auc_score
    return auc_score(score, y, weight)


def stratified_auc_parts(score, y, strata, weight=None) -> tuple[float, float]:
    """(sum of U, number of pairs) over strata: AUC computed only between reports of the same stratum.

    Vectorised (``groupby.rank``); ``weight`` is not supported here (use the unweighted version)."""
    s = pd.Series(np.asarray(score, float))
    y = np.asarray(y, bool)
    g = pd.Series(np.asarray(strata))
    ok = np.isfinite(s.to_numpy())
    s, y, g = s[ok].reset_index(drop=True), y[ok], g[ok].reset_index(drop=True)
    r = s.groupby(g).rank(method="average").to_numpy()
    n1 = pd.Series(y.astype(float)).groupby(g).transform("sum").to_numpy()
    n = pd.Series(np.ones(len(y))).groupby(g).transform("sum").to_numpy()
    n0 = n - n1
    # per row: positives contribute (rank - (n1+1)/2) so that a stratum's U = sum over its positives
    u = np.where(y, r - (n1 + 1) / 2, 0.0)
    pairs = np.where(y, n0, 0.0)
    return float(u.sum()), float(pairs.sum())


def stratified_auc(score, y, strata) -> float:
    u, p = stratified_auc_parts(score, y, strata)
    return u / p if p > 0 else np.nan


def stratum_parts(score, y, strata) -> pd.DataFrame:
    """Per-stratum (U, pairs) so that stratified AUCs can be bootstrapped over strata cheaply."""
    s = np.asarray(score, float)
    y = np.asarray(y, bool)
    g = np.asarray(strata)
    ok = np.isfinite(s)
    df = pd.DataFrame({"s": s[ok], "y": y[ok], "g": g[ok]})
    df["r"] = df.groupby("g")["s"].rank(method="average")
    df["n1"] = df.groupby("g")["y"].transform("sum")
    df["n"] = df.groupby("g")["y"].transform("size")
    df["u"] = np.where(df["y"], df["r"] - (df["n1"] + 1) / 2, 0.0)
    df["p"] = np.where(df["y"], df["n"] - df["n1"], 0.0)
    return df.groupby("g")[["u", "p"]].sum()


def cluster_bootstrap(stat_fn, clusters, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Resample whole clusters with replacement; ``stat_fn(idx, copy)`` -> 1-D array of statistics.

    ``idx`` are the (repeating) row numbers of the resample, ``copy`` the number of the drawn cluster copy of
    each row (so that strata can be kept apart when a cluster is drawn twice).  Returns (n_boot, S)."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[i] for i in pick])
        copy = np.concatenate([np.full(len(groups[i]), c) for c, i in enumerate(pick)])
        out.append(np.atleast_1d(stat_fn(idx, copy)))
    return np.array(out, float)


def ci(boot: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2 * 100
    return np.nanpercentile(boot, a, axis=0), np.nanpercentile(boot, 100 - a, axis=0)


def pooled_table(scores: pd.DataFrame, y, clusters: dict, weight=None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Pooled AUC per score with 95 % cluster-bootstrap intervals, one interval per clustering in ``clusters``."""
    y = np.asarray(y, bool)
    cols = list(scores.columns)
    S = scores.to_numpy(float)
    fn = lambda idx, copy=None: np.array([auc(S[idx, j], y[idx], None if weight is None else np.asarray(weight)[idx])
                                          for j in range(S.shape[1])])
    est = fn(np.arange(len(y)))
    out = pd.DataFrame({"auc": est}, index=cols)
    for name, cl in clusters.items():
        lo, hi = ci(cluster_bootstrap(fn, cl, n_boot, seed))
        out[f"lo_{name}"], out[f"hi_{name}"] = lo, hi
    out.insert(0, "n_pos", int(y.sum()))
    out.insert(1, "n_none", int((~y).sum()))
    return out


def paired_diffs(scores: pd.DataFrame, y, pairs: list, cluster, weight=None, n_boot: int = 1000, seed: int = 0,
                 strata=None) -> pd.DataFrame:
    """AUC(a) - AUC(b) with a cluster-bootstrap interval for each (a, b) in ``pairs``.  ``strata`` -> stratified AUCs."""
    y = np.asarray(y, bool)
    S = {c: scores[c].to_numpy(float) for c in scores}
    st = None if strata is None else pd.factorize(np.asarray(strata))[0].astype(np.int64)

    def one(c, idx, copy):
        if st is None:
            return auc(S[c][idx], y[idx], None if weight is None else np.asarray(weight)[idx])
        return stratified_auc(S[c][idx], y[idx], st[idx])

    def fn(idx, copy=None):
        return np.array([one(a, idx, copy) - one(b, idx, copy) for a, b in pairs])

    est = fn(np.arange(len(y)))
    lo, hi = ci(cluster_bootstrap(fn, cluster, n_boot, seed))
    return pd.DataFrame({"diff": est, "lo": lo, "hi": hi}, index=[f"{a} - {b}" for a, b in pairs])


def wilson(k, n, z: float = 1.96):
    from .effort import wilson_ci
    return wilson_ci(k, n, z)


# ===================================================================================================
# scores of a report set
# ===================================================================================================
BASE_WANT = ["season", "hab", "hab_season", "trig_lit", "trig_tuned", "comb_lit", "comb_tuned", "full_lit",
             "full_tuned", "comb_tuned@es", "full_tuned@es"]
MULTS = {}          # filled lazily (needs trigger); see ``mults()``


def mults() -> dict:
    """Named multiplicative factors on the score: ``es`` = ``trigger.elev_season_factor`` (elevation x day of year)."""
    from . import trigger
    return {"es": lambda elev, doy: trigger.elev_season_factor(elev, doy)}


def production_params() -> dict:
    """{'lit': literature PriorParams, 'tuned': production (``trigger_params.json``) PriorParams}."""
    from . import trigger
    lit = prior.PriorParams()
    try:
        raw = json.loads(Path(trigger.PARAMS_PATH).read_text())
        if "literature" in raw:
            lit = trigger.params_from_dict(raw["literature"])
    except FileNotFoundError:
        pass
    return {"lit": lit, "tuned": trigger.load_trigger_params()}


@dataclass
class ScoreSet:
    """Scores of the reports in ``d``.

    ``nat``: DataFrame (index = reports) for every report; ``cell`` / ``region``: DataFrames for the located
    reports (``located`` is a boolean mask over ``d``); columns are ``(name, tol)`` tuples as MultiIndex.
    ``cell0``: cell-level values at tol 0 for the located reports (name -> (n_located, K)) with ``valid`` and
    the ``elev`` / ``doy`` needed to re-weight them.
    """
    d: pd.DataFrame
    located: np.ndarray
    nat: pd.DataFrame
    cell: pd.DataFrame
    region: pd.DataFrame
    cell0: dict
    valid: np.ndarray
    elev: np.ndarray
    hab: np.ndarray
    params: dict


def score_reports(d: pd.DataFrame, eng: Engine, units, params: dict, want=None, with_cell0=("full_tuned", "comb_tuned"),
                  mult=None) -> ScoreSet:
    """Evaluate ``want`` for every report in the three scopes.  ``units`` = ``(ru, rg, nu, located)`` of ``sample_units``."""
    ru, rg, nu, located = units
    want = list(want or BASE_WANT)
    mult = mults() if mult is None else mult
    dl = d[located]
    cell = eval_pairs(eng, dl["date"].values, np.arange(len(dl)), ru, params, want, mults=mult, keep0=list(with_cell0))
    cell0 = {k[1]: v for k, v in cell.items() if k[0] == "cell0"}
    cell = {k: v for k, v in cell.items() if k[0] != "cell0"}
    nat = eval_pairs(eng, d["date"].values, np.zeros(len(d), int), nu, params, want, mults=mult)
    reg_id = nearest_region(dl["loc_lon"].to_numpy(float), dl["loc_lat"].to_numpy(float))
    reg = eval_pairs(eng, dl["date"].values, reg_id, rg, params, want, mults=mult)

    def frame(res, index):
        f = pd.DataFrame(res, index=index)
        f.columns = pd.MultiIndex.from_tuples(f.columns, names=["score", "tol"])
        return f

    return ScoreSet(d, located, frame(nat, d.index), frame(cell, dl.index), frame(reg, dl.index), cell0,
                    ru.valid, eng.cells.elev[ru.pos], eng.cells.hab[ru.pos], params)


def national_daily(eng: Engine, nu: UnitCells, params: dict, years, months=range(5, 12),
                   want=("season", "comb_lit", "comb_tuned", "full_tuned")) -> pd.DataFrame:
    """Mean score over the national forest sample for every day of ``months`` in ``years`` (tol 0)."""
    days = pd.DatetimeIndex([x for y in years for x in pd.date_range(f"{y}-01-01", f"{y}-12-31")
                             if x.month in months and x <= eng.times[-1] and x >= eng.times[0]])
    res = eval_pairs(eng, days.values, np.zeros(len(days), int), nu, params, list(want), offsets=(0,), tols=(0,))
    return pd.DataFrame({k[0]: v for k, v in res.items()}, index=days)


def national_weather_summary(eng: Engine, nu: UnitCells, years, months=(6, 7, 8, 9, 10)) -> pd.DataFrame:
    """National-mean daily weather (mean over the forest sample's coarse-grid neighbours): rain and temperature
    per month for ``years``; plain bilinear values of the coarse fields (no lapse shift on precipitation)."""
    co = eng.co
    pos = nu.pos[0][nu.valid[0]]
    rows = []
    for var in ("precip", "tmean", "tmax"):
        F = co.w[var]
        T = len(co.times)
        tq = np.arange(T)[:, None]
        v = eng._gather(F, tq, pos[None, :])[:, 0, :].mean(1)
        if var != "precip":
            v = v + eng.dz[pos].mean()
        rows.append(pd.Series(v, index=co.times, name=var))
    df = pd.concat(rows, axis=1)
    df["year"], df["month"] = df.index.year, df.index.month
    df = df[df["year"].isin(list(years)) & df["month"].isin(list(months))]
    return df.groupby(["year", "month"]).agg(precip_mm=("precip", "sum"), tmean=("tmean", "mean"),
                                              tmax=("tmax", "mean")).round(1)


# ===================================================================================================
# metrics on a score view
# ===================================================================================================
SCOPES = ("nat", "nat_loc", "region", "cell")
HEADLINE = ["season", "comb_lit", "comb_tuned", "hab", "hab_season", "full_lit", "full_tuned"]


def view(ss: ScoreSet, scope: str, tol: int = 0):
    """``(reports, scores)`` for one scope: nat (all reports), nat_loc (national scores of the located reports),
    region, cell (located reports).  ``scores`` has one column per score name."""
    if scope == "nat":
        return ss.d, ss.nat.xs(tol, level="tol", axis=1)
    if scope == "nat_loc":
        return ss.d[ss.located], ss.nat[ss.located].xs(tol, level="tol", axis=1)
    if scope == "region":
        return ss.d[ss.located], ss.region.xs(tol, level="tol", axis=1)
    if scope == "cell":
        return ss.d[ss.located], ss.cell.xs(tol, level="tol", axis=1)
    raise ValueError(scope)


def year_table(rep: pd.DataFrame, S: pd.DataFrame, cols=None, years=None) -> pd.DataFrame:
    """Per-year n_pos, n_none and AUC of each score (NaN when a class is missing)."""
    cols = list(cols or S.columns)
    rows = []
    for yr, g in rep.groupby("year"):
        if years is not None and yr not in years:
            continue
        y = g["y"].to_numpy(bool)
        r = {"year": int(yr), "n_pos": int(y.sum()), "n_none": int((~y).sum())}
        for c in cols:
            r[c] = auc(S.loc[g.index, c], y) if y.any() and (~y).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("year")


def strata_labels(rep: pd.DataFrame, kind: str) -> np.ndarray:
    """Comparison groups: ``week`` (ISO year-week: tests *where*), ``year`` (tests where + within-season timing),
    ``region`` (same statistical region over all years: tests *when*, season and year-to-year), ``region_year``
    (same region and year: pure within-season timing)."""
    if kind == "week":
        return rep["yw"].to_numpy()
    if kind == "year":
        return rep["year"].to_numpy()
    reg = rep["region"].fillna("").to_numpy(str)
    if kind == "region":
        return np.where(reg == "", "?", reg)
    if kind == "region_year":
        return np.array([f"{a}|{b}" for a, b in zip(np.where(reg == "", "?", reg), rep["year"])])
    raise ValueError(kind)


def within_table(rep: pd.DataFrame, S: pd.DataFrame, cols=None, kinds=("week", "year", "region", "region_year"),
                 n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """Stratified AUC (pairs inside the same stratum) per score and comparison group, with 95 % cluster-bootstrap
    intervals (clusters: the strata themselves for week / year, years for the region groups).  Long format."""
    cols = list(cols or S.columns)
    y = rep["y"].to_numpy(bool)
    rows = []
    for kind in kinds:
        st = strata_labels(rep, kind)
        if kind in ("region", "region_year"):
            keep = np.array([not str(s).startswith("?") for s in st])
        else:
            keep = np.ones(len(rep), bool)
        yk, stk, yr = y[keep], st[keep], rep["year"].to_numpy()[keep]
        for c in cols:
            sc = S[c].to_numpy(float)[keep]
            est = stratified_auc(sc, yk, stk)
            if kind in ("week", "year"):
                parts = stratum_parts(sc, yk, stk)
                U, P = parts["u"].to_numpy(), parts["p"].to_numpy()
                rng = np.random.default_rng(seed)
                bs = []
                for _ in range(n_boot):
                    pick = rng.integers(0, len(U), len(U))
                    if P[pick].sum() > 0:
                        bs.append(U[pick].sum() / P[pick].sum())
                lo, hi = np.percentile(bs, [2.5, 97.5])
            else:
                code = pd.factorize(stk)[0].astype(np.int64)      # strata span clusters (years): drawn copies are pooled
                fn = lambda idx, copy: stratified_auc(sc[idx], yk[idx], code[idx])
                bs = cluster_bootstrap(fn, yr, n_boot, seed)[:, 0]
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append({"kind": kind, "score": c, "auc": est, "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows)


def calibration(rep: pd.DataFrame, score, n_bins: int = 10) -> pd.DataFrame:
    """Share of positive reports by decile of ``score`` (decile 1 = lowest scores), Wilson 95 % interval."""
    s = pd.Series(np.asarray(score, float), index=rep.index)
    ok = s.notna()
    q = pd.qcut(s[ok].rank(method="first"), n_bins, labels=False) + 1
    df = pd.DataFrame({"bin": q, "y": rep.loc[ok, "y"], "s": s[ok]})
    g = df.groupby("bin").agg(n=("y", "size"), n_pos=("y", "sum"), share_pos=("y", "mean"), score_mean=("s", "mean"),
                              score_lo=("s", "min"), score_hi=("s", "max"))
    lo, hi = wilson(g["n_pos"].to_numpy(), g["n"].to_numpy())
    g["lo"], g["hi"] = lo, hi
    return g


def dose_response(rep: pd.DataFrame, S: pd.DataFrame, cols=None) -> pd.DataFrame:
    """AUC against 'none' for each positive class separately: few, found, many (a useful score should rise)."""
    cols = list(cols or S.columns)
    none = rep["jurcki_outcome"] == "none"
    rows = []
    for cls in ("few", "found", "many"):
        m = none | (rep["jurcki_outcome"] == cls)
        y = (rep.loc[m, "jurcki_outcome"] == cls).to_numpy()
        r = {"class": cls, "n_pos": int(y.sum()), "n_none": int((~y).sum())}
        for c in cols:
            r[c] = auc(S.loc[m, c], y)
        rows.append(r)
    return pd.DataFrame(rows).set_index("class")


def half_month(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return d.month.to_numpy() * 2 + (d.day.to_numpy() > 15)


def year_quality(rep: pd.DataFrame, S: pd.DataFrame, cols, min_n: int = 30, n_boot: int = 2000, seed: int = 0) -> dict:
    """Reported vs predicted season quality by year, both as anomalies against the calendar.

    For every report the positive indicator ``y`` and every score are compared with their mean over *all* reports of the
    same half-month (pooled over years); a year's value is the mean of those residuals over its reports.  So a year is
    "better than a normal one" when its reports at that time of year were more often positive / scored higher than
    the average year.  Spearman correlation across the years with at least ``min_n`` reports, with a bootstrap
    interval over years.
    """
    hm = half_month(rep["date"])
    res = pd.DataFrame({"year": rep["year"].to_numpy(), "y": rep["y"].to_numpy(float) - rep.groupby(hm)["y"].transform("mean").to_numpy()},
                       index=rep.index)
    for c in cols:
        s = S[c].astype(float)
        res[c] = s.to_numpy() - s.groupby(hm).transform("mean").to_numpy()
    g = res.groupby("year")
    tab = g.mean()
    tab.insert(0, "n", g.size())
    tab.insert(1, "share_pos", rep.groupby("year")["y"].mean())
    use = tab[tab["n"] >= min_n]
    rng = np.random.default_rng(seed)
    out = {}
    for c in cols:
        if len(use) < 4 or use[c].nunique() < 2 or use["y"].nunique() < 2:
            out[c] = (np.nan, np.nan, np.nan)
            continue
        rho = stats.spearmanr(use["y"], use[c]).statistic
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(use), len(use))
            if len(set(use["y"].to_numpy()[i])) > 1 and len(set(use[c].to_numpy()[i])) > 1:
                bs.append(stats.spearmanr(use["y"].to_numpy()[i], use[c].to_numpy()[i]).statistic)
        out[c] = (float(rho), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float(rho), np.nan, np.nan)
    return {"table": tab, "n_years": len(use), "spearman": out}


def sensitivity(ss: ScoreSet, cols=("season", "comb_lit", "comb_tuned", "full_tuned"), scope: str = "nat",
                tol: int = 0, top_authors: int = 5) -> pd.DataFrame:
    """Pooled AUC under alternative report sets / weightings."""
    rep, S = view(ss, scope, tol)
    out = {}

    def row(name, mask=None, w=None):
        m = np.ones(len(rep), bool) if mask is None else np.asarray(mask)
        r = rep[m]
        y = r["y"].to_numpy(bool)
        out[name] = {"n_pos": int(y.sum()), "n_none": int((~y).sum()),
                     **{c: auc(S.loc[r.index, c], y, None if w is None else np.asarray(w)[m]) for c in cols}}

    row("all deduplicated reports")
    row("outings only", rep["report_kind"] == "outing")
    row("date uncertainty <= 1 day", rep["date_uncertainty_days"] <= 1)
    if (rep["author"] != "").any():
        row("author-week weighted (each author-week counts once)", None, author_week_weights(rep))
        top = rep["author"][rep["author"] != ""].value_counts().index[:top_authors]
        row(f"without the {top_authors} most active authors", ~rep["author"].isin(top))
    if scope != "nat":
        row("without Podravska (Pohorje / Maribor)", rep["region"].fillna("") != "Podravska")
    row("without 2015", rep["year"] != 2015)
    row("only 2013-2020", rep["year"].between(2013, 2020))
    row("only forum posts of August - October", rep["date"].dt.month.isin([8, 9, 10]))
    return pd.DataFrame(out).T


# ===================================================================================================
# retuning the trigger on presence / absence
# ===================================================================================================
@dataclass
class TuneData:
    """What a candidate parameter set is scored on: located reports (their ellipse), the rest on the national mean."""
    d: pd.DataFrame
    located: np.ndarray
    plan_cell: PairPlan | None
    plan_nat: PairPlan | None

    @property
    def y(self):
        return self.d["y"].to_numpy(bool)


def make_tune_data(d: pd.DataFrame, ru: UnitCells, nu: UnitCells, located: np.ndarray, k_cell: int = 30,
                   k_nat: int = 150, seed: int = 0) -> TuneData:
    dl = d[located]
    plan_cell = PairPlan(dl["date"].values, np.arange(len(dl)), subsample_units(ru, k_cell, seed)) if located.any() else None
    du = d[~located]
    plan_nat = PairPlan(du["date"].values, np.zeros(len(du), int), subsample_units(nu, k_nat, seed + 1)) if len(du) else None
    return TuneData(d, located, plan_cell, plan_nat)


def candidate_matrix(eng: Engine, td: TuneData, cands: list, what: str = "comb", mult=None, log_every: int = 0) -> np.ndarray:
    """Score of every candidate for every report (exact day): (n_cands, n_reports).

    ``what``: ``comb`` (season x trigger) or ``full`` (x habitat) or ``trig``; the located reports use their ellipse
    cells, the others the national sample.  Requires the engine's cells to hold the plans' cells."""
    out = np.empty((len(cands), len(td.d)), "float32")
    name = f"{what}_c"
    kw = dict(offsets=(0,), tols=(0,))
    for i, p in enumerate(cands):
        params = {"c": p}
        feats = {"c": eng.coarse_features(p)}
        if td.plan_cell is not None:
            out[i, td.located] = td.plan_cell.run(eng, params, [name], feats=feats, **kw)[(name, 0)]
        if td.plan_nat is not None:
            out[i, ~td.located] = td.plan_nat.run(eng, params, [name], feats=feats, **kw)[(name, 0)]
        if log_every and (i + 1) % log_every == 0:
            print(f"  candidates {i + 1}/{len(cands)}", flush=True)
    return out


def rank_auc_rows(M: np.ndarray, y: np.ndarray, cols=None) -> np.ndarray:
    """AUC of every row of ``M`` (candidates x reports) against ``y`` restricted to ``cols`` (one pooled AUC per row)."""
    if cols is not None:
        M, y = M[:, cols], y[cols]
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.full(len(M), np.nan)
    r = stats.rankdata(M, axis=1)
    return (r[:, y].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def nested_loyo(M: np.ndarray, y, years, extra=None, extra_years=None, weight_extra: float = 0.5) -> dict:
    """Leave-one-year-out selection of a candidate (row of ``M``) by pooled AUC on the other years.

    ``extra`` (candidates x finds) with ``extra_years``: a second objective already on an AUC-like scale (GBIF
    finds: mean temporal percentile); the selection then maximises ``(1 - w) * forum AUC + w * extra`` on the training years.
    ``extra`` may also be a function ``f(held_out_year) -> (n_candidates,)`` that returns the second objective on the other years.
    Returns the chosen row per held-out year, the training objective and the held-out score of every report.
    """
    y = np.asarray(y, bool)
    years = np.asarray(years)
    chosen, train = {}, {}
    held = np.full(M.shape[1], np.nan)
    for yr in np.unique(years):
        tr = years != yr
        if (y[tr]).all() or (~y[tr]).all():
            continue
        obj = rank_auc_rows(M, y, tr)
        if extra is not None:
            ex = extra(yr) if callable(extra) else np.nanmean(extra[:, extra_years != yr], axis=1)
            obj = (1 - weight_extra) * obj + weight_extra * ex
        c = int(np.nanargmax(obj))
        chosen[int(yr)], train[int(yr)] = c, float(obj[c])
        held[~tr] = M[c, ~tr]
    return {"chosen": chosen, "train": train, "held": held}


def gbif_comb_tol(cands: list, tol: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """GBIF find percentiles of season x trigger for every candidate, the objective of ``model.trigger``
    (temporal percentile of the find day among the same point's season days, +-``tol`` days): (n_cands, n_finds), finds' years."""
    from . import trigger
    finds = trigger.load_finds()
    panel = trigger.build_panel(finds)
    season = trigger.season_scores(panel)
    out = np.empty((len(cands), len(panel.finds)), "float32")
    for i, p in enumerate(cands):
        out[i] = trigger.percentiles(panel, season * trigger.trigger_scores(panel, p), tol)
    years = panel.series["year"].to_numpy()[panel.finds["sid"].to_numpy()]
    return out, years


@dataclass
class TargetGroupData:
    """GBIF target-group cell-days (``model.effort``) with the weather of their cells."""
    cd: pd.DataFrame
    cells: pd.DataFrame
    times: pd.DatetimeIndex
    w: dict
    season: np.ndarray
    y: np.ndarray
    year: np.ndarray
    strata: np.ndarray
    elev: np.ndarray
    doy: np.ndarray


def target_group_data(group: str = "macro") -> TargetGroupData:
    from . import effort
    d = effort.load_fungi(group=group, in_season_only=True)
    cd = effort.cell_day_table(d)
    cells = effort.cell_table(d)
    cells = cells[cells.index.isin(cd["cell"].unique())]
    times, w = effort.cell_weather(cells["lon"].to_numpy(), cells["lat"].to_numpy(), cells["elev"].to_numpy())
    strata = (cd["date"].dt.month.to_numpy() * 100 + np.digitize(cd["n"], [2, 3, 6, 11]) * 10
              + pd.Categorical(cd["src"]).codes)
    return TargetGroupData(cd, cells, times, w, effort.season_of(cd["date"]), cd["label"].to_numpy(bool),
                           cd["year"].to_numpy(), strata, cells["elev"].to_numpy(float)[cells.index.get_indexer(cd["cell"])],
                           cd["date"].dt.day_of_year.to_numpy())


def target_group_matrix(tg: TargetGroupData, cands: list) -> np.ndarray:
    """Trigger of every candidate at every target-group cell-day: (n_cands, n_cellday) float32."""
    from . import effort
    return effort.candidate_scores(tg.cd, tg.cells, tg.times, tg.w, cands)


def gbif_target_group(cands: dict, n_boot: int = 300, group: str = "macro", seed: int = 0) -> pd.DataFrame:
    """The GBIF target-group check of ``model.effort``: P(jurcki | a macrofungus was recorded) at the 25 km cell-day.

    ``cands`` {name: PriorParams}.  Per candidate: pooled AUC of season x trigger with a year-bootstrap interval, and the AUC
    within month x record-count x dataset strata (the weather signal alone) of the trigger.  Also the paired
    difference to the first candidate (same strata) with its interval.
    """
    from .. import features
    from . import effort
    d = effort.load_fungi(group=group, in_season_only=True)
    cd = effort.cell_day_table(d)
    cells = effort.cell_table(d)
    cells = cells[cells.index.isin(cd["cell"].unique())]
    times, w = effort.cell_weather(cells["lon"].to_numpy(), cells["lat"].to_numpy(), cells["elev"].to_numpy())
    names = list(cands)
    T = effort.candidate_scores(cd, cells, times, w, [cands[n] for n in names]).astype(float)
    season = effort.season_of(cd["date"])
    y = cd["label"].to_numpy(bool)
    strata = (cd["date"].dt.month.to_numpy() * 100 + np.digitize(cd["n"], [2, 3, 6, 11]) * 10
              + pd.Categorical(cd["src"]).codes)
    year = cd["year"].to_numpy()
    rows = []
    base_comb, base_trig = season * T[0], T[0]
    for i, n in enumerate(names):
        comb = season * T[i]
        est, lo, hi = effort.year_bootstrap(lambda idx: effort.auc_score(comb[idx], y[idx]), year, n_boot, seed)
        r = {"params": n, "n_cellday": len(cd), "n_pos": int(y.sum()), "auc_comb": est, "auc_comb_lo": lo, "auc_comb_hi": hi,
             "auc_trigger_strat": stratified_auc(T[i], y, strata)}
        if i:
            dc = effort.year_bootstrap(lambda idx: effort.auc_score(comb[idx], y[idx]) - effort.auc_score(base_comb[idx], y[idx]),
                                       year, n_boot, seed)
            ds = effort.year_bootstrap(lambda idx: stratified_auc(T[i][idx], y[idx], strata[idx])
                                       - stratified_auc(base_trig[idx], y[idx], strata[idx]), year, n_boot, seed)
            r.update(diff_comb=dc[0], diff_comb_lo=dc[1], diff_comb_hi=dc[2],
                     diff_strat=ds[0], diff_strat_lo=ds[1], diff_strat_hi=ds[2])
        rows.append(r)
    return pd.DataFrame(rows).set_index("params")


def gbif_elevation_factor(factors: dict, n_boot: int = 300, group: str = "macro", seed: int = 0,
                          params=None) -> pd.DataFrame:
    """GBIF target-group effect of an elevation x day-of-year factor ``f(elev, doy)`` on season x trigger.

    ``factors``: {name: f}; the first row is the unmodified season x trigger (production params).  Reports the pooled AUC, the AUC within
    month x record count x dataset strata, and the paired difference to the unmodified score with year-bootstrap intervals."""
    from . import effort, trigger
    params = params or trigger.load_trigger_params()
    d = effort.load_fungi(group=group, in_season_only=True)
    cd = effort.cell_day_table(d)
    cells = effort.cell_table(d)
    cells = cells[cells.index.isin(cd["cell"].unique())]
    times, w = effort.cell_weather(cells["lon"].to_numpy(), cells["lat"].to_numpy(), cells["elev"].to_numpy())
    T = effort.candidate_scores(cd, cells, times, w, [params])[0].astype(float)
    season = effort.season_of(cd["date"])
    elev = cells["elev"].to_numpy(float)[cells.index.get_indexer(cd["cell"])]
    doy = cd["date"].dt.day_of_year.to_numpy()
    y = cd["label"].to_numpy(bool)
    strata = (cd["date"].dt.month.to_numpy() * 100 + np.digitize(cd["n"], [2, 3, 6, 11]) * 10
              + pd.Categorical(cd["src"]).codes)
    year = cd["year"].to_numpy()
    base = season * T
    rows = []
    for name, f in {"none": None, **factors}.items():
        sc = base if f is None else base * f(elev, doy)
        est, lo, hi = effort.year_bootstrap(lambda idx: effort.auc_score(sc[idx], y[idx]), year, n_boot, seed)
        r = {"factor": name, "auc": est, "lo": lo, "hi": hi, "auc_strat": stratified_auc(sc, y, strata)}
        if f is not None:
            dc = effort.year_bootstrap(lambda idx: effort.auc_score(sc[idx], y[idx]) - effort.auc_score(base[idx], y[idx]), year, n_boot, seed)
            ds = effort.year_bootstrap(lambda idx: stratified_auc(sc[idx], y[idx], strata[idx])
                                       - stratified_auc(base[idx], y[idx], strata[idx]), year, n_boot, seed)
            r.update(diff=dc[0], diff_lo=dc[1], diff_hi=dc[2], diff_strat=ds[0], diff_strat_lo=ds[1], diff_strat_hi=ds[2])
        rows.append(r)
    return pd.DataFrame(rows).set_index("factor")


# ---- monthly elevation optimum -------------------------------------------------------------------
def elev_optimum_factor(elev, doy, o0: float, o1: float, sd: float, floor: float = 0.0, pivot: float = 250.0):
    """Gaussian elevation optimum that moves with the day of year: ``opt = o0 + o1 * (doy - pivot)`` metres (``o1`` m/day);
    ``floor + (1 - floor) * exp(-0.5 ((elev - opt) / sd)^2)``.  With o1 = 0 and floor = 1 it is the identity."""
    opt = o0 + o1 * (np.asarray(doy, float) - pivot)
    g = np.exp(-0.5 * ((np.asarray(elev, float) - opt) / sd) ** 2)
    return floor + (1 - floor) * g


def rule_table_factor(elev, doy, p: prior.PriorParams | None = None):
    """The literature rule of ``prior.habitat_score``: Gaussian around ``elev_opt_by_month`` (sd ``elev_sd``), by calendar month."""
    p = p or prior.PriorParams()
    month = pd.DatetimeIndex(pd.Timestamp("2021-01-01") + pd.to_timedelta(np.asarray(doy).ravel() - 1, unit="D")).month.to_numpy()
    opt = np.array([p.elev_opt_by_month.get(int(m), 700) for m in month], float).reshape(np.shape(doy))
    return np.exp(-0.5 * ((np.asarray(elev, float) - opt) / p.elev_sd) ** 2)


ELEV_GRID = [dict(o0=o0, o1=o1, sd=sd, floor=fl) for o0 in (500, 700, 900, 1100) for o1 in (0, 2, 4, 6, 8)
             for sd in (300, 500, 800) for fl in (0.0, 0.3, 0.6)]


def elevation_candidates(base_cell: np.ndarray, valid: np.ndarray, elev: np.ndarray, doy: np.ndarray, grid_=ELEV_GRID) -> np.ndarray:
    """Report scores (mean over cells) of ``base_cell`` (n, K) times each elevation factor of the grid: (n_grid, n)."""
    out = np.empty((len(grid_), len(base_cell)), "float32")
    vm = valid.astype("float32")
    nv = np.maximum(vm.sum(1), 1)
    for i, g in enumerate(grid_):
        f = elev_optimum_factor(elev, doy[:, None], **g)
        out[i] = (np.nan_to_num(base_cell) * f * vm).sum(1) / nv
    return out


# ===================================================================================================
# retune driver
# ===================================================================================================
def _cache_key(d: pd.DataFrame, n: int, seed: int, extra: str = "") -> str:
    import hashlib
    h = hashlib.md5((",".join(d["post_id"].astype(str)) + f"|{n}|{seed}|{extra}").encode()).hexdigest()[:10]
    return h


def paired_vec(a, b, y, cluster, n_boot: int = 1000, seed: int = 0, strata=None) -> dict:
    """Pooled (or stratified) AUC(a) - AUC(b) with a cluster-bootstrap 95 % interval for two score vectors."""
    ok = np.isfinite(a) & np.isfinite(b)
    f = pd.DataFrame({"a": np.asarray(a, float)[ok], "b": np.asarray(b, float)[ok]})
    t = paired_diffs(f, np.asarray(y)[ok], [("a", "b")], np.asarray(cluster)[ok], n_boot=n_boot, seed=seed,
                     strata=None if strata is None else np.asarray(strata)[ok])
    r = t.iloc[0]
    return {"diff": float(r["diff"]), "lo": float(r["lo"]), "hi": float(r["hi"])}


def retune(d: pd.DataFrame, eng: Engine, td: TuneData, ss: ScoreSet, n_cands: int = 800, seed: int = 0,
           n_boot: int = 1000, cache: bool = True, gbif: bool = True, log=print) -> dict:
    """Re-fit the trigger parameters on forum presence / absence (nested leave-one-year-out), alone and jointly with GBIF.

    Candidates are ``trigger.sample_params(n_cands, seed)`` (literature defaults first) plus the production parameters.
    Objective on the forum side: pooled AUC of season x trigger over all reports (located reports on their ellipse,
    the others on the national mean), exact day.  GBIF side: the mean temporal percentile of ``model.trigger``
    (season x trigger, +-3 days).  The result holds the held-out comparisons and the single all-data pick of each procedure.
    """
    from . import trigger
    prod_p = ss.params["tuned"]
    cands = trigger.sample_params(n_cands, seed) + [prod_p]
    iprod = len(cands) - 1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(d, n_cands, seed)
    f_m = CACHE_DIR / f"forum_comb_{key}.npy"
    if cache and f_m.exists():
        M = np.load(f_m)
    else:
        t0 = time.time()
        M = candidate_matrix(eng, td, cands, "comb", log_every=200)
        log(f"forum candidate matrix {M.shape} in {time.time() - t0:.0f} s")
        if cache:
            np.save(f_m, M)
    y = d["y"].to_numpy(bool)
    year = d["year"].to_numpy()
    loc = td.located
    out = {"n_candidates": n_cands, "n_reports": len(d)}
    G = gy = None
    if gbif:
        g_m, g_y = CACHE_DIR / f"gbif_comb_tol_{n_cands}_{seed}.npy", CACHE_DIR / f"gbif_years_{n_cands}_{seed}.npy"
        if cache and g_m.exists() and g_y.exists():
            G, gy = np.load(g_m), np.load(g_y)
        else:
            G, gy = gbif_comb_tol(cands)
            if cache:
                np.save(g_m, G), np.save(g_y, gy)
    a_forum = rank_auc_rows(M, y)
    out["in_sample"] = {"literature": float(a_forum[0]), "production": float(a_forum[iprod]), "best": float(a_forum[:iprod].max()),
                        "best_index": int(a_forum[:iprod].argmax())}
    if G is not None:
        gm = np.nanmean(G, axis=1)
        out["gbif_find_percentile"] = {"literature": float(gm[0]), "production": float(gm[iprod]),
                                       "best_forum_candidate": float(gm[int(a_forum[:iprod].argmax())]),
                                       "corr_forum_vs_gbif_over_candidates": float(np.corrcoef(a_forum[:iprod], gm[:iprod])[0, 1])}
    clusters = {"week": d["yw"].to_numpy(), "year": year}
    # GBIF target-group cell-days: second objective and held-out check
    tg = target_group_data()
    t_m = CACHE_DIR / f"tg_trigger_{n_cands}_{seed}.npy"
    if cache and t_m.exists():
        Ttg = np.load(t_m)
    else:
        Ttg = target_group_matrix(tg, cands).astype("float32")
        if cache:
            np.save(t_m, Ttg)
    Stg = tg.season[None, :] * Ttg
    tg_all = rank_auc_rows(Stg, tg.y)
    out["target_group_in_sample"] = {"literature": float(tg_all[0]), "production": float(tg_all[iprod]),
                                     "best": float(tg_all[:iprod].max()),
                                     "corr_forum_vs_tg": float(np.corrcoef(a_forum[:iprod], tg_all[:iprod])[0, 1])}
    procs = {"forum_only": (None, 0.0)}
    if G is not None:
        procs.update({"joint_gbif_finds_50": ("finds", 0.5), "joint_gbif_finds_25": ("finds", 0.25)})
    procs.update({"joint_target_group_50": ("tg", 0.5), "joint_target_group_25": ("tg", 0.25)})
    res = {}
    for name, (kind, w) in procs.items():
        if kind is None:
            extra = None
        elif kind == "finds":
            extra = (lambda yr, G=G, gy=gy: np.nanmean(G[:iprod][:, gy != yr], axis=1))
        else:
            extra = (lambda yr: rank_auc_rows(Stg[:iprod], tg.y, tg.year != yr))
        r = nested_loyo(M[:iprod], y, year, extra=extra, weight_extra=w)
        h = r["held"]
        ok = np.isfinite(h)
        entry = {"weight_second_objective": w, "n_distinct_choices": len(set(r["chosen"].values())),
                 "chosen_by_year": {int(k): int(v) for k, v in r["chosen"].items()},
                 "train_objective_mean": float(np.mean(list(r["train"].values()))),
                 "heldout_auc": auc(h[ok], y[ok]), "production_auc": auc(M[iprod][ok], y[ok]),
                 "literature_auc": auc(M[0][ok], y[ok]),
                 "heldout_within_year": stratified_auc(h[ok], y[ok], year[ok]),
                 "production_within_year": stratified_auc(M[iprod][ok], y[ok], year[ok]),
                 "heldout_auc_located": auc(h[ok & loc], y[ok & loc]), "production_auc_located": auc(M[iprod][ok & loc], y[ok & loc]),
                 "heldout_auc_unlocated": auc(h[ok & ~loc], y[ok & ~loc]), "production_auc_unlocated": auc(M[iprod][ok & ~loc], y[ok & ~loc])}
        for cname, cl in clusters.items():
            entry[f"vs_production_{cname}"] = paired_vec(h, M[iprod], y, cl, n_boot)
            entry[f"vs_literature_{cname}"] = paired_vec(h, M[0], y, cl, n_boot)
        # the same choices on the held-out GBIF target-group cell-days
        hc, ht = np.full(len(tg.y), np.nan), np.full(len(tg.y), np.nan)
        for yr, c in r["chosen"].items():
            m = tg.year == yr
            hc[m], ht[m] = Stg[c, m], Ttg[c, m]
        okt = np.isfinite(hc)
        entry["tg_heldout"] = {
            "n_cellday": int(okt.sum()), "auc_comb": auc(hc[okt], tg.y[okt]), "production_auc_comb": auc(Stg[iprod][okt], tg.y[okt]),
            "comb": paired_vec(hc[okt], Stg[iprod][okt], tg.y[okt], tg.year[okt], n_boot),
            "auc_strat": stratified_auc(ht[okt], tg.y[okt], tg.strata[okt]),
            "production_auc_strat": stratified_auc(Ttg[iprod][okt], tg.y[okt], tg.strata[okt]),
            "strat": paired_vec(ht[okt], Ttg[iprod][okt], tg.y[okt], tg.year[okt], n_boot, strata=tg.strata[okt])}
        if kind is None:
            obj = a_forum[:iprod]
        elif kind == "finds":
            obj = (1 - w) * a_forum[:iprod] + w * np.nanmean(G[:iprod], axis=1)
        else:
            obj = (1 - w) * a_forum[:iprod] + w * tg_all[:iprod]
        pick = int(np.nanargmax(obj))
        entry["all_data_pick"] = {"index": pick, "params": trigger.params_to_dict(cands[pick]), "forum_auc_in_sample": float(a_forum[pick]),
                                  "gbif_find_percentile": None if G is None else float(np.nanmean(G[pick])),
                                  "target_group_auc_in_sample": float(tg_all[pick])}
        res[name] = entry
        log(f"{name}: held-out forum {entry['heldout_auc']:.3f} (production {entry['production_auc']:.3f}, literature "
            f"{entry['literature_auc']:.3f}); target group held-out diff {entry['tg_heldout']['comb']['diff']:+.3f}; "
            f"distinct choices {entry['n_distinct_choices']}")
    out["procedures"] = res
    # candidate scatter for the frontier figure
    out["scatter"] = {"forum_auc": a_forum[:iprod].tolist(), "gbif_pct": (np.nanmean(G[:iprod], axis=1).tolist() if G is not None else None),
                      "target_group_auc": tg_all[:iprod].tolist()}
    out["candidates"] = {"literature": trigger.params_to_dict(cands[0]), "production": trigger.params_to_dict(prod_p)}
    out["_cands"] = cands
    out["_M"] = M
    return out


def recommend(rt: dict, target_group: pd.DataFrame, margin: float = 0.01) -> dict:
    """Apply the rule to each procedure's all-data pick.

    (a) held-out forum AUC beats production with a 95 % interval above 0 for both the week and the year bootstraps;
    (b) the GBIF target-group pooled AUC (season x trigger) and within-strata AUC (trigger) are not more than ``margin`` below
    production, both for the leave-one-year-out choices on held-out years (``tg_heldout``) and for the pick itself
    (``target_group`` table, paired differences, in sample)."""
    out = {}
    for name, e in rt["procedures"].items():
        lab = f"pick_{name}"
        ok_a = all(e[f"vs_production_{c}"]["lo"] > 0 for c in ("week", "year"))
        h = e["tg_heldout"]
        ok_h = bool(h["comb"]["diff"] >= -margin and h["strat"]["diff"] >= -margin)
        row = target_group.loc[lab] if lab in target_group.index else None
        ok_p = None if row is None else bool(row["diff_comb"] >= -margin and row["diff_strat"] >= -margin)
        out[name] = {"heldout_gain_ci_excludes_0": bool(ok_a), "target_group_heldout_not_hurt": ok_h,
                     "target_group_pick_not_hurt": ok_p, "recommend": bool(ok_a and ok_h and ok_p)}
    return out


def elevation_test(ss: ScoreSet, n_boot: int = 1000, gbif: bool = True) -> dict:
    """Monthly (day-of-year dependent) elevation optimum as a habitat factor, on the located forum reports.

    Base = production score (cell level, exact day).  Candidates: ``es`` (``trigger.elev_season_factor``, the TRENDS.md
    finding), ``rule_table`` (the month table of ``prior.PriorParams``) and a fitted Gaussian optimum (``ELEV_GRID``)
    chosen leave-one-year-out.  Held-out pooled AUC and paired difference to the base with week / year bootstrap intervals;
    optionally the same factors on the GBIF target-group cell-days."""
    from . import trigger
    dl = ss.d[ss.located]
    y = dl["y"].to_numpy(bool)
    year = dl["year"].to_numpy()
    doy = dl["date"].dt.day_of_year.to_numpy()
    base_cell = ss.cell0["full_tuned"]
    vm = ss.valid.astype("float32")
    nv = np.maximum(vm.sum(1), 1)
    mean = lambda x: (np.nan_to_num(x) * vm).sum(1) / nv
    base = mean(base_cell)
    es = mean(base_cell * trigger.elev_season_factor(ss.elev, doy[:, None]))
    rule = mean(base_cell * rule_table_factor(ss.elev, doy[:, None]))
    E = elevation_candidates(base_cell, ss.valid, ss.elev, doy)
    r = nested_loyo(E, y, year)
    fit = r["held"]
    ok = np.isfinite(fit)
    in_sample = rank_auc_rows(E, y)
    best = int(np.argmax(in_sample))
    out = {"n": int(len(dl)), "base_auc": auc(base, y), "es_auc": auc(es, y), "rule_table_auc": auc(rule, y),
           "fitted_in_sample_auc": float(in_sample[best]), "fitted_params": ELEV_GRID[best],
           "fitted_heldout_auc": auc(fit[ok], y[ok]), "base_on_same_rows": auc(base[ok], y[ok]),
           "fitted_distinct_choices": len(set(r["chosen"].values()))}
    cl = {"week": dl["yw"].to_numpy(), "year": year}
    for name, v in (("es", es), ("rule_table", rule), ("fitted_loyo", np.where(ok, fit, np.nan))):
        out[f"{name}_vs_base"] = {c: paired_vec(v, base, y, x, n_boot) for c, x in cl.items()}
    if gbif:
        fac = {"es": lambda e, dy: trigger.elev_season_factor(e, dy),
               "rule_table": lambda e, dy: rule_table_factor(e, dy),
               "fitted_all_years": lambda e, dy, g=ELEV_GRID[best]: elev_optimum_factor(e, dy, **g)}
        out["gbif_target_group"] = gbif_elevation_factor(fac, n_boot=min(n_boot, 400))
    return out


# ===================================================================================================
# year quality, 2015
# ===================================================================================================
def weekly_year(d: pd.DataFrame, daily: pd.DataFrame, year: int, cols=("season", "comb_lit", "comb_tuned", "full_tuned"),
                ref_years=range(2010, 2023)) -> pd.DataFrame:
    """For one year: reports per ISO week (n, positives, share) and the national model mean of that week next to the
    mean of the same week over ``ref_years`` (without ``year``)."""
    dd = daily.copy()
    dd["year"], dd["wk"] = dd.index.year, dd.index.isocalendar().week.astype(int).to_numpy()
    cur = dd[dd["year"] == year].groupby("wk")[list(cols)].mean().add_suffix("_year")
    clim = dd[dd["year"].isin([y for y in ref_years if y != year])].groupby("wk")[list(cols)].mean().add_suffix("_clim")
    g = d[d["year"] == year].assign(wk=lambda x: x["date"].dt.isocalendar().week.astype(int))
    rep = g.groupby("wk").agg(n=("y", "size"), n_pos=("y", "sum"))
    rep["share_pos"] = rep["n_pos"] / rep["n"]
    ref = d[d["year"].isin([y for y in ref_years if y != year])].assign(wk=lambda x: x["date"].dt.isocalendar().week.astype(int))
    rr = ref.groupby("wk").agg(n_ref=("y", "size"), pos_ref=("y", "sum"))
    rep["share_pos_ref"] = (rr["pos_ref"] / rr["n_ref"]).reindex(rep.index)
    return rep.join(cur).join(clim)


def national_year_signal(daily: pd.DataFrame, cols=("comb_lit", "comb_tuned", "full_tuned"), start=(8, 15), end=(10, 15)) -> pd.DataFrame:
    """Mean national model score 15 Aug - 15 Oct per year, and its z-score over the years 2010-2025."""
    dd = daily[(daily.index.month * 100 + daily.index.day >= start[0] * 100 + start[1])
               & (daily.index.month * 100 + daily.index.day <= end[0] * 100 + end[1])]
    m = dd.groupby(dd.index.year)[list(cols)].mean()
    ref = m.loc[2010:2025]
    return ((m - ref.mean()) / ref.std()).round(2)


# ===================================================================================================
# figures
# ===================================================================================================
PALETTE = {"season": "#8a8984", "comb_lit": "#eb6834", "comb_tuned": "#2a78d6", "hab": "#1baf7a", "hab_season": "#6250d6",
           "full_tuned": "#2a78d6", "full_lit": "#eb6834", "comb_tuned@es": "#e87ba4", "full_tuned@es": "#e87ba4"}
LABELS = {"season": "season only", "comb_lit": "literature trigger x season", "comb_tuned": "tuned trigger x season",
          "hab": "habitat only", "hab_season": "habitat x season", "full_tuned": "production score",
          "full_lit": "habitat x lit. trigger x season", "comb_tuned@es": "tuned x season x elev-season",
          "full_tuned@es": "production x elev-season"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e1",
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def _save_fig(fig, name, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / name
    fig.savefig(p, dpi=85, bbox_inches="tight")
    return p


def fig_auc(res: dict, fig_dir=FIG_DIR):
    plt = _mpl()
    cols = ["season", "comb_lit", "comb_tuned", "hab_season", "hab", "full_tuned"]
    panels = [("pooled: all reports", res["pooled"]["cell"], "auc"), ("within the same week (where?)", res["within"], "week"),
              ("within region and year (when?)", res["within"], "region_year")]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), sharey=True)
    for ax, (title, tab, kind) in zip(axes, panels):
        for i, c in enumerate(cols):
            if kind == "auc":
                r = tab.loc[c]
                est, lo, hi = r["auc"], r["lo_week"], r["hi_week"]
            else:
                r = tab[(tab["kind"] == kind) & (tab["score"] == c)].iloc[0]
                est, lo, hi = r["auc"], r["lo"], r["hi"]
            ax.plot([lo, hi], [i, i], color=PALETTE[c], lw=1.6, solid_capstyle="round")
            ax.plot([est], [i], "o", color=PALETTE[c], ms=5, mec="white", mew=1)
        ax.axvline(0.5, color="#52514e", lw=0.8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlim(0.44, 0.74)
        ax.set_xlabel("AUC, positive vs none")
    axes[0].set_yticks(range(len(cols)))
    axes[0].set_yticklabels([LABELS[c] for c in cols])
    axes[0].invert_yaxis()
    fig.suptitle("Located reports, own ellipse, exact day; 95 % interval over weeks", fontsize=8.5, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "backtest_auc.png", fig_dir)


def fig_years(res: dict, fig_dir=FIG_DIR):
    plt = _mpl()
    t = res["by_year_nat"]
    t = t[(t["n_none"] >= 8) & (t.index <= 2022)]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for c in ("season", "comb_lit", "comb_tuned"):
        ax.plot(t.index, t[c], "-o", color=PALETTE[c], lw=1.4, ms=4, mec="white", mew=0.8, label=LABELS[c])
    ax.axhline(0.5, color="#52514e", lw=0.8)
    if 2015 in t.index:
        ax.axvspan(2014.6, 2015.4, color="#eda100", alpha=0.18, lw=0)
    ax.set_xticks(t.index)
    ax.set_xticklabels([f"{y}\nn0={int(t.loc[y, 'n_none'])}" for y in t.index], fontsize=7.5)
    ax.set_ylabel("AUC positive vs none")
    ax.set_title("Per-year AUC, national forest mean on the report date (n0 = reports of 'none')", fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout()
    return _save_fig(fig, "backtest_years.png", fig_dir)


def fig_calibration(res: dict, fig_dir=FIG_DIR):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (title, key) in zip(axes, (("located reports, own ellipse", "cell"), ("all reports, national mean", "nat"))):
        for c in ("full_tuned", "comb_lit"):
            g = res["calibration"][key][c]
            x = np.arange(1, len(g) + 1) + (0.12 if c == "comb_lit" else -0.12)
            ax.errorbar(x, g["share_pos"], yerr=[g["share_pos"] - g["lo"], g["hi"] - g["share_pos"]], fmt="o", ms=4,
                        color=PALETTE[c], mec="white", mew=0.8, elinewidth=1.1, label=LABELS[c])
        base = res["base_rate"][key]
        ax.axhline(base, color="#52514e", lw=0.8, ls="--")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("score decile (1 = lowest)")
        ax.set_xticks(range(1, 11))
    axes[0].set_ylabel("share of reports with jurčki")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return _save_fig(fig, "backtest_calibration.png", fig_dir)


def fig_2015(res: dict, fig_dir=FIG_DIR):
    plt = _mpl()
    w = res["weekly_2015"]
    w = w.loc[24:45]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax = axes[0]
    ax.scatter(w.index, w["share_pos"], s=np.clip(w["n"], 3, 40) * 2.2, color="#2a78d6", edgecolor="white", lw=0.8, zorder=3,
               label="2015 (dot size = reports)")
    ax.plot(w.index, w["share_pos_ref"], color="#8a8984", lw=1.4, ls="--", label="2010-2022 without 2015")
    ax.set_ylabel("share with jurčki")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Forum reports 2015: share of outings that found jurčki, by ISO week", fontsize=8.5, loc="left")
    ax = axes[1]
    for c, name in (("comb_tuned", "production trigger x season"), ("comb_lit", "literature trigger x season")):
        ax.plot(w.index, w[f"{c}_year"], color=PALETTE[c], lw=1.8, label=f"{name}, 2015")
        ax.plot(w.index, w[f"{c}_clim"], color=PALETTE[c], lw=1.0, ls="--", alpha=0.7, label=f"{name}, other years")
    ax.set_ylabel("model, national forest mean")
    ax.set_xlabel("ISO week")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Model score (mean over Slovenian forest cells)", fontsize=8.5, loc="left")
    fig.tight_layout()
    return _save_fig(fig, "backtest_2015.png", fig_dir)


def fig_frontier(rt: dict, fig_dir=FIG_DIR):
    plt = _mpl()
    sc = rt["scatter"]
    if sc["gbif_pct"] is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    ins, gp, tgi = rt["in_sample"], rt["gbif_find_percentile"], rt["target_group_in_sample"]
    best = ins["best_index"]
    for ax, key, xlab, lit, prod in (
            (axes[0], "gbif_pct", "GBIF finds: mean temporal percentile", gp["literature"], gp["production"]),
            (axes[1], "target_group_auc", "GBIF target group: AUC of season x trigger", tgi["literature"], tgi["production"])):
        ax.scatter(sc[key], sc["forum_auc"], s=9, color="#8a8984", alpha=0.55, lw=0)
        ax.plot([lit], [ins["literature"]], "o", color="#eb6834", ms=8, mec="white", label="literature")
        ax.plot([prod], [ins["production"]], "o", color="#2a78d6", ms=8, mec="white", label="production (tuned on GBIF)")
        ax.plot([sc[key][best]], [ins["best"]], "o", color="#1baf7a", ms=8, mec="white", label="best on forum")
        ax.set_xlabel(xlab)
    axes[0].set_ylabel("forum reports: AUC positive vs none")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("800 candidate parameter sets: the forum and GBIF disagree about the trigger", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout()
    return _save_fig(fig, "backtest_frontier.png", fig_dir)


# ===================================================================================================
# driver
# ===================================================================================================
def _records(df) -> list:
    df = df.reset_index() if not (isinstance(df.index, pd.RangeIndex) and df.index.name is None) else df
    df = df.copy()
    df.columns = [c if isinstance(c, str) else "|".join(map(str, c)) for c in df.columns]
    return json.loads(df.to_json(orient="records", double_precision=4))


def author_cluster(rep: pd.DataFrame) -> np.ndarray:
    return np.where(rep["author"].to_numpy(str) == "", rep["post_id"].to_numpy(str), rep["author"].to_numpy(str))


def run_all(quick: bool = False, do_retune: bool = True, do_figures: bool = True, n_cands: int = 800, seed: int = 0,
            fig_dir=FIG_DIR, out=RESULTS, log=print) -> dict:
    t0 = time.time()
    n_boot = 200 if quick else 1000
    raw = load_reports(last_day=None)
    d = dedupe(load_reports(last_day=None))
    res: dict = {"generated": date_today(), "n_boot": n_boot,
                 "counts": {"usable_reports_all_years": int(len(raw)) + int(raw.attrs["n_before"]),
                            "lost_before_2010": int(raw.attrs["n_before"]),
                            "scorable": int(len(raw)), "duplicates_removed": int(d.attrs["n_duplicates"]),
                            "analysed": int(len(d)), "n_pos": int(d["y"].sum()), "n_none": int((1 - d["y"]).sum()),
                            "located": int(d["loc_lon"].notna().sum()), "authors_known": int((d["author"] != "").sum()),
                            "distinct_authors": int(d.loc[d["author"] != "", "author"].nunique())}}
    forest = Forest()
    cells, ru, rg, nu, located = sample_units(forest, d)
    res["counts"]["located_scored"] = int(located.sum())
    log(f"{len(d)} reports ({int(located.sum())} located), {len(cells)} cells, {time.time() - t0:.0f} s")
    eng = load_engine(cells)
    params = production_params()
    ss = score_reports(d, eng, (ru, rg, nu, located), params)
    log(f"scored in {time.time() - t0:.0f} s")
    cols = [c for c in BASE_WANT if not c.startswith("trig")]

    # pooled, all scopes
    pooled = {}
    for scope in SCOPES:
        rep, S = view(ss, scope, 0)
        pooled[scope] = pooled_table(S[cols], rep["y"], {"week": rep["yw"].to_numpy(), "author": author_cluster(rep),
                                                          "year": rep["year"].to_numpy()}, n_boot=n_boot)
    res["pooled"] = pooled
    tol_rows = {}
    for sc in ("nat", "cell"):
        for t in TOLS:
            r_, S_ = view(ss, sc, t)
            tol_rows[f"{sc}|{t}"] = {c: auc(S_[c], r_["y"]) for c in cols}
    res["tolerance"] = pd.DataFrame(tol_rows).T
    pairs = [("comb_lit", "season"), ("comb_tuned", "season"), ("comb_tuned", "comb_lit"), ("full_tuned", "comb_tuned"),
             ("full_tuned", "hab_season"), ("comb_tuned@es", "comb_tuned"), ("full_tuned@es", "full_tuned"),
             ("full_lit", "full_tuned")]
    paired = {}
    for scope in ("nat", "cell"):
        rep, S = view(ss, scope, 0)
        t1 = paired_diffs(S, rep["y"], pairs, rep["yw"].to_numpy(), n_boot=n_boot)
        t2 = paired_diffs(S, rep["y"], pairs, rep["year"].to_numpy(), n_boot=n_boot)
        paired[scope] = t1.rename(columns={"lo": "lo_week", "hi": "hi_week"}).join(
            t2[["lo", "hi"]].rename(columns={"lo": "lo_year", "hi": "hi_year"}))
    res["paired"] = paired
    rep, S = view(ss, "cell", 0)
    res["within"] = within_table(rep, S, cols, n_boot=n_boot // 2)
    repn, Sn = view(ss, "nat", 0)
    res["within_nat"] = within_table(repn, Sn, cols, kinds=("week", "year"), n_boot=n_boot // 2)
    res["by_year_cell"] = year_table(rep, S, ["season", "comb_lit", "comb_tuned", "hab", "full_tuned"])
    res["by_year_nat"] = year_table(repn, Sn, ["season", "comb_lit", "comb_tuned", "full_tuned"])
    res["calibration"] = {"cell": {c: calibration(rep, S[c]) for c in ("full_tuned", "comb_lit")},
                          "nat": {c: calibration(repn, Sn[c]) for c in ("full_tuned", "comb_lit")}}
    res["base_rate"] = {"cell": float(rep["y"].mean()), "nat": float(repn["y"].mean())}
    res["dose"] = {"nat": dose_response(repn, Sn, ["season", "comb_lit", "comb_tuned", "full_tuned"]),
                   "cell": dose_response(rep, S, ["season", "comb_lit", "comb_tuned", "full_tuned", "hab"])}
    res["sensitivity"] = {"nat": sensitivity(ss, scope="nat"), "cell": sensitivity(ss, scope="cell")}
    # years
    daily = national_daily(eng, nu, params, range(2010, 2027))
    res["year_quality"] = {sc: year_quality(*view(ss, sc, 0), ["season", "comb_lit", "comb_tuned", "full_tuned"])
                           for sc in ("nat", "cell")}
    res["national_signal"] = national_year_signal(daily)
    res["weekly_2015"] = weekly_year(d, daily, 2015)
    res["weather_2015"] = national_weather_summary(eng, nu, range(2010, 2023))
    res["by_region"] = d.assign(region=d["region"].replace("", "unlocated")).groupby("region")["y"].agg(["size", "mean"]).round(3)
    log(f"descriptive tables done, {time.time() - t0:.0f} s")

    if do_retune:
        td = make_tune_data(d, ru, nu, located)
        rt = retune(d, eng, td, ss, n_cands, seed, n_boot=n_boot, log=log)
        cands = rt.pop("_cands")
        rt.pop("_M")
        from . import trigger
        tg_cands = {"production": params["tuned"], "literature": params["lit"]}
        for name, e in rt["procedures"].items():
            tg_cands[f"pick_{name}"] = trigger.params_from_dict(e["all_data_pick"]["params"])
        res["target_group"] = gbif_target_group(tg_cands, n_boot=min(n_boot, 300))
        rec = recommend(rt, res["target_group"])
        rt["recommendation"] = rec
        res["retune"] = rt
        res["elevation"] = elevation_test(ss, n_boot=n_boot)
        winners = [n for n, v in rec.items() if v["recommend"]]
        res["recommended"] = winners[0] if winners else None
        if winners:
            pick = rt["procedures"][winners[0]]["all_data_pick"]
            Path(PARAMS_FORUM).write_text(json.dumps({
                "generated": res["generated"], "status": "PROPOSAL, not used by the app; production stays trigger_params.json",
                "procedure": winners[0], "params": pick["params"],
                "production": rt["candidates"]["production"], "literature": rt["candidates"]["literature"],
                "heldout": {k: rt["procedures"][winners[0]][k] for k in rt["procedures"][winners[0]]
                            if k.startswith("heldout") or k.startswith("vs_")},
                "gbif_target_group": json.loads(res["target_group"].loc[f"pick_{winners[0]}"].to_json()),
                "note": "docs/BACKTEST.md"}, indent=1))
            log(f"wrote {PARAMS_FORUM}")
        log(f"retune done, {time.time() - t0:.0f} s")
    if do_figures:
        paths = [fig_auc(res, fig_dir), fig_years(res, fig_dir), fig_calibration(res, fig_dir), fig_2015(res, fig_dir)]
        if do_retune:
            paths.append(fig_frontier(res["retune"], fig_dir))
        log("figures: " + ", ".join(p.name for p in paths if p is not None))
    # results file (small)
    def jsonable(o):
        if isinstance(o, pd.DataFrame):
            return _records(o)
        if isinstance(o, dict):
            return {str(k): jsonable(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.ndarray, tuple, list)):
            return [jsonable(x) for x in o]
        return o

    Path(out).write_text(json.dumps(jsonable({k: v for k, v in res.items()}), ensure_ascii=False, default=str))
    res["_ss"] = ss
    return res


def date_today() -> str:
    from datetime import date
    return date.today().isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap draws")
    ap.add_argument("--no-retune", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--candidates", type=int, default=800)
    a = ap.parse_args(argv)
    res = run_all(a.quick, not a.no_retune, not a.no_figures, a.candidates)
    print(f"wrote {RESULTS}")
    return res


if __name__ == "__main__":
    main()
