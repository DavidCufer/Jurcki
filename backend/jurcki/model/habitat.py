"""Habitat (static) suitability: where can jurcki grow, from presence/background data.

Presences are GBIF finds of the four jurcki species:
* dated records (data/processed/gbif_boletus.parquet), one per grid cell;
* Slovenian Forestry Institute (SFI) records from the raw extract. They have
  year only, rounded coordinates and ~3 km uncertainty, so each unique site
  stands for a 3 km disc: it contributes K random forest cells of that disc
  with total weight 1 (training) and is scored on a disc-averaged surface
  (evaluation), never on a single cell.

Sampling bias is handled by stratifying the background: it is drawn only from
forest-ish cells (treecover > 0.1), per region (Slovenia / near foreign
finds), and within each region its total weight equals that of the presences,
so the model cannot learn "Austria" or "near Ljubljana" as a signal.
Validation uses spatial block CV (whole blocks held out).

Run:  python -m jurcki.model.habitat [--block-km 10] [--final auto|model|blend]
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import ndimage, signal

from .. import config, features, grid
from . import evaluate, prior
from ..sources.gbif import JURCKI

RAW_GBIF = config.RAW / "gbif" / "boletus_2026-10-01.parquet"
DATED = config.PROCESSED / "gbif_boletus.parquet"
MODEL_DIR = config.INTERIM / "model"
PREVIEW_DIR = config.INTERIM / "preview"
OUT_TIF = MODEL_DIR / "habitat.tif"
OUT_MODEL = MODEL_DIR / "habitat_lgbm.txt"            # core features, valid everywhere
OUT_MODEL_FULL = MODEL_DIR / "habitat_lgbm_full.txt"  # + forest-type features
OUT_METRICS = config.PROCESSED / "habitat_metrics.json"

TREECOVER_MIN = 0.1       # background / prediction domain
SFI_RADIUS_M = 3000       # SFI coordinate uncertainty
SFI_CELLS = 6             # forest cells sampled per SFI site
FOREIGN_BUFFER_M = 20_000  # background distance from foreign finds
FOREIGN_WEIGHT = 0.5      # foreign finds count half a Slovenian one
ELEV_CEILING = (1700.0, 1900.0)  # suitability fades to 0 between these (treeline)
TREECOVER_RAMP = (TREECOVER_MIN, 0.4)  # mycorrhizal hosts needed: fades to 0 at low cover
BUILTUP_FULL = 1.0  # built-up share of a cell at which suitability is 0
TREECOVER_NEIGHBOURHOOD = 3  # cells; max cover nearby is used (finds sit at edges, GPS is noisy)

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=6, max_depth=3,
              min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=20.0, min_gain_to_split=0.0, verbose=-1,
              seed=1, num_threads=4)
N_ROUNDS = 250
MONOTONE = {"treecover": 1}
# Features used by the model. The other static layers (eastness, tpi500, tri,
# cropland, builtup) added no skill in spatial CV, so they are left out.
CORE = ["elev", "slope", "northness", "twi", "treecover", "grassland", "forest_edge_dist", "tpi2000"]

SANITY_SITES = {  # name: (lon, lat, radius_km)
    "Pohorje": (15.50, 46.47, 12), "Kocevski Rog": (14.80, 45.67, 10),
    "Sneznik": (14.43, 45.58, 8), "Pokljuka": (13.95, 46.34, 8),
    "Goricko": (16.05, 46.85, 12), "Haloze": (15.95, 46.35, 10),
    "Ljubljansko barje": (14.38, 45.96, 4), "Prekmurje fields": (16.30, 46.58, 6),
}


# ------------------------------------------------------------ sampling / CV

def block_ids(rows, cols, block_cells: int, n_cols: int | None = None) -> np.ndarray:
    """Id of the square block (``block_cells`` cells wide) containing each cell."""
    n_cols = n_cols or grid.grid_shape()[1]
    per_row = -(-n_cols // block_cells)
    return (np.asarray(rows) // block_cells) * per_row + np.asarray(cols) // block_cells


def assign_folds(blocks, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per item such that every block lies wholly in one fold."""
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    order = np.random.default_rng(seed).permutation(uniq)
    fold_of = {b: i % n_folds for i, b in enumerate(order)}
    return np.array([fold_of[b] for b in blocks], dtype=int)


def sample_background(eligible: np.ndarray, n: int, rng,
                      prob: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``n`` distinct random (rows, cols) among the True cells of ``eligible``.

    ``prob`` (grid-shaped, >= 0) makes the draw proportional to it instead of uniform.
    """
    rr, cc = np.nonzero(eligible)
    p = None
    if prob is not None:
        p = prob[rr, cc].astype(float)
        p = p / p.sum()
    idx = rng.choice(len(rr), size=min(n, len(rr)), replace=False, p=p)
    return rr[idx], cc[idx]


def effort_density(rows, cols, shape, sigma_cells: float, mix: float) -> np.ndarray:
    """Sampling-effort surface: ``mix`` x smoothed density of finds + (1 - mix) x uniform.

    Drawing background from it makes the model learn local habitat contrast
    instead of "where people look" (target-group / bias-file background).
    """
    cnt = np.zeros(shape, "float32")
    np.add.at(cnt, (np.asarray(rows), np.asarray(cols)), 1.0)
    dens = ndimage.gaussian_filter(cnt, sigma_cells)
    dens = dens / dens.sum()
    return mix * dens + (1 - mix) / dens.size


def disc_kernel(radius_cells: float) -> np.ndarray:
    """Boolean disc of the given radius (in cells) as a square array."""
    k = int(np.ceil(radius_cells))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return (x ** 2 + y ** 2) <= radius_cells ** 2


def sample_disc_cells(row, col, eligible, radius_cells, k, rng) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``k`` distinct eligible cells within ``radius_cells`` of (row, col)."""
    kern = disc_kernel(radius_cells)
    h = kern.shape[0] // 2
    dr, dc = np.nonzero(kern)
    rr, cc = row + dr - h, col + dc - h
    ok = (rr >= 0) & (rr < eligible.shape[0]) & (cc >= 0) & (cc < eligible.shape[1])
    rr, cc = rr[ok], cc[ok]
    ok = eligible[rr, cc]
    rr, cc = rr[ok], cc[ok]
    if len(rr) > k:
        idx = rng.choice(len(rr), size=k, replace=False)
        rr, cc = rr[idx], cc[idx]
    return rr, cc


def disc_mean(surface: np.ndarray, weight: np.ndarray, radius_cells: float) -> np.ndarray:
    """Mean of ``surface`` over a disc, counting only cells with ``weight`` > 0."""
    kern = disc_kernel(radius_cells).astype("float32")
    w = weight.astype("float32")
    num = signal.fftconvolve(np.nan_to_num(surface) * w, kern, mode="same")
    den = signal.fftconvolve(w, kern, mode="same")
    return np.where(den > 0.5, num / np.maximum(den, 1e-6), 0.0).astype("float32")


# ------------------------------------------------------------ data

def load_units(forest: np.ndarray, land: np.ndarray, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Presence units and training rows.

    units: one per dated cell / SFI site (kind, region, weight, row, col).
    rows:  training presences (unit id, cell, weight); SFI sites have several.
    """
    shape = forest.shape
    d = pd.read_parquet(DATED)
    d = d[d.is_jurcki]
    r, c = grid.lonlat_to_rowcol(d.decimallongitude.values, d.decimallatitude.values)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    d = d[ok].assign(row=r[ok], col=c[ok]).drop_duplicates(["row", "col"])
    unc = d.coordinateuncertaintyinmeters.fillna(0).values
    dated = pd.DataFrame({"kind": "dated", "row": d.row.values, "col": d.col.values,
                          "weight": np.where(unc <= 500, 1.0, 0.5)})

    raw = pd.read_parquet(RAW_GBIF)
    s = raw[(raw.institutioncode == "SFI") & raw.species.isin(JURCKI)]
    s = s.drop_duplicates(["decimallatitude", "decimallongitude"])
    r, c = grid.lonlat_to_rowcol(s.decimallongitude.values, s.decimallatitude.values)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    sfi = pd.DataFrame({"kind": "sfi", "row": r[ok], "col": c[ok], "weight": 1.0})

    units = pd.concat([dated, sfi], ignore_index=True)
    units["region"] = np.where(land[units.row, units.col], "si", "foreign")
    units.loc[units.region == "foreign", "weight"] *= FOREIGN_WEIGHT

    rad = SFI_RADIUS_M / config.GRID_RES
    parts = []
    for i, u in units.iterrows():
        if u.kind == "dated":
            rr, cc = np.array([u.row]), np.array([u.col])
        else:
            rr, cc = sample_disc_cells(u.row, u.col, forest, rad, SFI_CELLS, rng)
        parts.append(pd.DataFrame({"unit": i, "row": rr, "col": cc,
                                   "weight": u.weight / max(len(rr), 1)}))
    rows = pd.concat(parts, ignore_index=True)
    units["n_cells"] = rows.groupby("unit").size().reindex(units.index).fillna(0).astype(int)
    return units, rows


def load_background(units: pd.DataFrame, forest: np.ndarray, land: np.ndarray,
                    n_si: int, n_foreign: int, rng, bias_mix: float = 0.0,
                    bias_sigma_km: float = 10.0) -> pd.DataFrame:
    """Background cells per region, weighted to match each region's presence weight.

    ``bias_mix`` > 0 draws part of the background near the finds (see effort_density).
    """
    prob = None
    if bias_mix > 0:
        prob = effort_density(units.row, units.col, forest.shape,
                              bias_sigma_km * 1000 / config.GRID_RES, bias_mix)
    near = np.zeros(forest.shape, bool)
    fu = units[units.region == "foreign"]
    near[fu.row.values, fu.col.values] = True
    dist = ndimage.distance_transform_edt(~near) * config.GRID_RES
    out = []
    for region, elig, n in (("si", forest & land, n_si),
                            ("foreign", forest & ~land & (dist <= FOREIGN_BUFFER_M), n_foreign)):
        rr, cc = sample_background(elig, n, rng, prob)
        w = units.loc[units.region == region, "weight"].sum() / max(len(rr), 1)
        out.append(pd.DataFrame({"region": region, "row": rr, "col": cc, "weight": w}))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------ model

class Surfaces:
    """Whole-grid feature table and helpers to turn a model into a 0..1 surface."""

    def __init__(self):
        self.names, self.stack = features.load_static_stack()
        self.shape = self.stack.shape[1:]
        self.X = self.stack.reshape(len(self.names), -1).T  # (cells, features)
        self.treecover = self.stack[self.names.index("treecover")]
        self.elev = self.stack[self.names.index("elev")]
        self.forest = self.treecover > TREECOVER_MIN

    def at(self, rows, cols, names) -> np.ndarray:
        idx = [self.names.index(n) for n in names]
        return self.X[np.asarray(rows) * self.shape[1] + np.asarray(cols)][:, idx]

    def envelope(self) -> np.ndarray:
        """Hard ecological limits: no host trees, built-up cells, above the treeline."""
        lo, hi = ELEV_CEILING
        t0, t1 = TREECOVER_RAMP
        near = ndimage.maximum_filter(self.treecover, size=TREECOVER_NEIGHBOURHOOD)
        trees = np.clip((near - t0) / (t1 - t0), 0, 1)
        built = 1 - np.clip(self.stack[self.names.index("builtup")] / BUILTUP_FULL, 0, 1)
        return trees * built * np.clip((hi - self.elev) / (hi - lo), 0, 1)

    def predict(self, booster, names) -> np.ndarray:
        """Raw probability on the grid (0 outside the forest domain), before rescaling."""
        idx = [self.names.index(n) for n in names]
        flat = self.forest.ravel()
        out = np.zeros(flat.shape, "float32")
        out[flat] = booster.predict(self.X[flat][:, idx])
        return out.reshape(self.shape) * self.envelope()

    def prior(self, month: int = 9) -> np.ndarray:
        g = lambda n: self.stack[self.names.index(n)]
        s = prior.habitat_score(self.elev, g("slope"), self.treecover, g("forest_edge_dist"),
                                g("northness"), month=month)
        return (s * self.forest).astype("float32")


def fit(X, y, w, names, rounds: int = N_ROUNDS, seed: int = 1) -> lgb.Booster:
    """Shallow, strongly regularised LightGBM presence/background classifier."""
    w = np.asarray(w, float)
    w = w * len(w) / w.sum()
    params = dict(PARAMS, seed=seed,
                  monotone_constraints=[MONOTONE.get(n, 0) for n in names])
    return lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=list(names)), rounds)


def rescale(raw: np.ndarray, ref: np.ndarray, q: float = 99.5) -> np.ndarray:
    """Divide by the q-th percentile over the reference cells and clip to 0..1."""
    scale = max(float(np.percentile(raw[ref], q)), 1e-9)
    return np.clip(raw / scale, 0, 1).astype("float32")


def train_variant(sf: Surfaces, train: pd.DataFrame, names: list[str], rounds=N_ROUNDS):
    """Fit on rows with every feature in ``names`` valid; returns the booster."""
    X = sf.at(train.row, train.col, names)
    ok = np.isfinite(X).all(axis=1)
    return fit(X[ok], train.y.values[ok], train.weight.values[ok], names, rounds)


def model_surface(sf: Surfaces, core: lgb.Booster, full: lgb.Booster | None, land) -> np.ndarray:
    """Rescaled 0..1 surface: full model where forest-type layers exist, else core model."""
    ref = sf.forest & land
    s = rescale(sf.predict(core, CORE), ref)
    if full is not None:
        valid = np.isfinite(sf.stack[[sf.names.index(n) for n in features.STATIC_FOREST_TYPE]]).all(0)
        s_full = rescale(sf.predict(full, CORE + features.STATIC_FOREST_TYPE), ref)
        s = np.where(valid, s_full, s)
    return s


# ------------------------------------------------------------ evaluation

def unit_percentiles(surf, coarse, units, bg) -> dict[str, np.ndarray]:
    """Percentile of each presence unit among region-matched background cells.

    Dated units are scored on the cell, SFI units on the disc-averaged surface
    (and then compared with disc-averaged background), as their location is
    only known to ~3 km.
    """
    out = {}
    for (kind, region), g in units.groupby(["kind", "region"]):
        s = coarse if kind == "sfi" else surf
        b = bg[bg.region == region]
        pop = s[b.row.values, b.col.values]
        out[f"{region}_{kind}"] = np.array(
            [evaluate.percentile_of(s[r, c], pop) for r, c in zip(g.row, g.col)])
    return out


def cross_validate(sf: Surfaces, units, rows, bg, land, block_cells, n_folds=5, seed=0,
                   rounds=N_ROUNDS, blend=0.5, bg_eval=None) -> dict:
    """Spatial block CV of prior, core model, core+forest-type model and blends.

    Models train on ``bg``; held-out presences are ranked against ``bg_eval``
    (default ``bg``) of the held-out blocks.
    """
    bg_eval = bg if bg_eval is None else bg_eval
    rad = SFI_RADIUS_M / config.GRID_RES
    unit_fold = assign_folds(block_ids(units.row, units.col, block_cells), n_folds, seed)
    row_fold = unit_fold[rows.unit.values]
    bg_fold = assign_folds(block_ids(bg.row, bg.col, block_cells), n_folds, seed)
    pri = sf.prior()
    pri_c = disc_mean(pri, sf.forest, rad)
    full_names = CORE + features.STATIC_FOREST_TYPE
    valid = np.isfinite(sf.stack[[sf.names.index(n) for n in features.STATIC_FOREST_TYPE]]).all(0)
    train_all = pd.concat([rows.assign(y=1), bg.assign(y=0)], ignore_index=True)
    fold_all = np.concatenate([row_fold, bg_fold])
    ref = sf.forest & land

    acc: dict[str, dict[str, list]] = {}
    per_fold: dict[str, list] = {}
    for k in range(n_folds):
        tr = train_all[fold_all != k]
        core = train_variant(sf, tr, CORE, rounds)
        full = train_variant(sf, tr, full_names, rounds)
        s_core = rescale(sf.predict(core, CORE), ref)
        s_full = np.where(valid, rescale(sf.predict(full, full_names), ref), s_core)
        surfs = {"prior": pri, "core": s_core, "full": s_full,
                 "blend_core": blend * s_core + (1 - blend) * pri,
                 "blend_full": blend * s_full + (1 - blend) * pri}
        te_u = units[unit_fold == k]
        te_b = bg_eval[assign_folds(block_ids(bg_eval.row, bg_eval.col, block_cells), n_folds, seed) == k]
        for name, s in surfs.items():
            pu = unit_percentiles(s, disc_mean(s, sf.forest, rad) if name != "prior" else pri_c,
                                  te_u, te_b)
            for grp, p in pu.items():
                acc.setdefault(name, {}).setdefault(grp, []).append(p)
            si = np.concatenate([p for g, p in pu.items() if g.startswith("si_")] or [[]])
            per_fold.setdefault(name, []).append(evaluate.auc_from_percentiles(si))
    res = {}
    for name, groups in acc.items():
        res[name] = {}
        pooled_si = []
        for grp, ps in groups.items():
            p = np.concatenate(ps)
            res[name][grp] = evaluate.summarize(p, top=(0.1, 0.2))
            if grp.startswith("si_"):
                pooled_si.append(p)
        res[name]["si_all"] = evaluate.summarize(np.concatenate(pooled_si), top=(0.1, 0.2))
        res[name]["si_all"]["fold_auc"] = [float(a) for a in per_fold[name]]
    return res


# ------------------------------------------------------------ interpretation

def importances(booster: lgb.Booster) -> dict[str, float]:
    """Gain share per feature (sums to 1), sorted."""
    gain = booster.feature_importance("gain")
    share = gain / gain.sum()
    return {n: round(float(v), 4) for n, v in
            sorted(zip(booster.feature_name(), share), key=lambda t: -t[1])}


PD_GRIDS = {"elev": [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2200],
            "conifer_frac": [0, .1, .25, .5, .75, .9, 1.0],
            "treecover": [.1, .2, .3, .5, .7, .9, 1.0],
            "slope": [0, 3, 6, 10, 15, 20, 25, 30, 40],
            "twi": [5, 6, 7, 8, 9, 10, 12, 14],
            "forest_edge_dist": [-250, -100, 0, 50, 100, 250, 500, 1000, 2000]}


def partial_dependence(booster, X: pd.DataFrame, raw_ref: float, grids=PD_GRIDS) -> dict:
    """Mean prediction (relative to the 99.5th percentile ``raw_ref``) as one feature varies."""
    out = {}
    for f, vals in grids.items():
        if f not in X.columns:
            continue
        res = []
        for v in vals:
            Xv = X.copy()
            Xv[f] = v
            res.append(round(float(np.clip(booster.predict(Xv) / raw_ref, 0, 1.5).mean()), 3))
        out[f] = dict(zip([str(v) for v in vals], res))
    return out


def sanity_checks(sf: Surfaces, surf: np.ndarray, land) -> dict:
    """Mean suitability around known areas (all cells / forest cells) and above 1800 m."""
    yy, xx = np.mgrid[:sf.shape[0], :sf.shape[1]]
    out = {}
    for name, (lon, lat, km) in SANITY_SITES.items():
        r, c = grid.lonlat_to_rowcol(np.array([lon]), np.array([lat]))
        m = (yy - r[0]) ** 2 + (xx - c[0]) ** 2 <= (km * 1000 / config.GRID_RES) ** 2
        f = m & sf.forest
        out[name] = {"mean_all": round(float(surf[m].mean()), 3),
                     "mean_forest": round(float(surf[f].mean()), 3) if f.any() else None,
                     "forest_share": round(float(f.sum() / m.sum()), 2)}
    hi = land & (sf.elev > 1800)
    out["above_1800m"] = {"mean_all": round(float(surf[hi].mean()), 4), "max": round(float(surf[hi].max()), 3),
                          "cells": int(hi.sum())}
    for name, m in (("cropland>0.5 cells", sf.stack[sf.names.index("cropland")] > 0.5),
                    ("builtup>0.3 cells", sf.stack[sf.names.index("builtup")] > 0.3),
                    ("treecover<0.1 cells", ~sf.forest)):
        m = m & land
        out[name] = {"mean_all": round(float(surf[m].mean()), 3), "cells": int(m.sum())}
    out["Slovenia_forest_mean"] = round(float(surf[land & sf.forest].mean()), 3)
    return out


def save_preview(surf: np.ndarray, land: np.ndarray, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(np.where(land, surf, np.nan), cmap="YlGn", vmin=0, vmax=1)
    ax.set_title("habitat suitability")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ main

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block-km", type=float, default=10.0)
    ap.add_argument("--n-bg-si", type=int, default=20000)
    ap.add_argument("--n-bg-foreign", type=int, default=8000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--blend", type=float, default=0.5, help="model weight in the blend with the prior")
    ap.add_argument("--final", choices=["auto", "model", "blend"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    sf = Surfaces()
    land = grid.load_mask()
    units, rows = load_units(sf.forest, land, rng)
    bg = load_background(units, sf.forest, land, a.n_bg_si, a.n_bg_foreign, rng)
    print(units.groupby(["region", "kind"]).agg(n=("row", "size"), weight=("weight", "sum")))
    print("training rows", len(rows), "background", bg.groupby("region").size().to_dict())

    metrics = {"n_units": {f"{r}_{k}": int(n) for (r, k), n in units.groupby(["region", "kind"]).size().items()},
               "params": dict(PARAMS, rounds=N_ROUNDS), "block_km": a.block_km, "cv": {}}
    for km in sorted({a.block_km, 20.0}):
        cells = int(round(km * 1000 / config.GRID_RES))
        cv = cross_validate(sf, units, rows, bg, land, cells, a.folds, a.seed, blend=a.blend)
        metrics["cv"][f"{km:g}km"] = cv
        print(f"\n== spatial block CV, {km:g} km blocks, {a.folds} folds ==")
        for name, res in cv.items():
            for grp in ("si_all", "si_dated", "si_sfi", "foreign_dated"):
                if grp in res:
                    r = res[grp]
                    print(f"{name:11s} {grp:14s} n={r['n']:4d} auc={r['auc']:.3f} "
                          f"top10={r['hit_top10']:.3f} top20={r['hit_top20']:.3f}")

    # variant choice on the main block size: full (with forest-type layers) only if it helps SI AUC
    cv = metrics["cv"][f"{a.block_km:g}km"]
    use_full = cv["full"]["si_all"]["auc"] > cv["core"]["si_all"]["auc"] + 0.005
    key = "full" if use_full else "core"
    use_blend = a.final == "blend" or (a.final == "auto" and
                                       cv[f"blend_{key}"]["si_all"]["auc"] > cv[key]["si_all"]["auc"])
    metrics["final"] = {"variant": key, "blend_with_prior": bool(use_blend), "blend_model_weight": a.blend}

    # final fit on all data
    train = pd.concat([rows.assign(y=1), bg.assign(y=0)], ignore_index=True)
    full_names = CORE + features.STATIC_FOREST_TYPE
    core = train_variant(sf, train, CORE)
    full = train_variant(sf, train, full_names)
    s_model = model_surface(sf, core, full if use_full else None, land)
    s_prior = sf.prior()
    surf = a.blend * s_model + (1 - a.blend) * s_prior if use_blend else s_model
    surf = (surf * sf.forest).astype("float32")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    core.save_model(str(OUT_MODEL))
    full.save_model(str(OUT_MODEL_FULL))
    grid.write_tif(OUT_TIF, surf)
    save_preview(surf, land, PREVIEW_DIR / "habitat.png")

    # interpretation (from the full model, which has all features)
    sample = bg[bg.region == "si"].sample(min(4000, (bg.region == "si").sum()), random_state=1)
    Xs = pd.DataFrame(sf.at(sample.row, sample.col, full_names), columns=full_names)
    Xs = Xs[Xs.conifer_frac.notna()].reset_index(drop=True)
    ref = float(np.percentile(sf.predict(full, full_names)[sf.forest & land], 99.5))
    metrics["importance_core"] = importances(core)
    metrics["importance_full"] = importances(full)
    metrics["partial_dependence_full"] = partial_dependence(full, Xs, ref)
    metrics["sanity"] = {"final": sanity_checks(sf, surf, land),
                         "prior": sanity_checks(sf, s_prior, land),
                         "model": sanity_checks(sf, s_model, land)}
    OUT_METRICS.write_text(json.dumps(metrics, indent=1))
    print("\nfinal:", metrics["final"])
    print("importance (full):", metrics["importance_full"])
    print("wrote", OUT_TIF, OUT_MODEL, OUT_METRICS)
    return metrics


if __name__ == "__main__":
    main()
