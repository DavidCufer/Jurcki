"""Backtest metrics for score surfaces against reported finds.

Two questions are scored separately because they fail independently:
* spatial: on the day of a find, is the find's cell ranked high among all
  candidate (forest) cells of that day?
* temporal: at the find's cell, is the find day ranked high among the other
  days of that season?
A model can be good at "where" (habitat) and bad at "when" (trigger), or the
other way round.
"""
import numpy as np


def percentile_of(value, population):
    """Fraction of `population` strictly below `value` (ties count half)."""
    population = np.asarray(population, dtype=float)
    population = population[np.isfinite(population)]
    if population.size == 0 or not np.isfinite(value):
        return np.nan
    below = np.count_nonzero(population < value)
    ties = np.count_nonzero(population == value)
    return (below + 0.5 * ties) / population.size


def auc_from_percentiles(p):
    """Mean percentile of positives against background equals ROC AUC."""
    p = np.asarray(p, dtype=float)
    return float(np.nanmean(p)) if np.isfinite(p).any() else np.nan


def summarize(percentiles, top=(0.05, 0.10, 0.20)):
    """AUC plus hit rates: share of finds that fall in the top-k fraction."""
    p = np.asarray(percentiles, dtype=float)
    p = p[np.isfinite(p)]
    out = {"n": int(p.size), "auc": auc_from_percentiles(p)}
    for k in top:
        out[f"hit_top{int(k * 100)}"] = float(np.mean(p >= 1 - k)) if p.size else np.nan
    return out


def spatial_percentiles(score_by_day, finds, candidate_mask):
    """score_by_day: {date: 2-D array}; finds: iterable of (date, row, col)."""
    out = []
    for date, r, c in finds:
        surf = score_by_day.get(date)
        if surf is None:
            out.append(np.nan)
            continue
        out.append(percentile_of(surf[r, c], surf[candidate_mask]))
    return np.array(out)


def temporal_percentiles(series_by_cell, finds, season_days):
    """series_by_cell(row, col) -> (dates, scores); season_days filters comparison days."""
    out = []
    for date, r, c in finds:
        dates, scores = series_by_cell(r, c)
        dates = np.asarray(dates)
        scores = np.asarray(scores, dtype=float)
        sel = np.isin(dates, season_days)
        hit = dates == date
        if not hit.any():
            out.append(np.nan)
            continue
        out.append(percentile_of(scores[hit][0], scores[sel & ~hit]))
    return np.array(out)
