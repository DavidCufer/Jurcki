"""Literature-based prior: habitat suitability x fruiting trigger.

All functions are vectorised numpy and return scores in [0, 1]. They are the
starting point that the trained model is compared against (and can be
blended with), so every number here is a documented, tunable parameter.

Sources for the defaults:
* Bielefeld beech monitoring 2015-2024 (bioRxiv 2025.12.12.693895):
  peak fruiting at ~13 C mean temperature over the previous 20 days,
  fruiting rises linearly with rain accumulated over 26 days.
* Salerni et al. 2023 (Ital. J. Mycol.): sudden Tmax spikes suppress yield.
* Slovenian forager knowledge / mycological societies: jurcki need
  mycorrhizal hosts (spruce, beech, fir, oak, chestnut, pine), fruit roughly
  300-1600 m, move uphill through summer and downhill in autumn, favour
  forest edges, clearings and paths over dense closed stands.
"""
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class PriorParams:
    # trigger
    t_opt: float = 13.0         # C, optimum 20-day mean temperature
    t_sd: float = 4.0           # C, width of the temperature response
    t_window: int = 20          # days
    p_window: int = 26          # days
    p_half: float = 60.0        # mm over p_window giving 50 % response
    pulse_min: float = 20.0     # mm in 3 days counted as a soaking rain
    pulse_lag: tuple = (7, 16)  # days after a soaking rain when flushes appear
    heat_tmax: float = 30.0     # C, Tmax that suppresses fruiting
    heat_window: int = 10       # days
    frost_tmin: float = -1.0    # C
    frost_window: int = 7       # days
    # habitat
    elev_range: tuple = (250.0, 1700.0)  # m, outside -> ~0
    elev_opt_by_month: dict = field(default_factory=lambda: {
        5: 450, 6: 600, 7: 800, 8: 1000, 9: 950, 10: 700, 11: 450,
    })
    elev_sd: float = 450.0      # m
    slope_max: float = 40.0     # deg, steeper -> unsuitable
    edge_opt: float = 60.0      # m inside the forest edge
    treecover_min: float = 0.3  # fraction of cell under trees

    def to_dict(self):
        return asdict(self)


def _rolling_sum(x, window, axis=0):
    """Trailing sum over `window` steps along `axis` (first steps use what exists)."""
    c = np.cumsum(x, axis=axis)
    shifted = np.zeros_like(c)
    sl_dst = [slice(None)] * x.ndim
    sl_src = [slice(None)] * x.ndim
    sl_dst[axis] = slice(window, None)
    sl_src[axis] = slice(None, -window)
    shifted[tuple(sl_dst)] = c[tuple(sl_src)]
    return c - shifted


def _rolling_mean(x, window, axis=0):
    n = np.minimum(np.arange(1, x.shape[axis] + 1), window).astype(float)
    shape = [1] * x.ndim
    shape[axis] = -1
    return _rolling_sum(x, window, axis) / n.reshape(shape)


def _rolling_max(x, window, axis=0):
    from scipy.ndimage import maximum_filter1d
    # origin shifts the filter so it only looks backwards in time
    return maximum_filter1d(x, size=window, axis=axis, origin=(window - 1) // 2, mode="nearest")


def _rolling_min(x, window, axis=0):
    return -_rolling_max(-x, window, axis)


def weather_features(tmean, tmin, tmax, precip, p=PriorParams()):
    """Daily weather (time first) -> dict of lagged features with same shape."""
    rain3 = _rolling_sum(precip, 3)
    pulse = (rain3 >= p.pulse_min).astype(float)
    lo, hi = p.pulse_lag
    # 1 if a soaking rain ended between `lo` and `hi` days ago
    pulse_in_lag = _rolling_max(pulse, hi - lo + 1)
    pulse_in_lag = np.roll(pulse_in_lag, lo, axis=0)
    pulse_in_lag[:lo] = 0
    return {
        "t20": _rolling_mean(tmean, p.t_window),
        "p26": _rolling_sum(precip, p.p_window),
        "p7": _rolling_sum(precip, 7),
        "pulse": pulse_in_lag,
        "tmax10": _rolling_max(tmax, p.heat_window),
        "tmin7": _rolling_min(tmin, p.frost_window),
    }


def trigger_score(f, p=PriorParams()):
    """Fruiting trigger in [0, 1] from weather_features output."""
    temp = np.exp(-0.5 * ((f["t20"] - p.t_opt) / p.t_sd) ** 2)
    rain = f["p26"] / (f["p26"] + p.p_half)
    rain = 0.6 * rain + 0.4 * f["pulse"]
    heat = np.clip(1 - (f["tmax10"] - p.heat_tmax) / 5.0, 0.2, 1.0)
    frost = np.clip((f["tmin7"] - p.frost_tmin) / 3.0 + 1.0, 0.0, 1.0)
    return temp * rain * heat * frost


def habitat_score(elev, slope, treecover, edge_dist=None, northness=None, month=9,
                  p=PriorParams()):
    """Static suitability in [0, 1] for a given month (elevation band and aspect shift).

    edge_dist: metres to forest edge, positive inside forest (None -> ignored).
    northness: cos(aspect); north slopes help in hot summer months, south in autumn.
    """
    lo, hi = p.elev_range
    opt = p.elev_opt_by_month.get(month, 700)
    s_elev = np.exp(-0.5 * ((elev - opt) / p.elev_sd) ** 2)
    s_elev = np.where((elev < lo - 150) | (elev > hi + 150), 0.0, s_elev)
    s_slope = np.clip(1 - (slope / p.slope_max) ** 2, 0.0, 1.0)
    s_tree = np.clip((treecover - p.treecover_min) / (1 - p.treecover_min), 0.0, 1.0) ** 0.5
    score = s_elev * s_slope * s_tree
    if edge_dist is not None:
        # best a little inside the edge, still fine deep in open old stands
        s_edge = 0.6 + 0.4 * np.exp(-0.5 * ((edge_dist - p.edge_opt) / 150.0) ** 2)
        score = score * np.where(edge_dist < -30, 0.2, s_edge)
    if northness is not None:
        sign = 1.0 if month in (6, 7, 8) else (-1.0 if month in (10, 11) else 0.0)
        score = score * (1 + 0.15 * sign * northness)
    return np.clip(score, 0.0, 1.0)


def season_factor(month):
    """Rough seasonal prior of jurcki fruiting in Slovenia (0-1)."""
    table = {4: 0.05, 5: 0.25, 6: 0.55, 7: 0.6, 8: 0.85, 9: 1.0, 10: 0.9, 11: 0.35}
    return table.get(month, 0.0)
