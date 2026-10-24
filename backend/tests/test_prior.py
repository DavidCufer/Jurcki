import numpy as np

from jurcki.model import prior


def test_rolling_sum_trailing():
    x = np.arange(1, 6, dtype=float)
    assert prior._rolling_sum(x, 2).tolist() == [1, 3, 5, 7, 9]


def test_rolling_max_looks_backwards():
    x = np.array([0, 0, 5, 0, 0, 0], dtype=float)
    assert prior._rolling_max(x, 3).tolist() == [0, 0, 5, 5, 5, 0]


def test_trigger_responds_to_rain_and_temperature():
    days = 60
    t = np.full(days, 13.0)
    dry = prior.weather_features(t, t - 5, t + 5, np.zeros(days))
    rain = np.zeros(days)
    rain[40:42] = 25.0
    wet = prior.weather_features(t, t - 5, t + 5, rain)
    assert prior.trigger_score(dry)[-1] == 0
    s = prior.trigger_score(wet)
    assert s[50] > s[45] > 0       # pulse window starts 7 days after the rain
    hot = prior.weather_features(t + 12, t + 5, t + 20, rain)
    assert prior.trigger_score(hot)[50] < s[50]


def test_habitat_needs_forest_and_sane_elevation():
    elev = np.array([900.0, 900.0, 2400.0])
    slope = np.array([10.0, 10.0, 10.0])
    tree = np.array([0.9, 0.0, 0.9])
    h = prior.habitat_score(elev, slope, tree, month=8)
    assert h[0] > 0.5 and h[1] == 0 and h[2] == 0
