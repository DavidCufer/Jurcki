import numpy as np

from jurcki.model import evaluate


def test_percentile_and_summary():
    assert evaluate.percentile_of(5, [1, 2, 3, 4]) == 1.0
    assert evaluate.percentile_of(0, [1, 2, 3, 4]) == 0.0
    assert evaluate.percentile_of(2, [1, 2, 3]) == 0.5
    s = evaluate.summarize([0.99, 0.5, 0.0])
    assert s["n"] == 3 and abs(s["auc"] - 0.4967) < 1e-3 and abs(s["hit_top5"] - 1 / 3) < 1e-9


def test_spatial_percentiles():
    surf = np.arange(9, dtype=float).reshape(3, 3)
    mask = np.ones((3, 3), bool)
    p = evaluate.spatial_percentiles({"d": surf}, [("d", 2, 2), ("x", 0, 0)], mask)
    assert abs(p[0] - (8.5 / 9)) < 1e-9 and np.isnan(p[1])
