"""Forest-type classification tests on synthetic arrays (no network, no data files)."""
from datetime import date

import numpy as np

from jurcki.sources import forest_type as ft


def test_classify_splits_conifer_and_broadleaf():
    on = np.array([0.85, 0.85, 0.85, 0.30, np.nan, 0.85])
    off = np.array([0.70, 0.30, np.nan, 0.65, 0.65, 0.45])
    wc = np.array([True, True, True, True, True, False])
    tree, valid, conifer = ft.classify(on, off, wc)
    assert tree.tolist() == [True, True, True, False, True, False]   # low leaf-on NDVI dropped, NaN kept
    assert valid.tolist() == [True, True, False, False, True, False]
    assert conifer.tolist() == [True, False, False, False, True, False]


def test_classify_threshold_is_strict_and_adjustable():
    on, off, wc = np.full(2, 0.8), np.array([0.5, 0.51]), np.ones(2, bool)
    assert ft.classify(on, off, wc, conifer_min=0.5)[2].tolist() == [False, True]
    assert ft.classify(on, off, wc, conifer_min=0.4)[2].tolist() == [True, True]


def test_cell_layers_ratios_and_min_area():
    comp = {"ndvi_on": np.full((2, 2), 0.8, "float32"),
            "ndvi_off": np.array([[0.7, 0.3], [0.3, np.nan]], "float32")}
    s = ft.pixel_sums(comp, np.ones((2, 2), bool))
    cell = {k: np.array([[v.mean()]]) for k, v in s.items()}     # one cell of 4 pixels
    out = ft.cell_layers(cell)
    assert np.isclose(out["conifer_frac"][0, 0], 1 / 3)           # 1 of 3 valid tree pixels
    assert np.isclose(out["ndvi_leafoff"][0, 0], (0.7 + 0.3 + 0.3) / 3)
    assert np.isclose(out["ndvi_summer"][0, 0], 0.8)
    assert np.isclose(out["forest_type_valid"][0, 0], 0.75)
    empty = ft.cell_layers({k: np.zeros((1, 1)) for k in s})
    assert all(np.isnan(v).all() for v in empty.values())


def test_gmm_split_finds_crossover_of_two_modes():
    rng = np.random.default_rng(0)
    vals = np.r_[rng.normal(0.3, 0.05, 5000), rng.normal(0.7, 0.05, 3000)]
    thr, share = ft.gmm_split(vals)
    assert 0.45 < thr < 0.55 and abs(share - 3 / 8) < 0.03


def test_ndvi_masked_applies_scene_classification():
    red = np.full((1, 12), 500, "uint16")
    nir = np.full((1, 12), 3000, "uint16")
    scl = np.full((1, 12), 4, "uint8")
    scl[0, 0], scl[0, 11] = 9, 11            # cloud and snow at the ends
    v = ft.ndvi_masked(red, nir, scl)
    assert np.isnan(v[0, :3]).all() and np.isnan(v[0, -3:]).all()   # 2 px dilation
    assert np.allclose(v[0, 3:-3], 2500 / 3500)


def test_pick_scenes_prefers_new_coverage():
    w = np.ones((4, 4))
    a, b, c = (np.zeros((4, 4), bool) for _ in range(3))
    a[:, :3], b[:, :3], c[:, 3:] = True, True, True
    picked = ft.pick_scenes({"a": a, "b": b, "c": c}, w, n_max=2, depth=1)
    assert sorted(picked) == ["a", "c"] or sorted(picked) == ["b", "c"]
    assert ft.pick_scenes({"a": a}, w, n_max=3, depth=1) == ["a"]


def test_seasons_and_scene_dates():
    assert ft.scene_date("S2B_33TUL_20230714_0_L2A") == date(2023, 7, 14)
    assert ft.in_season(date(2022, 7, 1), "summer") and not ft.in_season(date(2022, 9, 1), "summer")
    assert ft.in_season(date(2022, 12, 20), "winter") and ft.in_season(date(2023, 4, 5), "winter")
    assert not ft.in_season(date(2022, 11, 14), "winter") and not ft.in_season(date(2023, 4, 6), "winter")
