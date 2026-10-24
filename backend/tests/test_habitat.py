import numpy as np

from jurcki.model import habitat


def test_block_ids_same_block_and_distinct():
    r = np.array([0, 39, 40, 0, 79])
    c = np.array([0, 39, 0, 40, 79])
    b = habitat.block_ids(r, c, 40, n_cols=1040)
    assert b[0] == b[1]
    assert len({b[0], b[2], b[3], b[4]}) == 4


def test_assign_folds_keeps_blocks_together_and_uses_all_folds():
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 700, 3000), rng.integers(0, 1040, 3000)
    blocks = habitat.block_ids(rows, cols, 40)
    folds = habitat.assign_folds(blocks, n_folds=5, seed=3)
    for b in np.unique(blocks):
        assert len(set(folds[blocks == b])) == 1
    assert set(folds) == set(range(5))
    assert np.array_equal(folds, habitat.assign_folds(blocks, n_folds=5, seed=3))
    assert not np.array_equal(folds, habitat.assign_folds(blocks, n_folds=5, seed=4))


def test_sample_background_inside_eligible_and_distinct():
    elig = np.zeros((50, 60), bool)
    elig[10:30, 20:40] = True
    r, c = habitat.sample_background(elig, 100, np.random.default_rng(1))
    assert len(r) == 100 and elig[r, c].all()
    assert len(set(zip(r, c))) == 100
    r, c = habitat.sample_background(elig, 10_000, np.random.default_rng(1))
    assert len(r) == elig.sum()  # capped at the number of eligible cells


def test_sample_background_follows_probability_surface():
    elig = np.ones((20, 20), bool)
    prob = np.zeros((20, 20))
    prob[:, :10] = 1.0
    r, c = habitat.sample_background(elig, 150, np.random.default_rng(2), prob)
    assert (c < 10).all()


def test_effort_density_sums_to_one_and_peaks_at_finds():
    d = habitat.effort_density([10, 10], [10, 11], (40, 40), 3.0, mix=0.8)
    assert abs(d.sum() - 1) < 1e-6
    assert d[10, 10] > d[35, 35]


def test_sample_disc_cells_within_radius_and_eligible():
    elig = np.ones((40, 40), bool)
    elig[:, :18] = False
    r, c = habitat.sample_disc_cells(20, 20, elig, 5, 6, np.random.default_rng(0))
    assert len(r) == 6 and elig[r, c].all()
    assert (((r - 20) ** 2 + (c - 20) ** 2) <= 25).all()
    # disc touching the grid edge and with no eligible cell
    r, c = habitat.sample_disc_cells(0, 0, np.zeros((10, 10), bool), 3, 6, np.random.default_rng(0))
    assert len(r) == 0


def test_disc_mean_normalises_over_forest_cells():
    surf = np.zeros((30, 30), "float32")
    forest = np.zeros((30, 30), "float32")
    forest[:, 15:] = 1
    surf[:, 15:] = 0.8
    m = habitat.disc_mean(surf, forest, 4)
    assert abs(m[15, 17] - 0.8) < 1e-3
