"""Terrain helper tests on synthetic surfaces (no network)."""
import numpy as np

from jurcki.sources import dem


def test_tile_name():
    assert dem.tile_name(46, 14) == "Copernicus_DSM_COG_10_N46_00_E014_00_DEM"


def test_needed_tiles_cover_slovenia():
    tiles = dem.needed_tiles()
    assert (46, 14) in tiles and (45, 15) in tiles and (46, 16) in tiles


def test_horn_gradient_plane():
    y, x = np.mgrid[0:20, 0:20].astype("float32")
    z = 0.5 * x - 0.25 * y       # rises east, falls south -> north gradient = +0.25
    dzdx, dzdy = dem.horn_gradient(z, 1.0)
    assert np.allclose(dzdx[5:-5, 5:-5], 0.5) and np.allclose(dzdy[5:-5, 5:-5], 0.25)


def test_tpi_and_tri_flat_and_peak():
    z = np.zeros((41, 41), dtype="float32")
    assert np.allclose(dem.tpi(z, 100, 25), 0) and np.allclose(dem.riley_tri(z), 0)
    z[20, 20] = 10
    assert dem.tpi(z, 100, 25)[20, 20] > 9
    assert dem.riley_tri(z)[20, 20] > 0


def test_flow_accumulation_conserves_and_fills_pits():
    z = np.add.outer(np.arange(30.0)[::-1], np.arange(30.0) * 0.01)
    z[10:14, 10:14] -= 5          # closed depression
    acc = dem.flow_accumulation(z)
    assert acc.shape == z.shape and acc.min() >= 1
    # every cell drains to exactly one border outlet
    border = np.ones_like(z, dtype=bool)
    border[1:-1, 1:-1] = False
    assert acc[border].sum() == z.size
    assert acc.max() > 50
