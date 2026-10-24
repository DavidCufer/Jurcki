"""Grid helper tests (no network, no data files)."""
import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import Resampling

from jurcki import config, grid


def test_shape_transform_bounds():
    assert grid.grid_shape() == (700, 1040)
    assert grid.grid_shape(25) == (7000, 10400)
    t = grid.grid_transform()
    assert (t.c, t.f, t.a, t.e) == (370_000, 200_000, 250, -250)
    assert grid.grid_bounds() == (370_000, 25_000, 630_000, 200_000)


def test_cell_centres_and_roundtrip():
    x, y = grid.cell_xy()
    assert x.shape == y.shape == grid.grid_shape()
    assert x[0, 0] == 370_125 and y[0, 0] == 199_875
    assert x[-1, -1] == 629_875 and y[-1, -1] == 25_125
    lon, lat = grid.cell_lonlat()
    r, c = grid.lonlat_to_rowcol(lon[::97, ::89], lat[::97, ::89])
    rows, cols = np.meshgrid(np.arange(0, 700, 97), np.arange(0, 1040, 89), indexing="ij")
    assert (r == rows).all() and (c == cols).all()


def test_cell_lonlat_covers_slovenia():
    lon, lat = grid.cell_lonlat()
    b = config.BBOX_WGS84
    assert lon.min() < b[0] and lon.max() > b[2]
    assert lat.min() < b[1] and lat.max() > b[3]
    r, c = grid.lonlat_to_rowcol(14.5058, 46.0569)  # Ljubljana
    assert 0 <= r < 700 and 0 <= c < 1040


def test_reproject_identity_and_constant():
    a = np.random.default_rng(0).random(grid.grid_shape()).astype("float32")
    out = grid.reproject_to_grid(a, grid.grid_transform(), config.CRS,
                                 resampling=Resampling.nearest)
    assert np.allclose(out, a)
    src = np.full((200, 400), 7.0, dtype="float32")   # WGS84 raster covering the grid
    out = grid.reproject_to_grid(src, from_origin(13.0, 47.2, 0.01, 0.01), config.CRS_WGS84)
    assert out.shape == grid.grid_shape() and np.allclose(out, 7.0)


def test_block_mean():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    assert grid.block_mean(a, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    a[0, 0] = np.nan
    assert grid.block_mean(a, 2)[0, 0] == np.float32(10 / 3)


def test_land_mask_synthetic_polygon():
    # 0.5 x 0.3 degree box around Ljubljana
    box = {"type": "Polygon", "coordinates": [[(14.25, 45.9), (14.75, 45.9),
                                                (14.75, 46.2), (14.25, 46.2), (14.25, 45.9)]]}
    m = grid.land_mask(box)
    assert m.shape == grid.grid_shape() and m.dtype == bool
    assert 0 < m.sum() < m.size
    r, c = grid.lonlat_to_rowcol(14.5, 46.05)
    assert m[r, c]
    r, c = grid.lonlat_to_rowcol(15.5, 46.05)
    assert not m[r, c]
