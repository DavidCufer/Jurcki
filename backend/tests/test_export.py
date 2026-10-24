"""Export tests: PNG placement (Web Mercator), colours, hotspots, meta, reports.  No network, no data files."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pyproj import Transformer

from jurcki import config, export, grid

TRIGLAV = (13.8366, 46.3783)
LJUBLJANA = (14.5058, 46.0569)
R = 6378137.0


# --- independent Web-Mercator math (what MapLibre does with an image quad) ----------------
def mx(lon):
    return R * math.radians(lon)


def my(lat):
    return R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def expected_pixel(lon, lat, bounds, w, h):
    """Fractional (row, col) of lon/lat inside a PNG whose outer edges are ``bounds``."""
    col = (mx(lon) - mx(bounds[0])) / (mx(bounds[2]) - mx(bounds[0])) * w
    row = (my(bounds[3]) - my(lat)) / (my(bounds[3]) - my(bounds[1])) * h
    return row, col


def to_3794(lon, lat):
    return Transformer.from_crs("EPSG:4326", config.CRS, always_xy=True).transform(lon, lat)


@pytest.fixture(scope="module")
def full_mask():
    return np.ones(grid.grid_shape(), bool)


@pytest.fixture(scope="module")
def geo():
    return export.png_geometry()


# --- geometry -------------------------------------------------------------------------------
def test_geometry_size_and_square_pixels(geo):
    assert 2000 <= geo["nx"] <= 2600
    b = geo["bounds"]
    # Slovenia is inside the PNG
    assert b[0] < 13.38 and b[2] > 16.60 and b[1] < 45.42 and b[3] > 46.88
    # Mercator pixels are square: width/height in Mercator metres matches pixel counts
    wm = mx(b[2]) - mx(b[0])
    hm = my(b[3]) - my(b[1])
    assert wm / geo["nx"] == pytest.approx(hm / geo["ny"], rel=1e-5)
    # ~100-150 m on the ground at 46 N
    ground = wm / geo["nx"] * math.cos(math.radians(46))
    assert 90 < ground < 150


def test_lonlat_to_pixel_matches_independent_formula(geo):
    for lon, lat in (TRIGLAV, LJUBLJANA):
        r, c = export.lonlat_to_pixel(lon, lat)
        er, ec = expected_pixel(lon, lat, geo["bounds"], geo["nx"], geo["ny"])
        assert r == pytest.approx(er, abs=1e-6) and c == pytest.approx(ec, abs=1e-6)
    # corners
    r, c = export.lonlat_to_pixel(geo["bounds"][0], geo["bounds"][3])
    assert (float(r), float(c)) == pytest.approx((0.0, 0.0), abs=1e-6)
    r, c = export.lonlat_to_pixel(geo["bounds"][2], geo["bounds"][1])
    assert (float(r), float(c)) == pytest.approx((geo["ny"], geo["nx"]), abs=1e-6)


# --- placement of values --------------------------------------------------------------------
def test_resampled_value_equals_field_at_pixel_centre(geo, full_mask):
    """A linear (sawtooth) field in EPSG:3794 metres is reproduced exactly at pixel centres."""
    x, y = grid.cell_xy()
    period = 20_000.0
    f = (((x - config.GRID_ORIGIN[0]) % period) / period * 0.8 + 0.1).astype("float32")
    pix = export.resample_to_png(f, full_mask)
    for lon, lat in (TRIGLAV, LJUBLJANA):
        er, ec = expected_pixel(lon, lat, geo["bounds"], geo["nx"], geo["ny"])
        r, c = int(er), int(ec)
        # pixel centre in lon/lat via Mercator, then to the model CRS
        clon = math.degrees((mx(geo["bounds"][0]) + (c + 0.5) * geo["px"]) / R)
        clat = math.degrees(2 * math.atan(math.exp((my(geo["bounds"][3]) - (r + 0.5) * geo["px"]) / R))
                            - math.pi / 2)
        px_, _ = to_3794(clon, clat)
        want = ((px_ - config.GRID_ORIGIN[0]) % period) / period * 0.8 + 0.1
        # the cell-centre sampling of the field is piecewise-linear with 250 m steps
        assert pix[r, c] == pytest.approx(want, abs=0.001)


def _block_score(lon, lat, half=4):
    """Zeros with a (2*half)^2-cell block of 0.9 around the cell containing lon/lat."""
    rows, cols = grid.grid_shape()
    s = np.zeros((rows, cols), "float32")
    r0, c0 = grid.lonlat_to_rowcol(lon, lat)
    s[r0 - half:r0 + half, c0 - half:c0 + half] = 0.9
    # block centre (cell corner shared by the four central cells) in EPSG:3794
    cx = config.GRID_ORIGIN[0] + c0 * config.GRID_RES
    cy = config.GRID_ORIGIN[1] - r0 * config.GRID_RES
    return s, (cx, cy), (r0, c0)


@pytest.mark.parametrize("where", [TRIGLAV, LJUBLJANA])
def test_png_block_lands_on_the_right_pixels(where, geo, full_mask, tmp_path):
    score, (cx, cy), (r0, c0) = _block_score(*where)
    path = export.write_score_png(score, tmp_path / "s.png", full_mask)
    im = np.asarray(Image.open(path))
    assert im.shape == (geo["ny"], geo["nx"], 4)

    # the pixel at the known lon/lat sits inside the block and carries the block colour
    er, ec = expected_pixel(*where, geo["bounds"], geo["nx"], geo["ny"])
    px = im[int(er), int(ec)]
    lut = export.colour_lut(export.LEVELS[0])
    want = lut[int(round(0.9 * (export.LEVELS[0] - 1)))]
    assert tuple(px) == tuple(want) and px[3] > 150

    # centroid of the opaque blob == block centre, projected with pyproj + Mercator math
    rows, cols = np.nonzero(im[..., 3] > 128)
    t = Transformer.from_crs(config.CRS, "EPSG:4326", always_xy=True)
    blon, blat = t.transform(cx, cy)
    br, bc = expected_pixel(blon, blat, geo["bounds"], geo["nx"], geo["ny"])
    assert rows.mean() + 0.5 == pytest.approx(br, abs=1.5)
    assert cols.mean() + 0.5 == pytest.approx(bc, abs=1.5)
    # block is 8 cells = 2 km wide on the ground -> about 2000/105 pixels
    assert 0.75 * 2000 / 105 < (cols.max() - cols.min() + 1) < 1.35 * 2000 / 105 + 3
    # and nothing else is drawn
    assert rows.max() - rows.min() < 40 and cols.max() - cols.min() < 40


def test_grid_cell_value_matches_pixel_value(full_mask):
    """Inside a constant 8x8-cell block the pixel value equals the cell value exactly."""
    score, _, _ = _block_score(*LJUBLJANA)
    score[score > 0] = 0.7
    pix = export.resample_to_png(score, full_mask)
    r, c = export.lonlat_to_pixel(*LJUBLJANA)
    assert pix[int(r), int(c)] == pytest.approx(0.7, abs=1e-5)
    r0, c0 = grid.lonlat_to_rowcol(*LJUBLJANA)
    assert score[r0, c0] == pytest.approx(0.7)


# --- masking / colours ------------------------------------------------------------------------
def test_mask_and_threshold_transparent(tmp_path):
    rows, cols = grid.grid_shape()
    lj_r, lj_c = grid.lonlat_to_rowcol(*LJUBLJANA)
    score = np.full((rows, cols), 0.8, "float32")
    score[:, :lj_c - 100] = 0.01               # west of here: below MIN_SCORE
    score[:lj_r - 100, :] = np.nan             # north of here: no data
    mask = np.ones((rows, cols), bool)
    mask[:, lj_c + 100:] = False               # east of here: outside the country

    path = export.write_score_png(score, tmp_path / "m.png", mask)
    im = np.asarray(Image.open(path))

    def alpha_at(lon, lat):
        r, c = export.lonlat_to_pixel(lon, lat)
        return im[int(r), int(c), 3]

    assert alpha_at(*LJUBLJANA) > 200                  # inside, score 0.8 -> opaque
    assert alpha_at(LJUBLJANA[0] + 0.4, LJUBLJANA[1]) == 0   # ~30 km east: outside mask
    assert alpha_at(LJUBLJANA[0] - 0.4, LJUBLJANA[1]) == 0   # ~30 km west: score 0.01
    assert alpha_at(LJUBLJANA[0], LJUBLJANA[1] + 0.4) == 0   # ~44 km north: NaN


def test_colour_ramp_monotone_alpha_and_legend():
    lut = export.colour_lut(64)
    a = lut[:, 3].astype(int)
    assert (np.diff(a) >= 0).all() and a[0] == 0 and a[-1] == round(export.ALPHA_MAX * 255)
    assert a[round(0.03 * 63)] == 0
    # luminance falls monotonically from low to high scores (light yellow -> dark red)
    lum = lut[:, :3] @ np.array([0.2126, 0.7152, 0.0722])
    assert (np.diff(lum) <= 0.5).all() and lum[0] > lum[-1] + 100
    # legend: ends at 0 and 1, hex colours, ascending
    vals = [v for v, _ in export.LEGEND]
    assert vals[0] == 0.0 and vals[-1] == 1.0 and vals == sorted(vals)
    assert all(len(h) == 7 and h[0] == "#" for _, h in export.LEGEND)
    # lowest / highest LUT colours are the legend ends
    assert tuple(lut[0, :3]) == export._hex_rgb(export.LEGEND[0][1])
    assert tuple(lut[-1, :3]) == export._hex_rgb(export.LEGEND[-1][1])


# --- hotspots ---------------------------------------------------------------------------------
def test_hotspots_declustered_sorted_and_capped():
    rows, cols = grid.grid_shape()
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    s = gaussian_filter(rng.random((rows, cols)), 6)
    s = ((s - s.min()) / (s.max() - s.min())).astype("float32")
    cells = export.pick_hotspots(s, min_score=0.3)
    assert 0 < len(cells) <= export.HOTSPOT_MAX
    vals = [s[r, c] for r, c in cells]
    assert vals == sorted(vals, reverse=True)
    pts = np.array(cells, float) * config.GRID_RES
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    d[np.diag_indices(len(pts))] = np.inf
    assert d.min() >= export.HOTSPOT_MIN_SEP_M - 1e-6
    assert max(vals) == pytest.approx(float(s.max()))      # the global maximum is always kept
    assert min(vals) >= 0.3
    # NaN and masked cells are never picked
    s2 = s.copy()
    s2[np.unravel_index(np.argmax(s), s.shape)] = np.nan
    m = np.ones_like(s, bool)
    m[:, : cols // 2] = False
    assert all(c >= cols // 2 for _, c in export.pick_hotspots(s2, m, min_score=0.3))
    assert export.pick_hotspots(np.zeros_like(s)) == []


def test_export_day_writes_png_and_hotspots(tmp_path, full_mask):
    score, _, _ = _block_score(*TRIGLAV, half=10)
    habitat = np.full_like(score, 0.8)
    trigger = np.where(score > 0, 1.0, 0.0).astype("float32")
    elev = np.full_like(score, 1234.4)
    res = export.export_day("2026-10-24", score, habitat, trigger, tmp_path, mask=full_mask, elev=elev)
    assert (tmp_path / "score_2026-10-24.png").exists()
    assert res["n_hotspots"] > 0
    assert (tmp_path / "score_2026-10-24.png").stat().st_size < 2_000_000
    gj = json.loads((tmp_path / "hotspots_2026-10-24.geojson").read_text())
    feats = gj["features"]
    assert len(feats) == res["n_hotspots"]
    scores = [f["properties"]["score"] for f in feats]
    assert scores == sorted(scores, reverse=True)
    for f in feats:
        p = f["properties"]
        assert set(p) == {"score", "habitat", "trigger", "elev"}
        assert 0 <= p["score"] <= 1 and p["elev"] == 1234
        lon, lat = f["geometry"]["coordinates"]
        assert abs(lon - TRIGLAV[0]) < 0.1 and abs(lat - TRIGLAV[1]) < 0.1
    # hotspot coordinates sit on cell centres: round-trip to the same cell
    lon, lat = feats[0]["geometry"]["coordinates"]
    r, c = grid.lonlat_to_rowcol(lon, lat)
    assert score[r, c] == pytest.approx(feats[0]["properties"]["score"], abs=1e-3)


def test_export_day_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        export.export_day("2026-10-24", np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), tmp_path,
                          mask=np.ones(grid.grid_shape(), bool), elev=None)


# --- meta / reports ---------------------------------------------------------------------------
def test_write_meta_follows_contract(tmp_path, geo):
    p = export.write_meta(["2026-10-25", "2026-10-24"], "2026-10-25", tmp_path, extra={"demo": True})
    m = json.loads(p.read_text())
    assert m["schema_version"] == 1
    assert m["days"] == ["2026-10-24", "2026-10-25"] and m["first_forecast_day"] == "2026-10-25"
    assert m["bounds"] == geo["bounds"] and m["demo"] is True
    assert m["legend"] == [list(x) for x in export.LEGEND]
    import re
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", m["generated"])   # has an offset


def test_write_reports(tmp_path):
    df = pd.DataFrame([
        {"lon": 14.3, "lat": 46.0, "date": pd.Timestamp("2026-10-20"), "source": "forum",
         "precision_m": 5000.0, "text": "jurčki", "url": "https://x.si/a"},
        {"lon": 14.8, "lat": 46.1, "date": "2026-10-21", "source": "gbif",
         "precision_m": np.nan, "text": None, "url": None},
        {"lon": np.nan, "lat": 46.1, "date": "2026-10-21", "source": "gbif",
         "precision_m": 30, "text": None, "url": None},
    ])
    p = export.write_reports(df, tmp_path)
    feats = json.loads(p.read_text())["features"]
    assert len(feats) == 2
    assert feats[0]["geometry"]["coordinates"] == [14.3, 46.0]
    assert feats[0]["properties"] == {"date": "2026-10-20", "source": "forum", "precision_m": 5000,
                                      "text": "jurčki", "url": "https://x.si/a"}
    assert feats[1]["properties"]["precision_m"] is None and feats[1]["properties"]["url"] is None
    assert json.loads(export.write_reports(df.iloc[:0], tmp_path).read_text())["features"] == []
