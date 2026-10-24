"""Shared constants: study area, grid, paths.

Every module uses the same model grid so layers stack cell-for-cell.
Grid is defined in EPSG:3794 (Slovenia D96/TM, metres).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"          # downloaded source files (git-ignored)
CACHE = DATA / "cache"      # HTTP / API caches (git-ignored)
INTERIM = DATA / "interim"  # big derived rasters (git-ignored)
PROCESSED = DATA / "processed"  # small derived products (committed)
OUTPUT = DATA / "output"    # maps served to the Android app

CRS = "EPSG:3794"
CRS_WGS84 = "EPSG:4326"

# Slovenia bounding box with a small buffer (lon/lat)
BBOX_WGS84 = (13.30, 45.40, 16.65, 46.90)  # minx, miny, maxx, maxy

# Model grid in EPSG:3794. Slovenia spans roughly x 373k-625k, y 31k-195k.
GRID_ORIGIN = (370_000, 200_000)  # top-left corner (x, y)
GRID_EXTENT = (260_000, 175_000)  # width, height in metres
GRID_RES = 250                    # metres per cell

for _p in (RAW, CACHE, INTERIM, PROCESSED, OUTPUT):
    _p.mkdir(parents=True, exist_ok=True)
