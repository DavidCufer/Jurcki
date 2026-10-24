"""Generate a tiny synthetic dataset that follows the app data contract (docs/ARCHITECTURE.md).

    source backend/.venv/bin/activate && python data/output/sample/make_sample.py

The values are made up (smooth random fields inside a rough Slovenia outline) - they exist only
so the Android app can be tested without the real backend.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
BOUNDS = [13.37, 45.42, 16.61, 46.88]  # minLon, minLat, maxLon, maxLat (EPSG:4326), adjusted below
PX_DEG = 0.01  # ~1 km pixels: tiny, but plenty for testing (columns: degrees of longitude)
R = 6378137.0  # Web Mercator sphere radius: the contract wants rows spaced in EPSG:3857
FIRST_DAY = date(2026, 10, 24)
N_DAYS = 4
LEGEND = [[0.0, "#FFF7BC"], [0.35, "#FEC44F"], [0.65, "#EC7014"], [1.0, "#7F0000"]]

# Very rough outline of Slovenia (lon, lat) - only used as a mask.
OUTLINE = [
    (13.42, 45.60), (13.38, 45.85), (13.60, 46.20), (13.65, 46.45), (14.20, 46.50), (14.55, 46.43),
    (15.00, 46.65), (15.60, 46.68), (16.10, 46.87), (16.55, 46.85), (16.50, 46.55), (16.20, 46.40),
    (15.70, 46.20), (15.65, 45.85), (15.30, 45.72), (14.90, 45.48), (14.50, 45.50), (14.20, 45.45),
    (13.95, 45.48), (13.70, 45.45),
]

def _my(lat_deg):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def _lat(y):
    return np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


rng = np.random.default_rng(42)
W = int(round((BOUNDS[2] - BOUNDS[0]) / PX_DEG))
PX_M = R * np.radians(PX_DEG)                              # square pixels in Mercator metres
y_bot = _my(BOUNDS[1])
H = int(np.ceil((_my(BOUNDS[3]) - y_bot) / PX_M))
BOUNDS[3] = round(float(_lat(y_bot + H * PX_M)), 6)        # bounds = outer pixel edges
lon = BOUNDS[0] + (np.arange(W) + 0.5) * PX_DEG          # column -> lon
lat = _lat(y_bot + (H - np.arange(H) - 0.5) * PX_M)      # row 0 = north, Mercator-spaced
LON, LAT = np.meshgrid(lon, lat)


def blobs(n: int, sigma_lo: float, sigma_hi: float) -> np.ndarray:
    f = np.zeros((H, W))
    for _ in range(n):
        cx = rng.uniform(BOUNDS[0], BOUNDS[2])
        cy = rng.uniform(BOUNDS[1], BOUNDS[3])
        s = rng.uniform(sigma_lo, sigma_hi)
        f += rng.uniform(0.5, 1.0) * np.exp(-(((LON - cx) / s) ** 2 + ((LAT - cy) / (0.7 * s)) ** 2))
    return f / f.max()


# mask
img = Image.new("L", (W, H), 0)
px = [((x - BOUNDS[0]) / PX_DEG, (_my(BOUNDS[3]) - _my(y)) / PX_M) for x, y in OUTLINE]
ImageDraw.Draw(img).polygon(px, fill=255)
MASK = np.asarray(img) > 0

habitat = np.clip(blobs(40, 0.08, 0.25), 0, 1) ** 0.8
elev = 150 + 1500 * blobs(25, 0.1, 0.3)

legend_v = np.array([s[0] for s in LEGEND])
legend_rgb = np.array([[int(s[1][i:i + 2], 16) for i in (1, 3, 5)] for s in LEGEND], dtype=float)


def colorize(score: np.ndarray) -> np.ndarray:
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for c in range(3):
        rgba[..., c] = np.interp(score, legend_v, legend_rgb[:, c]).astype(np.uint8)
    alpha = np.clip((score - 0.05) / 0.25, 0, 1) * 0.9 * 255   # transparent where score ~ 0
    alpha[~MASK] = 0
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba


def top_cells(score: np.ndarray, k: int = 30, min_sep_px: int = 6):
    s = np.where(MASK, score, 0.0)
    order = np.argsort(s, axis=None)[::-1]
    picked: list[tuple[int, int]] = []
    for flat in order:
        r, c = divmod(int(flat), W)
        if s[r, c] < 0.2 or len(picked) >= k:
            break
        if all((r - pr) ** 2 + (c - pc) ** 2 >= min_sep_px ** 2 for pr, pc in picked):
            picked.append((r, c))
    return picked


days = [(FIRST_DAY + timedelta(days=i)).isoformat() for i in range(N_DAYS)]
trigger_base = blobs(12, 0.3, 0.8)
for i, d in enumerate(days):
    # the "weather" field drifts a little from day to day
    shift = int(round(i * 8))
    trigger = np.roll(trigger_base, shift, axis=1)
    trigger = np.clip(0.25 + 0.9 * trigger, 0, 1)
    score = habitat * trigger
    score = score / score[MASK].max()

    Image.fromarray(colorize(score), "RGBA").save(OUT / f"score_{d}.png", optimize=True)

    feats = []
    for r, c in top_cells(score):
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(float(lon[c]), 5), round(float(lat[r]), 5)]},
            "properties": {
                "score": round(float(score[r, c]), 3),
                "habitat": round(float(habitat[r, c]), 3),
                "trigger": round(float(trigger[r, c]), 3),
                "elev": int(round(float(elev[r, c]))),
            },
        })
    (OUT / f"hotspots_{d}.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False), encoding="utf-8")

reports = [
    # lon, lat, date, source, precision_m, text
    (14.31, 46.05, "2026-10-20", "forum", 5000, "Ob Pohorju prvi lepi jurčki po dežju."),
    (14.84, 46.12, "2026-10-21", "gbif", 30, "Boletus edulis (iNaturalist), bukov gozd."),
    (15.20, 46.35, "2026-10-19", "forum", 2000, "Slovenske gorice: nekaj mladih jurčkov, večinoma črvivih."),
    (13.95, 45.98, "2026-10-22", "news", 10000, "Gobarji poročajo o dobri rasti na Trnovski planoti."),
    (14.62, 45.69, "2026-10-18", "gbif", 100, "Boletus aereus, hrastov gozd, Kočevsko."),
]
(OUT / "reports.geojson").write_text(json.dumps({
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": {"date": dt, "source": src, "precision_m": prec, "text": text},
    } for x, y, dt, src, prec, text in reports],
}, ensure_ascii=False), encoding="utf-8")

(OUT / "meta.json").write_text(json.dumps({
    "schema_version": 1,
    "generated": f"{FIRST_DAY.isoformat()}T06:00:00+00:00",
    "days": days,
    "first_forecast_day": days[1],
    "bounds": BOUNDS,
    "legend": LEGEND,
}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"wrote {N_DAYS} days, {W}x{H}px, to {OUT}")
