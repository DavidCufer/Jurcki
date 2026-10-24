"""Slovenian gazetteer: GeoNames SI + hand-curated mushroom regions.

Build once with ``python -m jurcki.geo.gazetteer`` (downloads GeoNames dumps
from download.geonames.org if they are not cached in ``data/raw/geonames``).
Outputs:

* ``data/processed/gazetteer_si.parquet``  one row per place/feature
* ``data/processed/regions_si.geojson``    the curated regions as ellipse polygons

Columns of the parquet: gid, name, ascii_name, norm_name, alt_names
(``|``-joined, normalised), fclass, fcode, kind, lon, lat, population,
elevation, radius_m, a_m, b_m, az_deg (ellipse, curated only), source
(``geonames``/``curated``), weight, strict, basis.

Radii are *approximate* by design: villages 1.5 km, towns 1.5 km + 12*sqrt(pop)
metres, municipalities 4.5 km + 20*sqrt(pop) (cap 12 km), peaks 2-4 km, etc.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..sources._http import download
from .regions_curated import CURATED
from .slovene import fold

GN_DIR = config.RAW / "geonames"
GN_URL = "https://download.geonames.org/export/dump/"
GAZ_PATH = config.PROCESSED / "gazetteer_si.parquet"
REGIONS_PATH = config.PROCESSED / "regions_si.geojson"

_COLS = ("gid name ascii alt lat lon fclass fcode cc cc2 a1 a2 a3 a4 pop elev dem tz mod").split()

# GeoNames feature code -> our kind.  Everything else is ignored (rivers,
# hotels, stations, airports, springs, caves ...).
KIND = {
    "PPLC": "place", "PPLA": "place", "PPLA2": "place", "PPL": "place", "PPLL": "place", "PPLX": "place",
    "ADM1": "municipality",
    "MT": "mountain", "MTS": "mountain", "PK": "mountain", "HLL": "mountain", "HLLS": "mountain",
    "RDGE": "mountain", "MNTS": "mountain",
    "FRST": "forest", "PLAT": "plateau", "VAL": "valley", "VALS": "valley",
    "RGN": "region", "AREA": "area", "PLN": "area", "LK": "lake",
}

# GeoNames labels in English / German / Italian that nobody writes in Slovenian text
_FOREIGN = {"valley", "lake", "region", "carniola", "carinthia", "district", "marsh", "littoral", "park", "hill",
            "mount", "slovene", "white", "dry", "lower", "upper", "julian", "karawanken", "koschuta", "selenitza",
            "venezia", "sattel", "sanntaler", "pass", "alps", "airport", "hotel", "minotel", "sport", "dol"}
_FOREIGN_SINGLE = {"slovenia", "tisovec"}   # "Tisovec" is a one-off oddity (RGN near Kozje)


def fetch_geonames() -> dict[str, Path]:
    """Download (cached) GeoNames SI dump and the SI alternate names."""
    GN_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, rel, name in (("dump", "SI.zip", "SI_dump.zip"), ("alt", "alternatenames/SI.zip", "SI_alt.zip")):
        dest = GN_DIR / name
        if not (GN_DIR / ("dump/SI.txt" if key == "dump" else "alt/SI.txt")).exists():
            if not download(GN_URL + rel, dest):
                raise RuntimeError(f"GeoNames file not found: {GN_URL + rel}")
            with zipfile.ZipFile(dest) as z:
                (GN_DIR / key).mkdir(exist_ok=True)
                z.extract("SI.txt", GN_DIR / key)
        paths[key] = GN_DIR / key / "SI.txt"
    return paths


def _clean_name(n: str) -> str:
    for pre in ("Urban Municipality of ", "Municipality of ", "Občina ", "Mestna občina ", "Opština "):
        if n.startswith(pre):
            return n[len(pre):].strip()
    return n.strip()


def _radius(row) -> float:
    fc, pop, elev = row["fcode"], row["population"], row["elevation"]
    if fc in ("PPLC", "PPLA", "PPLA2"):
        return min(10000.0, 1500 + 12 * math.sqrt(max(pop, 0)))
    if fc in ("PPL", "PPLL"):
        return 1500 + 12 * math.sqrt(max(pop, 0)) if pop > 1500 else 1500.0
    if fc == "PPLX":
        return 1000.0
    if fc == "ADM1":
        return min(12000.0, 4500 + 20 * math.sqrt(max(pop, 0)))
    if fc in ("MT",):
        return 4000.0 if elev >= 1500 else 2500.0
    if fc == "PK":
        return 2500.0
    if fc in ("HLL",):
        return 1500.0
    if fc in ("RDGE", "MNTS"):
        return 3000.0
    if fc in ("MTS", "HLLS"):
        return 6000.0
    if fc == "PLAT":
        return 6000.0
    if fc == "FRST":
        return 6000.0
    if fc in ("VAL", "VALS"):
        return 4000.0
    if fc in ("RGN", "AREA", "PLN"):
        return 8000.0
    if fc == "LK":
        return 1500.0
    return 2000.0


def _read_geonames() -> pd.DataFrame:
    paths = fetch_geonames()
    d = pd.read_csv(paths["dump"], sep="\t", names=_COLS, dtype=str, keep_default_na=False, quoting=3)
    d = d[d.fcode.isin(KIND)].copy()
    d["lat"] = d.lat.astype(float)
    d["lon"] = d.lon.astype(float)
    d["population"] = pd.to_numeric(d["pop"], errors="coerce").fillna(0).astype(int)
    d["elevation"] = pd.to_numeric(d["elev"], errors="coerce").fillna(pd.to_numeric(d["dem"], errors="coerce"))
    d["elevation"] = d["elevation"].fillna(0).astype(float)
    # alternate Slovenian names (language 'sl', not historic), only those differing from the main name
    a = pd.read_csv(paths["alt"], sep="\t", names="aid gid lang name pref short colloq hist from to".split(),
                    dtype=str, keep_default_na=False, quoting=3)
    a = a[(a.lang == "sl") & (a.hist != "1")]
    alt: dict[str, set] = {}
    for gid, nm in zip(a.gid, a.name):
        alt.setdefault(gid, set()).add(_clean_name(nm))
    d["name_clean"] = d["name"].map(_clean_name)
    rows = []
    for r in d.itertuples(index=False):
        nm = r.name_clean
        low = fold(nm).split()
        if any(w in _FOREIGN for w in low) and r.fcode not in ("PPL", "PPLA", "PPLC", "PPLL"):
            continue
        if fold(nm) in _FOREIGN_SINGLE:
            continue
        if r.fcode in ("ADM1",) and nm.lower().startswith("četrtna"):
            continue
        alts = {x for x in alt.get(r.gid, set()) if fold(x) != fold(nm) and len(fold(x)) >= 3}
        # municipalities with double names: "Miren - Kostanjevica", "Dobrova-Polhov Gradec"
        if r.fcode == "ADM1":
            for part in nm.replace("–", "-").split("-"):
                part = part.strip()
                if part and fold(part) != fold(nm) and len(part) >= 4:
                    alts.add(part)
        rows.append(dict(
            gid=f"gn:{r.gid}", name=nm, ascii_name=r.ascii, norm_name=fold(nm),
            alt_names="|".join(sorted({fold(x) for x in alts})),
            fclass=r.fclass, fcode=r.fcode, kind=KIND[r.fcode], lon=r.lon, lat=r.lat,
            population=r.population, elevation=r.elevation, source="geonames", weight=1.0, strict=False,
            basis=f"GeoNames {r.fcode} {r.gid}"))
    g = pd.DataFrame(rows)
    # the SI dump has a few features with coordinates far outside Slovenia (e.g. lon 17.5, lat 45.18); drop them
    x0, y0, x1, y1 = config.BBOX_WGS84
    inside = g.lon.between(x0 - 0.1, x1 + 0.1) & g.lat.between(y0 - 0.1, y1 + 0.1)
    g = g[inside].copy()
    g["radius_m"] = g.apply(_radius, axis=1)
    for c in ("a_m", "b_m", "az_deg"):
        g[c] = np.nan
    return g


def _curated_frame() -> pd.DataFrame:
    rows = []
    for i, c in enumerate(CURATED):
        a, b = c["a_km"] * 1000, c["b_km"] * 1000
        alts = []
        for al in c["aliases"]:
            alts.append(al if isinstance(al, tuple) else (al, 1.0))
        rows.append(dict(
            gid=f"cur:{i:03d}", name=c["name"], ascii_name=fold(c["name"]), norm_name=fold(c["name"]),
            # aliases kept with weights as "norm:weight"
            alt_names="|".join(f"{fold(n)}:{w}" for n, w in alts),
            fclass="curated", fcode="CUR", kind=c["kind"], lon=c["lon"], lat=c["lat"], population=0,
            elevation=0.0, radius_m=math.sqrt(a * b), a_m=a, b_m=b, az_deg=float(c["az"]),
            source="curated", weight=c["weight"], strict=bool(c["strict"]), basis=c["basis"]))
    return pd.DataFrame(rows)


def build_gazetteer(save: bool = True) -> pd.DataFrame:
    g = _read_geonames()
    cur = _curated_frame()

    # municipalities that share a name with a settlement within 15 km are folded into the settlement
    # (the settlement keeps its identity, takes the municipality's larger radius)
    towns = g[g.kind == "place"]
    by_norm = towns.groupby("norm_name")
    drop = []
    for i, m in g[g.kind == "municipality"].iterrows():
        if m.norm_name in by_norm.groups:
            cand = towns.loc[by_norm.groups[m.norm_name]]
            d = np.hypot((cand.lat - m.lat) * 111.2, (cand.lon - m.lon) * 77.0)
            if (d < 15).any():
                j = d.idxmin()
                g.loc[j, "radius_m"] = max(g.loc[j, "radius_m"], 0.85 * m.radius_m)
                if g.loc[j, "population"] == 0:
                    g.loc[j, "population"] = m.population
                g.loc[j, "basis"] += f" + municipality {m.gid}"
                drop.append(i)
    g = g.drop(index=drop)

    # curated entries shadow non-town GeoNames features with the same normalised name
    cur_names = set(cur.norm_name)
    for c in CURATED:
        for al in c["aliases"]:
            nm = al[0] if isinstance(al, tuple) else al
            cur_names.add(fold(nm))
    shadow = g.norm_name.isin(cur_names) & ~g.fcode.isin(["PPLA", "PPLC"])
    # keep villages named like an ambiguous curated entry (Hrušica) as competing candidates
    keep_dup = g.norm_name.isin({"hrusica", "bloke"}) & (g.kind == "place")
    g = g[~(shadow & ~keep_dup)]

    out = pd.concat([cur, g], ignore_index=True)
    out["population"] = out.population.astype(int)
    if save:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(GAZ_PATH, index=False)
        save_regions_geojson()
    return out


def ellipse_polygon(lat, lon, a_m, b_m, az_deg, n=48) -> list[list[float]]:
    """Closed ring [[lon, lat], ...] of an ellipse (local equirectangular approximation)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    az = math.radians(az_deg)
    # ellipse in local ENU: major axis along bearing az (clockwise from north)
    u, v = a_m * np.cos(t), b_m * np.sin(t)
    east = u * math.sin(az) + v * math.cos(az)
    north = u * math.cos(az) - v * math.sin(az)
    dlat = north / 111_200.0
    dlon = east / (111_200.0 * math.cos(math.radians(lat)))
    ring = [[round(lon + x, 5), round(lat + y, 5)] for x, y in zip(dlon, dlat)]
    ring.append(ring[0])
    return ring


def save_regions_geojson(path: Path = REGIONS_PATH) -> None:
    feats = []
    for c in CURATED:
        a, b = c["a_km"] * 1000, c["b_km"] * 1000
        feats.append({"type": "Feature",
                      "properties": {"name": c["name"], "kind": c["kind"], "aliases": [
                          al[0] if isinstance(al, tuple) else al for al in c["aliases"]],
                          "centre_lat": c["lat"], "centre_lon": c["lon"], "a_km": c["a_km"], "b_km": c["b_km"],
                          "azimuth_deg": c["az"], "radius_m": round(math.sqrt(a * b)), "basis": c["basis"],
                          "approximate": True},
                      "geometry": {"type": "Polygon",
                                   "coordinates": [ellipse_polygon(c["lat"], c["lon"], a, b, c["az"])]}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))


@lru_cache(maxsize=1)
def load_gazetteer(rebuild: bool = False) -> pd.DataFrame:
    """Gazetteer DataFrame (from parquet; builds it if missing)."""
    if GAZ_PATH.exists() and not rebuild:
        return pd.read_parquet(GAZ_PATH)
    return build_gazetteer()


def check_against_geonames(g: pd.DataFrame | None = None) -> pd.DataFrame:
    """Offset (km) between curated centres and a same-named raw GeoNames feature; for sanity checks."""
    paths = fetch_geonames()
    d = pd.read_csv(paths["dump"], sep="\t", names=_COLS, dtype=str, keep_default_na=False, quoting=3)
    d["n"] = d.name.map(fold)
    rows = []
    for c in CURATED:
        m = d[(d.n == fold(c["name"])) & d.fcode.isin(["MT", "MTS", "RGN", "AREA", "PLAT", "FRST", "VAL", "RDGE", "HLLS", "PLN"])]
        for r in m.itertuples():
            dist = math.hypot((float(r.lat) - c["lat"]) * 111.2, (float(r.lon) - c["lon"]) * 77.0)
            rows.append((c["name"], r.fcode, float(r.lat), float(r.lon), round(dist, 1)))
    return pd.DataFrame(rows, columns=["curated", "gn_fcode", "gn_lat", "gn_lon", "offset_km"])


if __name__ == "__main__":
    g = build_gazetteer()
    print(g.groupby(["source", "kind"]).size())
    print("wrote", GAZ_PATH, REGIONS_PATH, len(g))
