"""GBIF Boletus occurrences from the AWS open-data parquet snapshot.

The snapshot (~10k files, ~300 GB) is scanned directly from S3 with pyarrow:
per file we first read only genus + coordinates, and fetch the remaining
columns just for files/rows that match (genus Boletus inside the bbox).
DuckDB httpfs is not used because its extension host is blocked here.

Run:  python -m jurcki.sources.gbif [--snapshot YYYY-MM-DD] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from jurcki import config

BUCKET = "gbif-open-data-eu-central-1"
REGION = "eu-central-1"
GENUS = "Boletus"
JURCKI = (
    "Boletus edulis",
    "Boletus aereus",
    "Boletus reticulatus",  # accepted name of B. aestivalis
    "Boletus pinophilus",
)
MAX_UNCERTAINTY_M = 5000

# columns used to select rows
_KEY_COLS = ["genus", "decimallatitude", "decimallongitude"]
# columns fetched for matching rows
_KEEP_COLS = [
    "gbifid", "datasetkey", "species", "scientificname", "verbatimscientificname",
    "taxonkey", "specieskey", "taxonrank", "coordinateuncertaintyinmeters",
    "eventdate", "year", "month", "day", "countrycode", "stateprovince",
    "elevation", "basisofrecord", "institutioncode", "occurrencestatus",
    "recordedby", "issue",
]

# Well-known datasets (titles from the GBIF registry); the bucket has no metadata.
DATASET_TITLES = {
    "50c9509d-22c7-4a22-a47d-8c48425ef4a7": "iNaturalist Research-grade Observations",
    "8a863029-f435-446a-821e-275f4f641165": "Observation.org, Nature data from around the World",
}

RAW_DIR = config.RAW / "gbif"
OUT_PARQUET = config.PROCESSED / "gbif_boletus.parquet"
OUT_CSV = config.PROCESSED / "gbif_boletus.csv"
OUT_SUMMARY = config.PROCESSED / "gbif_boletus_summary.txt"


# ---------------------------------------------------------------- S3 access

def _fs() -> pafs.S3FileSystem:
    """Anonymous S3 filesystem; honours HTTPS_PROXY (CA via AWS_CA_BUNDLE)."""
    import os

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    kw = {"proxy_options": proxy} if proxy else {}
    return pafs.S3FileSystem(anonymous=True, region=REGION, **kw)


def latest_snapshot(fs: pafs.S3FileSystem | None = None) -> str:
    """Return the newest snapshot date (YYYY-MM-DD) in the bucket."""
    fs = fs or _fs()
    infos = fs.get_file_info(pafs.FileSelector(f"{BUCKET}/occurrence/"))
    dates = sorted(i.base_name for i in infos
                   if i.type == pafs.FileType.Directory and i.base_name[:2] == "20")
    if not dates:
        raise RuntimeError("no GBIF snapshots found")
    return dates[-1]


def list_files(fs: pafs.S3FileSystem, snapshot: str) -> list[str]:
    """All parquet part files of a snapshot."""
    sel = pafs.FileSelector(f"{BUCKET}/occurrence/{snapshot}/occurrence.parquet/")
    return sorted(i.path for i in fs.get_file_info(sel) if i.type == pafs.FileType.File)


# ---------------------------------------------------------------- extraction

def _bbox_mask(t: pa.Table, bbox) -> pa.ChunkedArray:
    lon, lat = t["decimallongitude"], t["decimallatitude"]
    m = pc.and_(pc.greater_equal(lon, bbox[0]), pc.less_equal(lon, bbox[2]))
    m = pc.and_(m, pc.and_(pc.greater_equal(lat, bbox[1]), pc.less_equal(lat, bbox[3])))
    return pc.and_(m, pc.equal(t["genus"], GENUS)).fill_null(False)


def _strings(col: pa.ChunkedArray) -> list[list[str]]:
    """Python lists of strings from a GBIF list<struct<array_element>> column."""
    return [[e["array_element"] if isinstance(e, dict) else e for e in (x or [])]
            for x in col.to_pylist()]


def hash_recorders(col: pa.ChunkedArray) -> list[str | None]:
    """Hash each recordedby list into a short anonymous id (names are dropped)."""
    out = []
    for names in _strings(col):
        if not names:
            out.append(None)
            continue
        key = "|".join(sorted(n.strip().lower() for n in names if n))
        out.append(hashlib.sha1(key.encode()).hexdigest()[:12] if key else None)
    return out


def _scan_file(fs, path: str, bbox) -> pd.DataFrame | None:
    """Matching rows of one parquet part file (None if no match)."""
    f = pq.ParquetFile(path, filesystem=fs, pre_buffer=True)
    key = f.read(columns=_KEY_COLS)
    mask = _bbox_mask(key, bbox)
    if not pc.any(mask).as_py():
        return None
    key = key.filter(mask)
    t = f.read(columns=_KEEP_COLS).filter(mask)
    df = t.drop_columns(["recordedby", "issue"]).to_pandas()
    df["recorder_id"] = hash_recorders(t["recordedby"])
    df["issue"] = [";".join(x) if x else None for x in _strings(t["issue"])]
    df["decimallatitude"] = key["decimallatitude"].to_numpy()
    df["decimallongitude"] = key["decimallongitude"].to_numpy()
    return df


def _scan_retry(fs, path, bbox, tries=4):
    for k in range(tries):
        try:
            return _scan_file(fs, path, bbox)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def extract_raw(snapshot: str, workers: int = 48, max_files: int | None = None,
                bbox=config.BBOX_WGS84) -> pd.DataFrame:
    """Scan the snapshot and return all Boletus rows inside bbox."""
    fs = _fs()
    files = list_files(fs, snapshot)[:max_files]
    print(f"snapshot {snapshot}: scanning {len(files)} files with {workers} threads")
    parts, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_scan_retry, fs, p, bbox) for p in files]
        for i, fut in enumerate(as_completed(futs), 1):
            df = fut.result()  # raises if a file failed after retries
            if df is not None:
                parts.append(df)
            if i % 500 == 0 or i == len(files):
                n = sum(len(p) for p in parts)
                print(f"  {i}/{len(files)} files, {n} rows, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_KEEP_COLS)


# ---------------------------------------------------------------- cleaning

def clean(df: pd.DataFrame, max_uncertainty: float = MAX_UNCERTAINTY_M,
          round_decimals: int = 3) -> pd.DataFrame:
    """Drop absent/undated/unlocated/imprecise records and duplicates.

    Duplicate = same species (or scientific name if no species), date and
    coordinates rounded to `round_decimals`; the most precise record is kept.
    Adds `date` and `is_jurcki` columns.
    """
    d = df.copy()
    status = d["occurrencestatus"].astype("string").str.upper()
    d = d[status.ne("ABSENT").fillna(True)]
    d = d.dropna(subset=["decimallatitude", "decimallongitude", "year", "month", "day"])
    d = d[~((d["decimallatitude"] == 0) & (d["decimallongitude"] == 0))]
    unc = d["coordinateuncertaintyinmeters"]
    d = d[~(unc > max_uncertainty)]  # unknown uncertainty is kept

    d["date"] = pd.to_datetime(
        d[["year", "month", "day"]].astype("int64"), errors="coerce")
    d = d.dropna(subset=["date"])

    name = d["species"].fillna(d["scientificname"])
    d["_key"] = (name.astype(str) + "|" + d["date"].dt.strftime("%Y-%m-%d") + "|"
                 + d["decimallatitude"].round(round_decimals).astype(str) + "|"
                 + d["decimallongitude"].round(round_decimals).astype(str))
    d = d.sort_values(["coordinateuncertaintyinmeters", "gbifid"], na_position="last")
    d = d.drop_duplicates("_key").drop(columns="_key")
    d["is_jurcki"] = d["species"].isin(JURCKI)
    return d.sort_values(["date", "gbifid"]).reset_index(drop=True)


# ---------------------------------------------------------------- summary

def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Count tables: species, year, month, country, dataset (+ SI per year)."""
    sp = df["species"].fillna("(genus only)")
    out = {
        "species": sp.value_counts().rename("n").to_frame(),
        "year": pd.crosstab(df["year"], df["is_jurcki"].map({True: "jurcki", False: "other"})),
        "month": df["month"].value_counts().sort_index().rename("n").to_frame(),
        "country": df["countrycode"].fillna("??").value_counts().rename("n").to_frame(),
    }
    si = df[df["countrycode"] == "SI"]
    out["year_SI"] = pd.crosstab(si["year"], si["is_jurcki"].map({True: "jurcki", False: "other"}))
    ds = df.groupby("datasetkey").agg(n=("gbifid", "size"), n_jurcki=("is_jurcki", "sum"),
                                       institution=("institutioncode", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""))
    ds["title"] = [DATASET_TITLES.get(k, "") for k in ds.index]
    out["dataset"] = ds.sort_values("n", ascending=False)
    return out


def format_summary(tables: dict[str, pd.DataFrame], total: int) -> str:
    parts = [f"rows after cleaning: {total}"]
    for name, t in tables.items():
        if name == "dataset":
            t = t.head(25)
        parts.append(f"\n== {name} ==\n{t.to_string()}")
    return "\n".join(parts)


# ---------------------------------------------------------------- CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", help="snapshot date; default latest")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-files", type=int, help="scan only the first N files (testing)")
    ap.add_argument("--force", action="store_true", help="re-scan even if cached")
    args = ap.parse_args(argv)

    snapshot = args.snapshot or latest_snapshot()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{args.max_files}" if args.max_files else ""
    raw_path = RAW_DIR / f"boletus_{snapshot}{suffix}.parquet"

    t0 = time.time()
    if raw_path.exists() and not args.force:
        print(f"using cached {raw_path}")
        raw = pd.read_parquet(raw_path)
    else:
        raw = extract_raw(snapshot, args.workers, args.max_files)
        raw.to_parquet(raw_path, index=False)
        print(f"wrote {raw_path} ({len(raw)} rows) in {time.time() - t0:.0f}s")

    df = clean(raw)
    print(f"raw rows {len(raw)} -> cleaned {len(df)}")
    if args.max_files:
        return
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    text = f"GBIF snapshot {snapshot}; raw rows {len(raw)}\n" + format_summary(summarize(df), len(df))
    OUT_SUMMARY.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
