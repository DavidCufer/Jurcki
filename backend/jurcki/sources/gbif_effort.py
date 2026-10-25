"""Target-group sample for observer effort: all Fungi records near Slovenia.

Presence-only jurcki finds mix "where/when jurcki fruit" with "where/when
people look for mushrooms and report them". All other fungi records show the
second part: a day with many fungi records but no jurcki is a weak absence,
a jurcki share above the effort baseline is a real signal.

Same anonymous pyarrow S3 scan as ``sources/gbif`` (same bucket, no other
host): per part file only kingdom + coordinates + year are read first, and
the remaining columns are fetched for matching rows (kingdom == Fungi,
inside the bbox, year >= 2010).

Run:  python -m jurcki.sources.gbif_effort [--snapshot YYYY-MM-DD] [--force]
Output: data/processed/gbif_fungi.parquet (+ gbif_fungi_summary.txt);
the raw extract is cached in data/raw/gbif/ (git-ignored).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from jurcki import config
from jurcki.sources import gbif

KINGDOM = "Fungi"
MIN_YEAR = 2010

_KEY_COLS = ["kingdom", "decimallatitude", "decimallongitude", "year"]
_TAXON = ["phylum", "class", "order", "family", "genus"]
_KEEP_COLS = [
    "gbifid", "datasetkey", "species", "scientificname", "occurrencestatus",
    "coordinateuncertaintyinmeters", "year", "month", "day", "countrycode",
    "basisofrecord", "recordedby",
] + _TAXON

# columns of the committed table
OUT_COLS = ["gbifid", "species", "genus", "family", "order", "class", "lat", "lon",
            "coordinateuncertaintyinmeters", "date", "datasetkey", "basisofrecord",
            "countrycode", "recorder_id", "is_jurcki"]
OUT_PARQUET = config.PROCESSED / "gbif_fungi.parquet"
OUT_SUMMARY = config.PROCESSED / "gbif_fungi_summary.txt"
RAW_DIR = gbif.RAW_DIR
MAX_COMMIT_MB = 20


def _mask(t: pa.Table, bbox, min_year: int = MIN_YEAR) -> pa.ChunkedArray:
    lon, lat = t["decimallongitude"], t["decimallatitude"]
    m = pc.and_(pc.greater_equal(lon, bbox[0]), pc.less_equal(lon, bbox[2]))
    m = pc.and_(m, pc.and_(pc.greater_equal(lat, bbox[1]), pc.less_equal(lat, bbox[3])))
    m = pc.and_(m, pc.greater_equal(t["year"], min_year))
    return pc.and_(m, pc.equal(t["kingdom"], KINGDOM)).fill_null(False)


def _scan_file(fs, path: str, bbox, min_year: int = MIN_YEAR) -> pd.DataFrame | None:
    """Matching rows of one parquet part file (None if no match)."""
    f = pq.ParquetFile(path, filesystem=fs, pre_buffer=True)
    key = f.read(columns=_KEY_COLS)
    mask = _mask(key, bbox, min_year)
    if not pc.any(mask).as_py():
        return None
    key = key.filter(mask)
    t = f.read(columns=_KEEP_COLS).filter(mask)
    df = t.drop_columns(["recordedby"]).to_pandas()
    df["recorder_id"] = gbif.hash_recorders(t["recordedby"])
    df["decimallatitude"] = key["decimallatitude"].to_numpy()
    df["decimallongitude"] = key["decimallongitude"].to_numpy()
    return df


def _scan_retry(fs, path, bbox, min_year, tries=4):
    for k in range(tries):
        try:
            return _scan_file(fs, path, bbox, min_year)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def extract_raw(snapshot: str, workers: int = 48, max_files: int | None = None,
                bbox=config.BBOX_WGS84, min_year: int = MIN_YEAR) -> pd.DataFrame:
    """Scan the snapshot and return all Fungi rows inside bbox from ``min_year``."""
    fs = gbif._fs()
    files = gbif.list_files(fs, snapshot)[:max_files]
    print(f"snapshot {snapshot}: scanning {len(files)} files with {workers} threads")
    parts, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_scan_retry, fs, p, bbox, min_year) for p in files]
        for i, fut in enumerate(as_completed(futs), 1):
            df = fut.result()
            if df is not None:
                parts.append(df)
            if i % 500 == 0 or i == len(files):
                n = sum(len(p) for p in parts)
                print(f"  {i}/{len(files)} files, {n} rows, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_KEEP_COLS)


def clean(raw: pd.DataFrame, max_uncertainty: float = gbif.MAX_UNCERTAINTY_M) -> pd.DataFrame:
    """``gbif.clean`` (full dates, <= 5 km, dedupe) and the compact column set."""
    d = gbif.clean(raw, max_uncertainty)
    d = d.rename(columns={"decimallatitude": "lat", "decimallongitude": "lon"})
    d["gbifid"] = d["gbifid"].astype("int64")
    for c in ("species", "genus", "family", "order", "class", "datasetkey", "basisofrecord",
              "countrycode", "recorder_id"):
        d[c] = d[c].astype("category")
    for c in ("lat", "lon"):
        d[c] = d[c].astype("float32")
    d["coordinateuncertaintyinmeters"] = d["coordinateuncertaintyinmeters"].astype("float32")
    return d[OUT_COLS].reset_index(drop=True)


def load(path=OUT_PARQUET) -> pd.DataFrame:
    """The cleaned fungi table (``date`` as datetime64)."""
    return pd.read_parquet(path)


def summarize(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Counts per year / country / dataset / month and the top species."""
    jur = d["is_jurcki"].map({True: "jurcki", False: "other"})
    si = d[d["countrycode"] == "SI"]
    return {
        "year": pd.crosstab(d["date"].dt.year, jur),
        "year_SI": pd.crosstab(si["date"].dt.year, si["is_jurcki"].map({True: "jurcki", False: "other"})),
        "country": d["countrycode"].astype(str).fillna("??").value_counts().rename("n").to_frame(),
        "month": pd.crosstab(d["date"].dt.month, jur),
        "dataset": d.groupby("datasetkey", observed=True).agg(
            n=("gbifid", "size"), n_jurcki=("is_jurcki", "sum"),
            n_recorders=("recorder_id", "nunique")).sort_values("n", ascending=False)
        .assign(title=lambda x: [gbif.DATASET_TITLES.get(k, "") for k in x.index]),
        "top_species": d["species"].astype(str).replace("nan", "(genus only)").value_counts().head(30).rename("n").to_frame(),
        "basisofrecord": d["basisofrecord"].astype(str).value_counts().rename("n").to_frame(),
    }


def format_summary(tables: dict, total: int) -> str:
    parts = [f"rows after cleaning: {total}"]
    for name, t in tables.items():
        if name == "dataset":
            t = t.head(25)
        parts.append(f"\n== {name} ==\n{t.to_string()}")
    return "\n".join(parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", help="snapshot date; default latest")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-files", type=int, help="scan only the first N files (testing)")
    ap.add_argument("--force", action="store_true", help="re-scan even if cached")
    args = ap.parse_args(argv)

    snapshot = args.snapshot or gbif.latest_snapshot()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{args.max_files}" if args.max_files else ""
    raw_path = RAW_DIR / f"fungi_{snapshot}{suffix}.parquet"
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
    df.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    mb = OUT_PARQUET.stat().st_size / 1e6
    print(f"wrote {OUT_PARQUET} ({mb:.1f} MB)")
    if mb > MAX_COMMIT_MB:
        print(f"warning: larger than {MAX_COMMIT_MB} MB, consider reducing columns")
    text = f"GBIF snapshot {snapshot}; raw rows {len(raw)}\n" + format_summary(summarize(df), len(df))
    OUT_SUMMARY.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
