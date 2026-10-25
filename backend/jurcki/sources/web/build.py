"""Turn collected rows into parquet files and a summary report."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ...config import INTERIM, PROCESSED

INTERIM_WEB = INTERIM / "web"
SLIM_LIMIT = 50 * 1024 * 1024

POSTS_COLS = ["id", "source", "board", "thread_id", "thread_title", "url", "post_dt",
              "author_hash", "text", "has_quote", "n_images", "has_images", "image_urls",
              "mentions_jurcek", "mentions_mushroom"]
NEWS_COLS = ["id", "source", "section", "url", "published", "title", "lead", "text",
             "mentions_jurcek", "mentions_mushroom", "match_via"]


def write_source(kind: str, name: str, rows) -> Path:
    """Materialise rows of one source into data/interim/web/<kind>_<name>.parquet."""
    cols = POSTS_COLS if kind == "posts" else NEWS_COLS
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df:
            df[c] = None
    df = df[cols].drop_duplicates("id").reset_index(drop=True)
    tcol = "post_dt" if kind == "posts" else "published"
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    INTERIM_WEB.mkdir(parents=True, exist_ok=True)
    p = INTERIM_WEB / f"{kind}_{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def combine(kind: str) -> Path:
    """Merge the per-source files; slim the text out when the file gets huge."""
    parts = sorted(INTERIM_WEB.glob(f"{kind}_*.parquet"))
    parts = [p for p in parts if not p.stem.endswith("_full")]
    name = "web_posts" if kind == "posts" else "web_news"
    cols = POSTS_COLS if kind == "posts" else NEWS_COLS
    tcol = "post_dt" if kind == "posts" else "published"
    frames = [pd.read_parquet(p) for p in parts]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    df = df.sort_values([tcol, "id"]).reset_index(drop=True)
    out = PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    if out.stat().st_size > SLIM_LIMIT:
        df.to_parquet(INTERIM_WEB / f"{name}_full.parquet", index=False, compression="zstd")
        slim = df.drop(columns=["text"]).assign(text_len=df["text"].str.len())
        slim.to_parquet(out, index=False, compression="zstd")
        logging.getLogger("jurcki.web").warning("%s > 50 MB: full text kept in interim, slim version in processed", name)
    return out


def report(kind: str, n_examples: int = 10, seed: int = 1) -> str:
    name = "web_posts" if kind == "posts" else "web_news"
    tcol = "post_dt" if kind == "posts" else "published"
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    full = INTERIM_WEB / f"{name}_full.parquet"
    if full.exists():
        df = pd.read_parquet(full)
    L = [f"=== {name}: {len(df)} rows ==="]
    if df.empty:
        return "\n".join(L)
    d = df.assign(year=df[tcol].dt.year, month=df[tcol].dt.month)
    for src, g in d.groupby("source"):
        L.append(f"\n--- {src}: {len(g)} rows, {int(g.mentions_jurcek.sum())} mention jurcek, "
                 f"{int(g.mentions_mushroom.sum())} mention mushrooms, "
                 f"range {g[tcol].min()} -> {g[tcol].max()}")
        if kind == "posts":
            L.append(f"    distinct authors: {g.author_hash.nunique()}, with images: {int(g.has_images.sum())}")
        L.append("    per year (rows / jurcek):")
        for y, gy in g.groupby("year"):
            L.append(f"      {int(y)}: {len(gy)} / {int(gy.mentions_jurcek.sum())}")
        piv = g.groupby(["year", "month"]).size().unstack(fill_value=0)
        L.append("    per year x month:\n" + piv.to_string())
        ex = g.sample(min(n_examples, len(g)), random_state=seed)
        L.append("    examples:")
        for _, r in ex.iterrows():
            t = (r["text"] or "").replace("\n", " ")[:220]
            L.append(f"      [{r[tcol]:%Y-%m-%d}] {r.get('thread_title') or r.get('title')!s:.60} :: {t}")
    return "\n".join(L)
