"""CLI:  python -m jurcki.sources.web --source gobe|slotech|medovernet|news|all

Idempotent and resumable: every page is cached under data/raw/web/<host>/, so
re-running only fetches what is missing.  Rows are derived from the cache.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from . import build
from .fetch import BlockedHost, Fetcher

SOURCES = ["gobe", "slotech", "medovernet", "n1", "svet24", "siol"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jurcki.sources.web")
    ap.add_argument("--source", default="all",
                    help="gobe|slotech|medovernet|news|n1|svet24|siol|all (comma separated ok)")
    ap.add_argument("--offline", action="store_true", help="never touch the network; use the cache only")
    ap.add_argument("--min-interval", type=float, default=2.0,
                    help="seconds between requests per host (never below 2.0)")
    ap.add_argument("--since-year", type=int, default=None,
                    help="slotech/news: only fetch threads/articles from this year on, newest first")
    ap.add_argument("--combine-only", action="store_true", help="only merge interim files into processed/")
    ap.add_argument("--report", action="store_true", help="print the summary report and exit")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    if a.report:
        print(build.report("posts"))
        print(build.report("news"))
        return 0

    sel = []
    for s in a.source.split(","):
        s = s.strip()
        sel += SOURCES if s == "all" else (["n1", "svet24", "siol"] if s == "news" else [s])
    if not a.combine_only:
        f = Fetcher(min_interval=max(2.0, a.min_interval), offline=a.offline)
        for s in dict.fromkeys(sel):
            log(f"== source {s}")
            try:
                if s == "gobe":
                    from . import gobe_forum
                    build.write_source("posts", "gobe", gobe_forum.collect(f, log=log))
                elif s == "slotech":
                    from . import slotech
                    build.write_source("posts", "slotech", slotech.collect(f, log=log, since_year=a.since_year))
                elif s == "medovernet":
                    from . import medovernet
                    build.write_source("posts", "medovernet", medovernet.collect(f, log=log))
                elif s in ("n1", "svet24", "siol"):
                    from . import news
                    build.write_source("news", s, news.collect(f, s, log=log, since_year=a.since_year))
                else:
                    log(f"unknown source {s}")
                    continue
            except BlockedHost as e:
                log(f"!! {s}: BLOCKED by network policy: {e}")
    for kind in ("posts", "news"):
        build.combine(kind)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
