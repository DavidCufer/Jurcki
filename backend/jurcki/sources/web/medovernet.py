"""med.over.net forum (now medover.zurnal24.si).

Status: BLOCKED.  med.over.net answers every request with a 301 to
medover.zurnal24.si, and that host is refused by the network policy
(proxy CONNECT 403).  We do not use mirrors / archives / alternative hosts, so
this source yields no rows; it only records the situation.  If the host becomes
reachable, add a parser here (cache layout and row schema are in gobe_forum.py).
"""
from __future__ import annotations

from typing import Iterator

from .fetch import BlockedHost, Fetcher

TARGET = "https://med.over.net/forum/"


def collect(f: Fetcher, log=print) -> Iterator[dict]:
    try:
        pg = f.get(TARGET)
    except BlockedHost as e:
        log(f"[medovernet] blocked: {e}")
        return
    log(f"[medovernet] {TARGET} -> HTTP {pg.status}; no parser implemented/verified (host unreachable when this "
        "module was written), yielding no rows")
    return
    yield  # pragma: no cover
