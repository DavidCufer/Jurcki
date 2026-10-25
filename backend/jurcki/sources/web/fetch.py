"""Polite, cached, resumable HTTP fetching for the web sources.

* identifies itself with a descriptive User-Agent
* obeys robots.txt for every host (incl. Crawl-delay if larger than our delay)
* at most one request per ``min_interval`` seconds per host, sequential per host
* every response (incl. 404s) is cached gzip-compressed under
  ``data/raw/web/<host>/`` together with the fetch timestamp, so nothing is
  fetched twice and relative dates ("Danes") can be resolved later
* a host that the network policy refuses (proxy CONNECT 403 / DNS failure) is
  reported as blocked; we never try mirrors or alternative hosts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ...config import RAW

log = logging.getLogger("jurcki.web")

USER_AGENT = "jurcki-research/0.1 (private, non-commercial)"
WEB_RAW = RAW / "web"
_HDR = re.compile(r"\A<!--jurcki-fetch (\{.*?\})-->\n", re.S)


class BlockedHost(RuntimeError):
    """The host cannot be reached under the network policy."""


class RobotsDisallowed(RuntimeError):
    pass


class NotCached(RuntimeError):
    """Offline mode and the page is not in the cache."""


@dataclass
class Page:
    url: str
    status: int
    text: str
    fetched_at: datetime  # tz-aware UTC
    from_cache: bool

    @property
    def ok(self) -> bool:
        return self.status == 200


def cache_path(url: str, root: Path = WEB_RAW) -> Path:
    u = urlsplit(url)
    key = u.path.strip("/") or "index"
    if u.query:
        key += "?" + u.query
    safe = re.sub(r"[^A-Za-z0-9._=,;-]+", "_", key)[:120]
    h = hashlib.sha1(url.encode()).hexdigest()[:10]
    return root / u.netloc / f"{safe}.{h}.html.gz"


def _read(path: Path, url: str) -> Page:
    raw = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
    m = _HDR.match(raw)
    meta = json.loads(m.group(1)) if m else {}
    t = datetime.fromisoformat(meta.get("t", "1970-01-01T00:00:00+00:00"))
    return Page(url, int(meta.get("status", 200)), raw[m.end():] if m else raw, t, True)


class Fetcher:
    def __init__(self, min_interval: float = 2.0, offline: bool = False,
                 root: Path = WEB_RAW, timeout: int = 40, user_agent: str = USER_AGENT):
        self.min_interval = min_interval
        self.offline = offline
        self.root = Path(root)
        self.timeout = timeout
        self.ua = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent,
                                     "Accept-Language": "sl,en;q=0.5"})
        self._last: dict[str, float] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._blocked: dict[str, str] = {}
        self._glock = threading.Lock()
        self.n_network = 0

    # -- helpers ---------------------------------------------------------
    def _lock(self, host: str) -> threading.Lock:
        with self._glock:
            return self._locks.setdefault(host, threading.Lock())

    def _wait(self, host: str, delay: float):
        dt = time.monotonic() - self._last.get(host, 0.0)
        if dt < delay:
            time.sleep(delay - dt)

    def blocked(self, host: str) -> str | None:
        return self._blocked.get(host)

    def _raw_get(self, host: str, url: str, delay: float) -> requests.Response:
        """One rate-limited request with retries. Caller holds the host lock."""
        last_exc: Exception | None = None
        for attempt in range(4):
            self._wait(host, delay)
            self._last[host] = time.monotonic()
            self.n_network += 1
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.exceptions.ProxyError as e:
                if "403" in str(e):  # CONNECT refused by the network policy
                    self._blocked[host] = "proxy refused CONNECT (403, network policy)"
                    raise BlockedHost(f"{host}: {self._blocked[host]}") from e
                last_exc = e  # anything else is treated as transient
                log.warning("proxy error for %s: %s", url, str(e)[:200])
                time.sleep(10 * (attempt + 1))
                continue
            except requests.exceptions.SSLError as e:
                raise BlockedHost(f"{host}: TLS error {e}") from e
            except requests.exceptions.RequestException as e:
                last_exc = e
                time.sleep(5 * (attempt + 1))
                continue
            self._last[host] = time.monotonic()
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(10 * (attempt + 1))
                continue
            return r
        raise RuntimeError(f"{url}: giving up ({last_exc})")

    # -- robots ----------------------------------------------------------
    def _rp(self, scheme: str, host: str):
        key = f"{scheme}://{host}"
        if key in self._robots:
            return self._robots[key]
        rp = urllib.robotparser.RobotFileParser()
        url = f"{key}/robots.txt"
        try:
            pg = self.get(url, max_age=timedelta(days=1), check_robots=False)
        except NotCached:
            self._robots[key] = None
            return None
        if pg.status == 200:
            rp.parse(pg.text.splitlines())
        elif pg.status in (401, 403):
            rp.disallow_all = True
        else:
            rp.allow_all = True
        self._robots[key] = rp
        return rp

    def allowed(self, url: str) -> bool:
        u = urlsplit(url)
        rp = self._rp(u.scheme, u.netloc)
        if rp is None:
            return True
        return rp.can_fetch(self.ua, url)

    def crawl_delay(self, url: str) -> float:
        u = urlsplit(url)
        rp = self._robots.get(f"{u.scheme}://{u.netloc}")
        d = rp.crawl_delay(self.ua) if rp else None
        return float(d) if d else 0.0

    # -- main entry ------------------------------------------------------
    def get(self, url: str, max_age: timedelta | None = None,
            check_robots: bool = True, refetch_if=None) -> Page:
        """Return a Page, from cache when possible.

        max_age: refetch cached pages older than this (None = cache forever).
        refetch_if: optional callable(Page) -> bool to force a refetch.
        """
        host = urlsplit(url).netloc
        path = cache_path(url, self.root)
        cached = _read(path, url) if path.exists() else None
        if cached is not None:
            stale = (max_age is not None and
                     datetime.now(timezone.utc) - cached.fetched_at > max_age)
            if not stale and not (refetch_if and refetch_if(cached)):
                return cached
        if self.offline:
            if cached is not None:
                return cached
            raise NotCached(url)
        if host in self._blocked:
            raise BlockedHost(f"{host}: {self._blocked[host]}")
        if check_robots and not self.allowed(url):
            raise RobotsDisallowed(url)
        with self._lock(host):
            delay = max(self.min_interval, self.crawl_delay(url))
            r = self._raw_get(host, url, delay)
        text = r.content.decode(_guess_enc(r), "replace")
        now = datetime.now(timezone.utc)
        meta = json.dumps({"t": now.isoformat(), "status": r.status_code,
                           "url": r.url}, ensure_ascii=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(gzip.compress(f"<!--jurcki-fetch {meta}-->\n{text}".encode("utf-8"), 6))
        tmp.rename(path)
        return Page(url, r.status_code, text, now, False)


def _guess_enc(r: requests.Response) -> str:
    ct = r.headers.get("content-type", "")
    m = re.search(r"charset=([\w-]+)", ct, re.I)
    if m:
        return m.group(1)
    m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", r.content[:4096], re.I)
    if m:
        return m.group(1).decode()
    return "utf-8"
