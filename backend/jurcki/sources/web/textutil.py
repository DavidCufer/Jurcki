"""Slovenian date parsing, mushroom relevance regexes, author hashing, html->text."""
from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .fetch import WEB_RAW

TZ = ZoneInfo("Europe/Ljubljana")

# ---------------------------------------------------------------- dates
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6,
           "jul": 7, "avg": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12}


def _month(word: str) -> int | None:
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return _MONTHS.get(w[:3])


_TIME = r"(?P<h>\d{1,2})[:.](?P<mi>\d{2})(?:[:.](?P<s>\d{2}))?"
_RE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?")
_RE_REL = re.compile(r"\b(danes|včeraj|vceraj|predvčerajšnjim)\s*(?:ob)?\s*" + _TIME, re.I)
_RE_REL_ONLY = re.compile(r"\b(danes|včeraj|vceraj)\b", re.I)
_RE_TXT = re.compile(r"(?P<d>\d{1,2})\.?\s+(?P<mon>[^\W\d_]{3,12})\.?,?\s+(?P<y>\d{4})(?:[,\s]+(?:ob\s+)?" + _TIME + ")?", re.I)
_RE_NUM = re.compile(r"(?P<d>\d{1,2})\.\s*(?P<m>\d{1,2})\.\s*(?P<y>\d{4})(?:[,\s]+(?:ob\s+)?" + _TIME + ")?")


def parse_sl_datetime(s: str, ref: datetime | None = None) -> datetime | None:
    """Parse the date formats seen on Slovenian sites into an aware datetime
    (Europe/Ljubljana).  ``ref`` (the time the page was fetched) resolves
    "Danes"/"Včeraj".  Returns None if nothing parses.

    Handles: ISO 8601, "16 September 2026, 10:59", "27. avg 2016 ob 22:50",
    "danes ob 18:04", "Včeraj ob 09:45", "29. 5. 2026, 06.00", "24.10.2026".
    """
    if not s:
        return None
    s = re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()
    s = re.sub(r"(?i)\b(danes|včeraj|vceraj)(ob)\b", r"\1 \2", s)
    m = _RE_ISO.search(s)
    if m:
        y, mo, d, h, mi, sec, tz = m.groups()
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(sec or 0))
        if tz:
            from datetime import timezone
            if tz == "Z":
                off = timezone.utc
            else:
                sign = 1 if tz[0] == "+" else -1
                t = tz[1:].replace(":", "")
                off = timezone(sign * timedelta(hours=int(t[:2]), minutes=int(t[2:])))
            return dt.replace(tzinfo=off).astimezone(TZ)
        return dt.replace(tzinfo=TZ)
    m = _RE_REL.search(s)
    if m and ref is not None:
        base = ref.astimezone(TZ).date()
        w = m.group(1).lower()
        if w.startswith("v") and "raj" in w and not w.startswith("pred"):
            base -= timedelta(days=1)
        elif w.startswith("pred"):
            base -= timedelta(days=2)
        return datetime(base.year, base.month, base.day, int(m.group("h")),
                        int(m.group("mi")), int(m.group("s") or 0), tzinfo=TZ)
    m = _RE_REL_ONLY.search(s)
    if m and ref is not None:
        base = ref.astimezone(TZ).date()
        if m.group(1).lower().startswith("v"):
            base -= timedelta(days=1)
        return datetime(base.year, base.month, base.day, tzinfo=TZ)
    for m in _RE_TXT.finditer(s):
        mon = _month(m.group("mon"))
        if mon:
            return datetime(int(m.group("y")), mon, int(m.group("d")),
                            int(m.group("h") or 0), int(m.group("mi") or 0),
                            int(m.group("s") or 0), tzinfo=TZ)
    m = _RE_NUM.search(s)
    if m:
        return datetime(int(m.group("y")), int(m.group("m")), int(m.group("d")),
                        int(m.group("h") or 0), int(m.group("mi") or 0),
                        int(m.group("s") or 0), tzinfo=TZ)
    return None


# ---------------------------------------------------------------- relevance
def fold(s: str) -> str:
    """Lowercase and strip diacritics (č->c, š->s, ž->z) so that 'jurcki'
    and 'jurčki' are the same."""
    s = unicodedata.normalize("NFKD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if not unicodedata.combining(c))


# Boletus: jurček/jurčki/jurčka/jurčke/jurčkov/jurčkih..., gobani, goban, edulis...
RE_JURCEK = re.compile(
    r"\bjurc(?:ek|k\w*)|\bgoban\w*|\bboletus\b|\bedulis\b|\bvrganj\w*|\bpinicola\b|\baereus\b|\bpinophilus\b|\breticulatus\b")
# any mushroom talk (kept as effort data)
RE_MUSHROOM = re.compile(
    r"\bgob\b|\bgob(?:a|e|i|o|u|am|ah|ami|ic\w*|ar\w*|j\w*|ov\w*|nat\w*|ni)\b|\bgobar\w*|\bgobic\w*"
    r"|\blisick\w*|\bmarel\w*|\bsirovk\w*|\bbukovk\w*|\bstorovk\w*|\bkukmak\w*|\bmusnic\w*"
    r"|\bpuhavk\w*|\bkrvavic\w*|\brecnjak\w*|\bsmrcek\w*|\bsmrck\w*|\bbledivk\w*|\bcesnovk\w*"
    r"|\bgolobic\w*|\bsivk\w*|\bsampinjon\w*|\bmycel\w*|\bgobarsk\w*|\bboletus\b|\bfungi\b"
    r"|\bsuhe gobe\b|\bmushroom\w*|\bjurc(?:ek|k\w*)|\bgoban\w*")
# "weak" title test for thread discovery (word-initial 'gob' but not gobec/gobcati)
RE_TITLE = re.compile(r"\bgob(?!ec\b|c|ez\b|elin)\w*|\bjurc\w*|\bmusnic\w*|\blisick\w*|\bmarel\w*"
                      r"|\bbukovk\w*|\bsirovk\w*|\bstorovk\w*|\bvrganj\w*|\bkukmak\w*|\bnabiran\w*")


def mentions_jurcek(text: str) -> bool:
    return bool(RE_JURCEK.search(fold(text)))


def mentions_mushroom(text: str) -> bool:
    return bool(RE_MUSHROOM.search(fold(text)))


def title_looks_mushroomy(title: str) -> bool:
    return bool(RE_TITLE.search(fold(title)))


# ---------------------------------------------------------------- privacy
_SALT: bytes | None = None


def _salt() -> bytes:
    global _SALT
    if _SALT is None:
        p = WEB_RAW / "_author_salt"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(secrets.token_hex(16))
        _SALT = p.read_text().strip().encode()
    return _SALT


def author_hash(source: str, name: str | None) -> str | None:
    """Salted hash of the display name (per source). Usernames are never stored."""
    if not name or not name.strip():
        return None
    h = hashlib.sha256(_salt() + b"|" + source.encode() + b"|" + name.strip().lower().encode())
    return h.hexdigest()[:16]


def stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- html -> text
def unescape_all(s: str, rounds: int = 4) -> str:
    """html.unescape until stable, so double-escaped entities (&amp;scaron;) are fixed too."""
    import html
    for _ in range(rounds):
        t = html.unescape(s)
        if t == s:
            break
        s = t
    return s


def clean_text(s: str) -> str:
    s = unescape_all(s)
    s = s.replace("\xa0", " ").replace("​", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def node_text(node) -> str:
    """Plain text of a bs4 node with <br>/<p> turned into newlines."""
    for br in node.find_all("br"):
        br.replace_with("\n")
    for p in node.find_all(["p", "div", "li", "h1", "h2", "h3", "h4"]):
        p.insert_before("\n")
        p.insert_after("\n")
    return clean_text(node.get_text())


def make_soup(html: str):
    """BeautifulSoup with the stdlib parser (no lxml dependency in the venv)."""
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser")
