"""News articles about mushrooms from n1info.si, svet24.si (incl. its Dolenjski
list / regional pages) and siol.net.

Discovery (robots.txt checked for each host; none of these paths is disallowed):
* n1info.si  : sitemap_post_*.xml (all articles since 2020); keep URLs whose slug
               looks mushroom-related, then fetch those articles.
* svet24.si  : /iskanje?q=<query>&date_from=<y>-01-01&date_to=<y>-12-31&stran=<n>
               for each year and several queries; candidates are filtered on the
               result snippet (title + lead) before fetching the article.
* siol.net   : keyword pages /kljucne-besede/<gobe|gobarjenje|jurcek|nabiranje>-<id>/articles?page=<n>.
Not covered: rtvslo.si (its robots.txt explicitly bans Anthropic crawlers and has no
sitemap), dolenjskilist.si (redirects to dolenjskilist.svet24.si, blocked by the
network policy), 24ur.com (blocked).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Iterator
from urllib.parse import quote, urljoin, urlsplit

from .fetch import BlockedHost, Fetcher
from .textutil import (clean_text, unescape_all, fold, make_soup, mentions_jurcek, mentions_mushroom,
                       node_text, parse_sl_datetime, stable_id)

N1_SITEMAP = "https://n1info.si/sitemap.xml"
SVET24_QUERIES = ["gobe", "jurčki", "jurček", "gobani", "gobarji", "gobarska sezona",
                  "nabiranje gob", "suša gobe", "gobe rastejo"]
SIOL_KEYWORDS = ["gobe-19720", "gobarjenje-77549", "jurcek-77548", "nabiranje-62279"]
RE_SLUG = re.compile(r"(?<![a-z])(gob(?!ec|c|elin|ez)|jurc|lisick|marel|bukovk|musnic|sirovk|vrganj|kukmak)")

ARTICLE_TYPES = {"NewsArticle", "Article", "ReportageNewsArticle", "BlogPosting", "OpinionNewsArticle"}


# ------------------------------------------------------------------ parsing
def _ld_articles(soup) -> list[dict]:
    out = []
    for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(sc.string or sc.get_text() or "null")
        except ValueError:
            continue
        stack = [data]
        while stack:
            d = stack.pop()
            if isinstance(d, list):
                stack.extend(d)
            elif isinstance(d, dict):
                t = d.get("@type")
                ts = set(t if isinstance(t, list) else [t])
                if ts & ARTICLE_TYPES:
                    out.append(d)
                if "@graph" in d:
                    stack.append(d["@graph"])
    return out


_BAD_PARENT = {"nav", "footer", "aside", "header", "form", "noscript"}
_BAD_CLASS = re.compile(r"(card|related|share|social|newsletter|comment|sidebar|banner|promo|advert|teaser|widget|menu|tags?$)", re.I)


def _bad(el) -> bool:
    for p in el.parents:
        if p.name in _BAD_PARENT:
            return True
        cls = " ".join(p.get("class") or [])
        if cls and _BAD_CLASS.search(cls) and p.name not in ("body", "html", "main"):
            if not re.search(r"article[-_]?(content|body|text)", cls, re.I):
                return True
    return False


def find_body_container(soup):
    """The element holding the article body: the one with the most paragraph text."""
    art = soup.select_one("article.article-content")  # svet24
    if art is not None:
        return art
    best, cont = 0, None
    groups: dict[int, list] = {}
    parents = {}
    for p in soup.find_all("p"):
        if _bad(p):
            continue
        t = p.get_text(" ", strip=True)
        if len(t) < 25:
            continue
        groups.setdefault(id(p.parent), []).append(len(t))
        parents[id(p.parent)] = p.parent
    for k, lens in groups.items():
        if sum(lens) > best:
            best, cont = sum(lens), parents[k]
    return cont


def extract_body(soup) -> str:
    """Plain text of the article body."""
    cont = find_body_container(soup)
    if cont is None:
        return ""
    paras = []
    for el in cont.find_all(["p", "h2", "h3", "li", "blockquote"]):
        if _bad(el) and cont.name != "article":
            continue
        if el.name == "li" and el.find("p"):
            continue
        if el.find_parent("blockquote") and el.name != "blockquote":
            continue
        t = clean_text(el.get_text(" ", strip=True))
        if t and (not paras or paras[-1] != t):
            paras.append(t)
    drop = re.compile(r"^(Preberite še|Preberi še|Več na|Foto:|Fotografija:|Deli članek|Oglas|Naročite|Ostanite z nami|Bodite vedno na tekočem|"
                      r"Za hitre novice|Pošljite|Sledite|Piškotki|Email|Strinjam se|Hvala za prijavo|Na vaš e-naslov|Nastavi$|Z oddajo e-po|Želim lokalne|Zapri$|Odjava|Prijavite se na e-novice)", re.I)
    paras = [p for p in paras if not drop.match(p)]
    return "\n\n".join(paras)


def parse_article(html: str, url: str) -> dict | None:
    soup = make_soup(html)
    ld = _ld_articles(soup)
    title = lead = pub = None
    if ld:
        d = ld[0]
        title = d.get("headline")
        lead = d.get("description")
        pub = d.get("datePublished")
    meta = lambda *keys: next((m.get("content") for k in keys for m in
                               soup.find_all("meta", attrs={"property": k}) + soup.find_all("meta", attrs={"name": k})
                               if m.get("content")), None)
    title = title or meta("og:title", "twitter:title")
    lead = lead or meta("og:description", "description")
    pub = pub or meta("article:published_time", "datePublished")
    if not pub:
        t = soup.find("time", attrs={"datetime": True})
        pub = t["datetime"] if t else None
    if not title:
        h = soup.find("h1")
        title = h.get_text(" ", strip=True) if h else None
    dt = parse_sl_datetime(pub) if pub else None
    if not title:
        return None
    title = unescape_all(title or "")
    lead = unescape_all(lead or "")
    title = re.sub(r"\s*[-|–]\s*(Svet24\.si|siol\.net|N1)\s*$", "", clean_text(title), flags=re.I)
    body = extract_body(soup)
    lead = clean_text(lead or "")
    body = "\n\n".join(p for p in body.split("\n\n") if p != title and p != lead)
    section = urlsplit(url).path.strip("/").split("/")
    section = "/".join(section[:2]) if len(section) > 2 else (section[0] if section else "")
    return {"title": title, "published": dt, "lead": lead, "text": body, "section": section}


def parse_sitemap(xml: str) -> list[tuple[str, str | None]]:
    return [(m.group(1).strip(), (m.group(2) or "").strip() or None)
            for m in re.finditer(r"<loc>([^<]+)</loc>(?:\s*<lastmod>([^<]*)</lastmod>)?", xml)]


def parse_svet24_search(html: str) -> list[dict]:
    s = make_soup(html)
    out = []
    srch = s.find("search")
    arts = srch.find_all_next("article") if srch else s.find_all("article")
    for a in arts:
        l = a.find("a", href=True)
        if not l or not re.search(r"-\d{4,8}$", l["href"].split("?")[0]):
            continue
        txt = a.get_text(" ", strip=True)
        m = re.findall(r"\d{1,2}\. \d{1,2}\. \d{4}(?:, \d{2}\.\d{2})?", txt)
        h = a.find(["h2", "h3"])
        out.append({"url": urljoin("https://svet24.si", l["href"]),
                    "title": h.get_text(" ", strip=True) if h else "",
                    "snippet": txt, "date": parse_sl_datetime(m[-1]) if m else None})
    return out


def svet24_has_next(html: str, page: int) -> bool:
    return f"stran={page + 1}" in html


def parse_siol_keyword(html: str) -> list[str]:
    s = make_soup(html)
    cont = s.select_one(".keyword_page__results_form") or s
    urls = []
    for a in cont.select("article.card a[href]"):
        h = a["href"].split("?")[0]
        if re.search(r"-\d{4,7}$", h) and not h.startswith("/C/"):
            urls.append(urljoin("https://siol.net", h))
    return list(dict.fromkeys(urls))


# ------------------------------------------------------------------ discovery
def n1_candidates(f: Fetcher, log) -> list[tuple[str, str | None]]:
    idx = f.get(N1_SITEMAP, max_age=timedelta(days=2))
    maps = [u for u, _ in parse_sitemap(idx.text) if "sitemap_post_" in u]
    log(f"[n1] {len(maps)} post sitemaps")
    out = {}
    for i, m in enumerate(maps):
        pg = f.get(m, max_age=timedelta(days=30) if i < 3 else None)
        for u, lm in parse_sitemap(pg.text):
            if RE_SLUG.search(fold(urlsplit(u).path)):
                out[u] = lm
        if i % 20 == 0:
            log(f"[n1] sitemap {i + 1}/{len(maps)}: {len(out)} mushroom-slug URLs so far")
    return list(out.items())


def svet24_candidates(f: Fetcher, log, since_year=None) -> list[dict]:
    this = datetime.now().year
    first = max(2008, since_year or 2008)
    found: dict[str, dict] = {}
    for y in range(this, first - 1, -1):
        for q in SVET24_QUERIES:
            page = 1
            while page <= 120:
                url = (f"https://svet24.si/iskanje?q={quote(q)}&date_from={y}-01-01&date_to={y}-12-31"
                       + (f"&stran={page}" if page > 1 else ""))
                # current year is still changing -> refresh daily
                pg = f.get(url, max_age=timedelta(days=1) if y == this else None)
                if not pg.ok:
                    break
                res = parse_svet24_search(pg.text)
                for r in res:
                    found.setdefault(r["url"], r)
                if not res or not svet24_has_next(pg.text, page):
                    break
                page += 1
        log(f"[svet24] year {y}: {len(found)} candidate URLs so far")
    keep = [r for r in found.values()
            if mentions_mushroom(r["title"] + " " + r["snippet"]) or RE_SLUG.search(fold(urlsplit(r["url"]).path))]
    log(f"[svet24] {len(found)} search hits, {len(keep)} look mushroom-related")
    return keep


def siol_keywords_in(html: str) -> list[str]:
    """Mushroom-related keyword pages linked from an article (snowballing)."""
    out = []
    for m in re.finditer(r"kljucne-besede/([a-z0-9-]+-\d+)/articles", html):
        kw = m.group(1)
        if RE_SLUG.search(re.sub(r"-\d+$", "", kw)) or re.search(r"(?<![a-z])(nabiranj)", kw):
            out.append(kw)
    return list(dict.fromkeys(out))


def siol_candidates(f: Fetcher, log, keywords=SIOL_KEYWORDS) -> list[str]:
    found: dict[str, None] = {}
    for kw in keywords:
        page = 1
        while page <= 200:
            url = f"https://siol.net/kljucne-besede/{kw}/articles" + (f"?page={page}" if page > 1 else "")
            pg = f.get(url, max_age=timedelta(days=2) if page <= 2 else None)
            if not pg.ok:
                break
            urls = parse_siol_keyword(pg.text)
            new = [u for u in urls if u not in found]
            if not new:
                break
            found.update(dict.fromkeys(new))
            page += 1
        log(f"[siol] keyword {kw}: {len(found)} article URLs so far")
    return list(found)


# ------------------------------------------------------------------ rows
def make_row(source: str, url: str, art: dict, via: str) -> dict:
    full = f"{art['title']}\n{art['lead']}\n{art['text']}"
    return {"id": stable_id(source, url), "source": source, "section": art["section"], "url": url,
            "published": art["published"], "title": art["title"], "lead": art["lead"], "text": art["text"],
            "mentions_jurcek": mentions_jurcek(full), "mentions_mushroom": mentions_mushroom(full),
            "match_via": via}


def collect(f: Fetcher, source: str, log=print, since_year: int | None = None) -> Iterator[dict]:
    """source: n1 | svet24 | siol.  Article pages are fetched newest-first and cached."""
    if source == "n1":
        cands = n1_candidates(f, log)
        cands.sort(key=lambda x: x[1] or "", reverse=True)
        todo = [(u, "sitemap-slug") for u, _ in cands]
    elif source == "svet24":
        c = svet24_candidates(f, log, since_year)
        c.sort(key=lambda r: r["date"].isoformat() if r["date"] else "", reverse=True)
        todo = [(r["url"], "search") for r in c]
    elif source == "siol":
        yield from _collect_siol(f, log, since_year)
        return
    else:
        raise ValueError(source)
    log(f"[{source}] fetching {len(todo)} articles")
    for i, (u, via) in enumerate(todo):
        try:
            pg = f.get(u)
        except BlockedHost:
            raise
        if not pg.ok:
            continue
        art = parse_article(pg.text, u)
        if art is None or not art["published"]:
            continue
        if since_year and art["published"].year < since_year:
            continue
        row = make_row(source, u, art, via)
        if not row["mentions_mushroom"]:
            continue  # candidate matched on a word like 'Jurčki' (band) but is not about mushrooms
        yield row
        if i % 50 == 0:
            log(f"[{source}] article {i + 1}/{len(todo)}")


def _collect_siol(f: Fetcher, log, since_year) -> Iterator[dict]:
    """Keyword pages -> articles; keywords found on mushroom articles are followed."""
    seen_kw: set[str] = set()
    pending = list(SIOL_KEYWORDS)
    done: set[str] = set()
    while pending:
        seen_kw.update(pending)
        urls = [u for u in siol_candidates(f, log, pending) if u not in done]
        pending = []
        log(f"[siol] fetching {len(urls)} new articles")
        for i, u in enumerate(urls):
            done.add(u)
            pg = f.get(u)
            if not pg.ok:
                continue
            art = parse_article(pg.text, u)
            if art is None or not art["published"]:
                continue
            row = make_row("siol", u, art, "keyword")
            if not row["mentions_mushroom"] or (since_year and art["published"].year < since_year):
                continue
            for kw in siol_keywords_in(pg.text):
                if kw not in seen_kw and kw not in pending:
                    log(f"[siol] new keyword page {kw} (from {u[-50:]})")
                    pending.append(kw)
            yield row
            if i % 50 == 0:
                log(f"[siol] article {i + 1}/{len(urls)}")
