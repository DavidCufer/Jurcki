"""slo-tech.com forum.

There is no mushroom board; field reports live in off-topic threads ("Gobarjenje",
"Katera goba je tole?", ...) in the boards "Loža" (16) and "Šport" (51).
robots.txt forbids /forum/isci (search), so threads are found by walking the board
listings (/forum/<board>/<page>) and keeping threads whose *title* looks
mushroom-related, plus a few known seed threads.  Posts carry ISO timestamps.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterator

from .fetch import Fetcher
from .textutil import (author_hash, make_soup, mentions_jurcek, mentions_mushroom,
                       node_text, parse_sl_datetime, stable_id, title_looks_mushroomy)

SOURCE = "slotech"
BASE = "https://slo-tech.com"
BOARDS = {16: "Loža", 51: "Šport"}
SEED_THREADS = [580582, 435648, 431088, 230696]  # Gobarjenje, Katera goba je tole?, ...


def parse_board(html: str) -> tuple[list[dict], int]:
    """-> (threads [{thread_id,title,messages,last_dt,sticky}], number_of_pages)"""
    s = make_soup(html)
    out = []
    for tr in s.select("table.forums tbody tr"):
        a = tr.select_one("td.name h3 a[href*='/forum/t']")
        if not a:
            continue
        m = re.search(r"/forum/t(\d+)", a["href"])
        msgs = tr.select_one("td.messages")
        t = tr.select_one("td.last_msg time")
        out.append({"thread_id": int(m.group(1)), "title": a.get_text(" ", strip=True),
                    "messages": int(re.sub(r"\D", "", msgs.get_text()) or 0) if msgs else None,
                    "last_dt": parse_sl_datetime(t["datetime"]) if t and t.get("datetime") else None,
                    "sticky": "sticky" in (tr.get("class") or [])})
    pages = [int(x) for x in re.findall(r'href="/forum/\d+/(\d+)"', html)]
    return out, (max(pages) + 1 if pages else 1)


def parse_thread(html: str) -> dict:
    """-> {title, posts:[...], next: relative url or None}"""
    s = make_soup(html)
    title = ""
    t = s.find("title")
    if t:
        title = re.sub(r"\s*@\s*Slo-Tech\s*$", "", t.get_text()).strip()
    posts = []
    for box in s.select("div.post"):
        anchor = box.find("a", attrs={"name": re.compile(r"^p\d+$")})
        content = box.select_one("div.content")
        if not anchor or content is None:
            continue
        pid = int(anchor["name"][1:])
        h4 = box.find("h4")
        author = None
        if h4:
            a = h4.find("a")
            author = a.get_text(" ", strip=True) if a else None
        tm = h4.find("time") if h4 else None
        dt = parse_sl_datetime(tm["datetime"]) if tm is not None and tm.get("datetime") else None
        imgs = []
        for d in content.select("div.image"):
            a = d.find("a", href=True)
            im = d.find("img", src=True)
            u = (a["href"] if a else None) or (im["src"] if im else None)
            if u:
                imgs.append(u)
        n_quotes = len(content.select("blockquote"))
        for q in content.select("p.user-quoted, blockquote"):
            q.decompose()
        for q in content.select("div.image, script, style"):
            q.decompose()
        for im in content.find_all("img"):
            im.decompose()
        posts.append({"post_id": pid, "author": author, "post_dt": dt,
                      "text": node_text(content), "image_urls": imgs, "n_quotes": n_quotes})
    nxt = s.select_one("div.pages a[rel=next]")
    return {"title": title, "posts": posts, "next": nxt["href"] if nxt else None}


def iter_threads(f: Fetcher, log=print, since_year: int | None = None,
                 board_ttl=timedelta(days=7)) -> Iterator[dict]:
    seen = set()
    for bid, bname in BOARDS.items():
        pg = f.get(f"{BASE}/forum/{bid}", max_age=board_ttl)
        _, n = parse_board(pg.text)
        log(f"[slotech] board {bname}: {n} listing pages")
        for k in range(n):
            pg = f.get(f"{BASE}/forum/{bid}" + (f"/{k}" if k else ""), max_age=board_ttl)
            threads, _ = parse_board(pg.text)
            for t in threads:
                if t["thread_id"] in seen or not title_looks_mushroomy(t["title"]):
                    continue
                seen.add(t["thread_id"])
                t["board"] = bname
                yield t
            if since_year:
                dts = [t["last_dt"] for t in threads if not t["sticky"] and t["last_dt"]]
                if dts and max(dts).year < since_year:
                    break
            if k % 100 == 0:
                log(f"[slotech] {bname} listing page {k + 1}/{n}")
    for tid in SEED_THREADS:
        if tid not in seen:
            seen.add(tid)
            yield {"thread_id": tid, "title": "", "messages": None, "board": "Loža", "last_dt": None}


def iter_posts(f: Fetcher, log=print, since_year: int | None = None) -> Iterator[dict]:
    threads = list(iter_threads(f, log, since_year))
    log(f"[slotech] {len(threads)} candidate threads")
    for i, t in enumerate(threads):
        url = f"{BASE}/forum/t{t['thread_id']}"
        got = {}
        title = t["title"]
        last_url, last_from_cache = url, False
        guard = 0
        while url and guard < 400:
            guard += 1
            pg = f.get(url)
            if not pg.ok:
                break
            d = parse_thread(pg.text)
            title = title or d["title"]
            for p in d["posts"]:
                got[p["post_id"]] = p
            last_url, last_from_cache = url, pg.from_cache
            url = BASE + d["next"] if d["next"] else None
        if (t["messages"] and len(got) < t["messages"] and last_from_cache and not f.offline):
            pg = f.get(last_url, max_age=timedelta(hours=12))
            d = parse_thread(pg.text) if pg.ok else {"posts": [], "next": None}
            for p in d["posts"]:
                got[p["post_id"]] = p
            nxt = d["next"]
            while nxt:
                pg = f.get(BASE + nxt)
                if not pg.ok:
                    break
                d = parse_thread(pg.text)
                for p in d["posts"]:
                    got[p["post_id"]] = p
                nxt = d["next"]
        for p in got.values():
            p.update(thread_id=t["thread_id"], thread_title=title, board=t["board"])
            yield p
        if i % 10 == 0:
            log(f"[slotech] thread {i + 1}/{len(threads)}: {title[:50]!r} ({len(got)} posts)")


def to_rows(posts: Iterator[dict]) -> Iterator[dict]:
    for p in posts:
        txt = p["text"]
        full = f"{p['thread_title']}\n{txt}"
        yield {
            "id": stable_id(SOURCE, p["post_id"]),
            "source": SOURCE,
            "board": p["board"],
            "thread_id": str(p["thread_id"]),
            "thread_title": p["thread_title"],
            "url": f"{BASE}/forum/t{p['thread_id']}/p{p['post_id']}#p{p['post_id']}",
            "post_dt": p["post_dt"],
            "author_hash": author_hash(SOURCE, p["author"]),
            "text": txt,
            "has_quote": p["n_quotes"] > 0,
            "n_images": len(p["image_urls"]),
            "has_images": len(p["image_urls"]) > 0,
            "image_urls": p["image_urls"],
            "mentions_jurcek": mentions_jurcek(txt),
            "mentions_mushroom": mentions_mushroom(full),
        }


def collect(f: Fetcher, log=print, since_year: int | None = None) -> Iterator[dict]:
    """Keep a thread's posts only if at least one post (or the title) mentions mushrooms."""
    by_thread: dict[str, list[dict]] = {}
    for r in to_rows(iter_posts(f, log, since_year)):
        by_thread.setdefault(r["thread_id"], []).append(r)
    for tid, rows in by_thread.items():
        if not any(r["mentions_mushroom"] for r in rows):
            continue  # title matched a word like "gobec" but nobody talks about mushrooms
        yield from rows
