"""gobe.si forum (SMF 2.1, Gobarsko društvo Lisička Maribor).

Boards (all mushroom related): 1 Nabiranje gob (field reports + id requests,
the "Gobarjenje <year>" season threads live here), 2 Gojenje gob, 3 Kulinarika,
4 Strupene gobe.  The forum is small (~18k posts, ~600 topics), so we crawl every
board and every topic.  robots.txt allows everything for generic agents.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Iterator
from urllib.parse import parse_qs, urlsplit

from .fetch import Fetcher, Page
from .textutil import (author_hash, make_soup, mentions_jurcek, mentions_mushroom, node_text,
                       parse_sl_datetime, stable_id)

SOURCE = "gobe_forum"
BASE = "https://www.gobe.si/forum/index.php"
BOARDS = {1: "Nabiranje gob", 2: "Gojenje gob", 3: "Kulinarika", 4: "Strupene gobe"}
POSTS_PER_PAGE = 15
TOPICS_PER_PAGE = 20


def _soup(html: str):
    return make_soup(html)


def parse_board(html: str) -> list[dict]:
    """Topic rows of a board listing page -> [{topic_id, title, replies}]."""
    out = []
    s = _soup(html)
    for row in s.select("#topic_container .windowbg"):
        a = row.select_one(".message_index_title a[href*='topic=']")
        if not a:
            continue
        m = re.search(r"topic=(\d+)", a["href"])
        if not m:
            continue
        stats = row.select_one(".board_stats")
        rep = re.search(r"Odgovorov:\s*(\d+)", stats.get_text(" ")) if stats else None
        out.append({"topic_id": int(m.group(1)),
                    "title": a.get_text(" ", strip=True),
                    "replies": int(rep.group(1)) if rep else None})
    return out


def board_page_count(html: str) -> int:
    offs = [int(m) for m in re.findall(r"board=\d+\.(\d+)", html)]
    return max(offs) // TOPICS_PER_PAGE + 1 if offs else 1


def parse_topic(html: str, fetched_at: datetime) -> list[dict]:
    """Posts of one topic page (author name kept only transiently for hashing)."""
    s = _soup(html)
    posts = []
    for box in s.select("#forumposts div.windowbg[id^=msg]"):
        mid = int(box["id"][3:])
        inner = box.select_one(".post .inner")
        if inner is None:
            continue
        subj = box.select_one(".keyinfo .subject_title a")
        title = subj.get_text(" ", strip=True) if subj else ""
        dl = box.select_one(".postinfo a.smalltext") or box.select_one(".postinfo")
        dt = parse_sl_datetime(dl.get_text(" ", strip=True), fetched_at) if dl else None
        h4 = box.select_one(".poster h4")
        author = h4.get_text(" ", strip=True) if h4 else None
        # images (skip smileys/emoticons); keep URLs only
        imgs = []
        for im in inner.select("img.bbc_img"):
            if "/Smileys/" not in im.get("src", ""):
                imgs.append(im.get("src"))
        for im in box.select(".attachments img"):
            if im.get("src"):
                imgs.append(im["src"])
        # quotes: drop (nested quotes are inside the outer blockquote)
        quotes = inner.find_all("blockquote")
        n_quotes = len([q for q in quotes if not q.find_parent("blockquote")])
        for q in quotes:
            q.decompose()
        for im in inner.find_all("img"):
            im.decompose()
        for sc in inner.find_all(["script", "style"]):
            sc.decompose()
        text = node_text(inner)
        num = box.select_one(".page_number")
        posts.append({
            "msg_id": mid, "subject": title, "post_dt": dt, "author": author,
            "text": text, "image_urls": imgs, "n_quotes": n_quotes,
            "index_in_topic": int(num.get_text().strip("# ")) if num and num.get_text().strip("# ").isdigit() else None,
        })
    return posts


def _topic_url(tid: int, off: int) -> str:
    return f"{BASE}?topic={tid}.{off}"


def iter_topics(f: Fetcher, boards=BOARDS, board_ttl=timedelta(days=3)) -> Iterator[dict]:
    seen = set()
    for bid, bname in boards.items():
        url = f"{BASE}?board={bid}.0"
        pg = f.get(url, max_age=board_ttl)
        n = board_page_count(pg.text) if pg.ok else 0
        for k in range(n):
            pg = f.get(f"{BASE}?board={bid}.{k * TOPICS_PER_PAGE}", max_age=board_ttl)
            if not pg.ok:
                continue
            for t in parse_board(pg.text):
                if t["topic_id"] in seen:
                    continue
                seen.add(t["topic_id"])
                t["board_id"], t["board"] = bid, bname
                yield t


def iter_posts(f: Fetcher, boards=BOARDS, log=print) -> Iterator[dict]:
    topics = list(iter_topics(f, boards))
    log(f"[gobe] {len(topics)} topics")
    for i, t in enumerate(topics):
        tid = t["topic_id"]
        exp = (t["replies"] + 1) if t["replies"] is not None else None
        npages = math.ceil(exp / POSTS_PER_PAGE) if exp else 1
        k = 0
        while k < npages:
            off = k * POSTS_PER_PAGE
            want = min(POSTS_PER_PAGE, exp - off) if exp else 1

            def stale(p: Page, want=want) -> bool:
                return p.ok and len(parse_topic(p.text, p.fetched_at)) < want

            pg = f.get(_topic_url(tid, off), refetch_if=stale)
            if not pg.ok:
                break
            posts = parse_topic(pg.text, pg.fetched_at)
            if exp is None and len(posts) == POSTS_PER_PAGE:
                npages += 1
            for p in posts:
                p.update(topic_id=tid, thread_title=t["title"], board=t["board"])
                yield p
            k += 1
        if i % 25 == 0:
            log(f"[gobe] topic {i + 1}/{len(topics)} (cache-miss requests so far: {f.n_network})")


def to_rows(posts: Iterator[dict]) -> Iterator[dict]:
    seen = set()
    for p in posts:
        if p["msg_id"] in seen:
            continue
        seen.add(p["msg_id"])
        txt = p["text"]
        full = f"{p['thread_title']}\n{txt}"
        yield {
            "id": stable_id(SOURCE, p["msg_id"]),
            "source": SOURCE,
            "board": p["board"],
            "thread_id": str(p["topic_id"]),
            "thread_title": p["thread_title"],
            "url": f"{BASE}?msg={p['msg_id']}",
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


def collect(f: Fetcher, log=print) -> Iterator[dict]:
    yield from to_rows(iter_posts(f, log=log))
