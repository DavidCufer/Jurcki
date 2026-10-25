"""Parsers for the web (forum / news) sources, offline, on small saved HTML fixtures."""
import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jurcki.sources.web import gobe_forum, news, slotech, textutil
from jurcki.sources.web import fetch as fetchmod
from jurcki.sources.web.textutil import parse_sl_datetime

FIX = Path(__file__).parent / "fixtures" / "web"
REF = datetime(2026, 10, 24, 20, 0, tzinfo=timezone.utc)  # = 22:00 local, 24 Oct 2026


def read(name):
    return (FIX / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- dates
@pytest.mark.parametrize("s,expected", [
    ("16 September 2026, 10:59", (2026, 9, 16, 10, 59)),
    ("04 December 2023, 14:57", (2023, 12, 4, 14, 57)),
    ("28 Marec 2026, 06:27", (2026, 3, 28, 6, 27)),
    ("27 Oktober 2024, 20:18", (2024, 10, 27, 20, 18)),
    ("27. avg 2016, 21:46", (2016, 8, 27, 21, 46)),
    ("27. avg 2016 ob 22:50", (2016, 8, 27, 22, 50)),
    ("22. okt 2026 13:55:06", (2026, 10, 22, 13, 55)),
    ("7. feb 2026 23:53:28", (2026, 2, 7, 23, 53)),
    ("1. maj 2012", (2012, 5, 1, 0, 0)),
    ("29. 5. 2026, 06.00", (2026, 5, 29, 6, 0)),
    ("24.10.2026 18:04", (2026, 10, 24, 18, 4)),
    ("12. marca 2019 ob 8:05", (2019, 3, 12, 8, 5)),
    ("2016-08-27T21:46:07+02:00", (2016, 8, 27, 21, 46)),
    ("2026-10-24T17:05:11.630585+00:00", (2026, 10, 24, 19, 5)),   # UTC -> local (CEST)
    ("2026-01-10T12:00:00Z", (2026, 1, 10, 13, 0)),                 # CET
])
def test_absolute_dates(s, expected):
    dt = parse_sl_datetime(s, REF)
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == expected
    assert dt.tzinfo is not None


@pytest.mark.parametrize("s,expected", [
    ("Danes ob 18:04", (2026, 10, 24, 18, 4)),
    ("danes ob 18:04", (2026, 10, 24, 18, 4)),
    ("Danesob 18:04", (2026, 10, 24, 18, 4)),          # get_text() without space
    ("Včeraj ob 09:45", (2026, 10, 23, 9, 45)),
    ("včeraj ob 9.45", (2026, 10, 23, 9, 45)),
    ("Predvčerajšnjim ob 10:00", (2026, 10, 22, 10, 0)),
])
def test_relative_dates(s, expected):
    dt = parse_sl_datetime(s, REF)
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == expected


def test_relative_uses_local_day_not_utc():
    ref = datetime(2026, 10, 24, 23, 30, tzinfo=timezone.utc)   # 01:30 on the 25th locally
    assert parse_sl_datetime("Danes ob 00:10", ref).day == 25
    assert parse_sl_datetime("Včeraj ob 23:10", ref).day == 24


def test_unparseable():
    assert parse_sl_datetime("") is None
    assert parse_sl_datetime("kmalu", REF) is None
    assert parse_sl_datetime("Danes ob 18:04") is None  # no reference time -> cannot resolve


# ---------------------------------------------------------------- relevance flags
@pytest.mark.parametrize("t", [
    "Danes polna košara jurčkov na Pohorju", "našel sem jurcke", "en velik jurček", "jurčka sem pustil",
    "Jurčke ni", "veliko gobanov", "so gobani že tu?", "Boletus edulis", "jurcki so ze rastli", "JURČKI!",
    "borovi jurčki in črni gobani",
])
def test_jurcek_flag_positive(t):
    assert textutil.mentions_jurcek(t)


@pytest.mark.parametrize("t", [
    "Marele in lisičke, jurčkov še ni", "Janez Jurčič", "gobec ti zatrem", "danes sem v službi", "kosmatinec",
])
def test_jurcek_flag_negative(t):
    if "jurčkov" in t:
        assert textutil.mentions_jurcek(t)
    else:
        assert not textutil.mentions_jurcek(t)


@pytest.mark.parametrize("t,exp", [
    ("Šel na Pohorje, niti ene gobe", True), ("nabiranje gob", True), ("lisičke in marele", True),
    ("gobarska sezona", True), ("nič ni, suša, gobarji doma", True), ("sirovke", True),
    ("Nobena zadeva", False), ("gobec", False),
])
def test_mushroom_flag(t, exp):
    assert textutil.mentions_mushroom(t) is exp


def test_title_filter():
    assert textutil.title_looks_mushroomy("Gobarjenje")
    assert textutil.title_looks_mushroomy("Katera goba je tole? Štorovka?")
    assert not textutil.title_looks_mushroomy("Dobil sem po gobcu")
    assert not textutil.title_looks_mushroomy("Celoletne gume")


def test_author_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(textutil, "WEB_RAW", tmp_path)
    monkeypatch.setattr(textutil, "_SALT", None)
    h = textutil.author_hash("x", "maco-maco")
    assert h == textutil.author_hash("x", " Maco-Maco ") and len(h) == 16
    assert "maco" not in h
    assert h != textutil.author_hash("y", "maco-maco")
    assert textutil.author_hash("x", None) is None
    assert (tmp_path / "_author_salt").exists()


# ---------------------------------------------------------------- gobe.si forum
def test_gobe_board():
    html = read("gobe_board.html")
    rows = gobe_forum.parse_board(html)
    assert len(rows) == 6
    assert rows[0] == {"topic_id": 17, "title": "Identifikacija gob - prosimo preberite!", "replies": 3}
    assert any(r["title"] == "Determiniranje 2026" and r["replies"] == 103 for r in rows)
    assert gobe_forum.board_page_count(html) == 26


def test_gobe_topic():
    posts = gobe_forum.parse_topic(read("gobe_topic.html"), REF)
    assert [p["msg_id"] for p in posts] == [26894, 26898, 26899]
    assert posts[0]["post_dt"] == datetime(2026, 9, 16, 10, 59, tzinfo=textutil.TZ)
    assert posts[0]["author"] == "klavir"          # kept only in-memory; rows store the hash
    assert posts[0]["text"].startswith("Pozdrav.\n\nOdpiram ponovno temo za Gobarjenje 2026")
    assert posts[1]["image_urls"] == ["https://i.ibb.co/Kxqb3S3x/DSC00548a.jpg"]
    assert "ibb.co" not in posts[1]["text"]
    assert posts[1]["index_in_topic"] == 1


def test_gobe_quote_removed_and_relative_date():
    html = """<div id="forumposts"><div class="windowbg" id="msg7"><div class="poster"><h4><a>Ana</a></h4></div>
    <div class="postarea"><div class="keyinfo"><div class="subject_title"><a>Odg: x</a></div>
    <div class="postinfo"><a class="smalltext">Včeraj ob 09:45</a></div></div>
    <div class="post"><div class="inner"><blockquote><cite>Navedek</cite>stari tekst<blockquote>globlje</blockquote></blockquote>
    Nov odgovor: jurčkov ni.<br/><img class="bbc_img" src="http://a/b.jpg"><img class="smiley" src="http://c/Smileys/x.gif"></div></div></div></div>"""
    p = gobe_forum.parse_topic(html, REF)[0]
    assert p["text"] == "Nov odgovor: jurčkov ni."
    assert p["n_quotes"] == 1 and p["image_urls"] == ["http://a/b.jpg"]
    assert p["post_dt"].date().isoformat() == "2026-10-23"


def test_gobe_rows_unique_ids_and_flags():
    posts = gobe_forum.parse_topic(read("gobe_topic.html"), REF)
    for p in posts:
        p.update(topic_id=1375, thread_title="Gobarjenje 2026 jesen", board="Nabiranje gob")
    rows = list(gobe_forum.to_rows(iter(posts + posts)))   # duplicates dropped
    assert len(rows) == 3
    assert len({r["id"] for r in rows}) == 3
    assert rows[1]["has_images"] and rows[1]["n_images"] == 1
    assert rows[0]["url"] == "https://www.gobe.si/forum/index.php?msg=26894"
    assert "klavir" not in str(rows)
    assert all(r["mentions_mushroom"] for r in rows)


# ---------------------------------------------------------------- slo-tech
def test_slotech_board():
    threads, pages = slotech.parse_board(read("slotech_board.html"))
    assert pages == 1354 and len(threads) == 6
    assert threads[0]["sticky"] and threads[0]["thread_id"] == 46757 and threads[0]["messages"] == 9083
    assert threads[2]["last_dt"].year == 2026 and not threads[2]["sticky"]


def test_slotech_thread():
    d = slotech.parse_thread(read("slotech_thread.html"))
    assert d["title"] == "Gobarjenje" and d["next"] == "/forum/t580582/49"
    posts = {p["post_id"]: p for p in d["posts"]}
    p = posts[5258301]
    assert p["post_dt"] == datetime(2016, 8, 27, 21, 46, 7, tzinfo=textutil.TZ)
    assert p["image_urls"] == ["http://shrani.si/f/1R/ME/3EjgNP9C/g1j.jpg", "http://shrani.si/f/2n/Ik/1B4Ffhpw/g1m.jpg"]
    assert "jurčkov in marel" in p["text"] and "shrani.si" not in p["text"]
    # quoted text is not double counted
    q = posts[5258336]
    assert q["n_quotes"] == 1
    assert "Kaj če bi tole temo" not in q["text"] and "Kamot lahko spohaš tudi jurčka." in q["text"]
    rows = list(slotech.to_rows({**x, "thread_id": 580582, "thread_title": "Gobarjenje", "board": "Loža"}
                                for x in d["posts"]))
    assert len({r["id"] for r in rows}) == len(rows)
    assert any(r["mentions_jurcek"] for r in rows)
    assert rows[0]["url"].startswith("https://slo-tech.com/forum/t580582/p")


# ---------------------------------------------------------------- news
def test_news_svet24_article():
    a = news.parse_article(read("svet24_article.html"), "https://svet24.si/lokalno/dolenjska/novice/jurcki-ze-rastejo-gobarski-eldorado-1902495")
    assert a["title"].startswith("Jurčki že rastejo")
    assert a["published"] == datetime(2026, 5, 29, 6, 15, 48, tzinfo=textutil.TZ)
    assert a["section"] == "lokalno/dolenjska"
    assert "Žrnovca" in a["text"] and "Hvala za prijavo" not in a["text"]
    assert a["lead"].startswith("Poglejte")


def test_news_siol_article():
    a = news.parse_article(read("siol_article.html"), "https://siol.net/novice/slovenija/gobarska-sezona-bolj-klavrna-a-vrhunec-se-obeta-536125")
    assert a["published"].year == 2020 and a["published"].month == 10
    assert "Bovškem" in a["text"] and len(a["text"]) > 2000
    row = news.make_row("siol", "https://siol.net/x-1", a, "keyword")
    assert row["mentions_jurcek"] and row["mentions_mushroom"]


def test_news_n1_article_and_sitemap():
    a = news.parse_article(read("n1_article.html"), "https://n1info.si/novice/slovenija/kaksne-cene-goriv/")
    assert a["published"] == datetime(2026, 10, 24, 19, 5, 11, tzinfo=textutil.TZ)
    assert a["section"] == "novice/slovenija"
    row = news.make_row("n1", "https://n1info.si/x", a, "sitemap-slug")
    assert not row["mentions_mushroom"]
    urls = news.parse_sitemap(read("n1_sitemap.xml"))
    assert len(urls) == 6 and urls[0][1].startswith("2026-10-24")
    assert news.RE_SLUG.search(textutil.fold("/novice/slovenija/jurcki-ze-rastejo/"))
    assert news.RE_SLUG.search(textutil.fold("/magazin/gobarska-sezona-letos/"))
    assert not news.RE_SLUG.search(textutil.fold("/novice/svet/gobec-in-sosedje/"))


def test_news_listing_parsers():
    res = news.parse_svet24_search(read("svet24_search.html"))
    assert len(res) == 6 and res[0]["url"].startswith("https://svet24.si/kulinarika/")
    assert res[0]["date"] == datetime(2026, 10, 6, 5, 3, tzinfo=textutil.TZ)
    assert news.svet24_has_next(read("svet24_search.html"), 1)
    urls = news.parse_siol_keyword(read("siol_keyword.html"))
    assert len(urls) == 6 and all(u.startswith("https://siol.net/") for u in urls)


# ---------------------------------------------------------------- fetcher (no network)
class _Resp:
    def __init__(self, text, status=200, ct="text/html; charset=utf-8"):
        self.content, self.status_code, self.headers = text.encode(), status, {"content-type": ct}
        self.url, self.encoding = "u", "utf-8"


def test_fetcher_cache_robots_and_offline(tmp_path):
    f = fetchmod.Fetcher(min_interval=0.0, root=tmp_path)
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        if url.endswith("/robots.txt"):
            return _Resp("User-agent: *\nDisallow: /private\n")
        return _Resp("<p>čaj</p>")

    f.session.get = fake_get
    p1 = f.get("https://example.org/a?x=1")
    p2 = f.get("https://example.org/a?x=1")
    assert p1.text == "<p>čaj</p>" and not p1.from_cache and p2.from_cache
    assert calls.count("https://example.org/a?x=1") == 1
    assert p2.fetched_at == p1.fetched_at
    with pytest.raises(fetchmod.RobotsDisallowed):
        f.get("https://example.org/private/x")
    assert fetchmod.USER_AGENT.startswith("jurcki-research/")
    off = fetchmod.Fetcher(offline=True, root=tmp_path)
    assert off.get("https://example.org/a?x=1").text == "<p>čaj</p>"
    with pytest.raises(fetchmod.NotCached):
        off.get("https://example.org/never-fetched")
    # stale -> refetch
    f.get("https://example.org/a?x=1", max_age=timedelta(seconds=-1))
    assert calls.count("https://example.org/a?x=1") == 2


def test_siol_keyword_snowball():
    html = ('<a href="/kljucne-besede/gobarjenje-77549/articles">x</a><a href="/kljucne-besede/luka-doncic-12996/articles">y</a>'
            '<a href="/kljucne-besede/nabiranje-62279/articles">z</a><a href="https://siol.net/kljucne-besede/lisicke-5/articles">w</a>')
    assert news.siol_keywords_in(html) == ["gobarjenje-77549", "nabiranje-62279", "lisicke-5"]


def test_medovernet_is_a_noop_when_blocked(tmp_path):
    from jurcki.sources.web import medovernet
    f = fetchmod.Fetcher(offline=True, root=tmp_path)
    with pytest.raises(fetchmod.NotCached):
        list(medovernet.collect(f, log=lambda *_: None))


def test_build_write_combine_report(tmp_path, monkeypatch):
    import pandas as pd
    from jurcki.sources.web import build
    monkeypatch.setattr(build, "INTERIM_WEB", tmp_path / "interim")
    monkeypatch.setattr(build, "PROCESSED", tmp_path)
    mk = lambda i, y, jur: {"id": f"i{i}", "source": "slotech", "board": "Loža", "thread_id": "1", "thread_title": "Gobarjenje",
                            "url": f"u{i}", "post_dt": datetime(y, 9, 1, 12, tzinfo=timezone.utc), "author_hash": f"a{i % 2}",
                            "text": "jurčki " * 3, "has_quote": False, "n_images": 0, "has_images": False, "image_urls": [],
                            "mentions_jurcek": jur, "mentions_mushroom": True}
    build.write_source("posts", "slotech", [mk(1, 2015, True), mk(2, 2016, False), mk(1, 2015, True)])
    out = build.combine("posts")
    df = pd.read_parquet(out)
    assert list(df.id) == ["i1", "i2"] and df.post_dt.dt.year.tolist() == [2015, 2016]
    rep = build.report("posts")
    assert "2015: 1 / 1" in rep and "2016: 1 / 0" in rep


def test_double_escaped_entities():
    assert textutil.unescape_all("&amp;scaron;ampinjoni &amp;amp; jur&amp;#269;ki") == "šampinjoni & jurčki"
    assert textutil.clean_text("Jur&amp;#269;ki  so tu") == "Jurčki so tu"
