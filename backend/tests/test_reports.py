"""Rule classifier / loaders of jurcki.model.reports.  Offline; every sentence below is synthetic (invented for the
tests) -- no scraped forum or news text may appear in this file."""
import json

import pandas as pd
import pytest

from jurcki.model import reports as R

SUN = pd.Timestamp("2019-09-15")     # a Sunday
MON = pd.Timestamp("2019-09-16")


def cls(text, title="Gobarjenje 2019", when=SUN, board="Nabiranje gob", source="gobe_forum"):
    return R.classify_rules(text, title, board, source, when)


# ---------------------------------------------------------------- text helpers
def test_prep_strips_urls_and_folds():
    t, has_url = R.prep("Jurčki na Pohorju http://example.org/foto/1.jpg  Čudovito!")
    assert has_url
    assert t == "jurcki na pohorju cudovito!"
    assert R.prep("Brez povezave.")[1] is False


@pytest.mark.parametrize("text,n", [
    ("Našel sem dva jurčka in nekaj gobanov.", 2),
    ("Nabral sem jesenske gobane in jurčkov ni bilo.", 2),
    ("Prinesel sem žametasti goban in svinjski goban.", 0),          # named non-jurčki
    ("To je Boletus luridus.", 0),
    ("To je Boletus edulis.", 1),
    ("Imam vrganje in korenjake.", 2),
    ("Danes samo lisičke.", 0),
])
def test_jur_mentions(text, n):
    t, _ = R.prep(text)
    assert len(R.jur_mentions(t)) == n


def test_diacritic_free_spellings_match():
    for s in ("jurckov", "JURČKOV", "jurčkov", "Jurcki"):
        t, _ = R.prep(s)
        assert R.jur_mentions(t), s


# ---------------------------------------------------------------- quantity / elevation / habitat
@pytest.mark.parametrize("text,expected", [
    ("nabral sem 2,5 kg jurčkov", (2.5, "kg")),
    ("pol kile jurčkov", (0.5, "kg")),
    ("dobil sem 12 jurčkov", (12.0, "count")),
    ("našel dva jurčka", (2.0, "count")),
    ("500 g jurčkov", (0.5, "kg")),
    ("nič posebnega", (None, None)),
])
def test_parse_quantity(text, expected):
    t, _ = R.prep(text)
    assert R.parse_quantity(t, R.jur_mentions(t)) == expected


@pytest.mark.parametrize("text,expected", [
    ("na 1200 m nadmorske višine", 1200),
    ("nad 1000 mnv", 1000),
    ("med 800-900 m", 850),
    ("leta 2015 je bilo drugače", None),
    ("hodil sem 5 m naprej", None),
])
def test_parse_elevation(text, expected):
    assert R.parse_elevation(R.prep(text)[0]) == expected


def test_parse_habitat():
    assert R.parse_habitat(R.prep("pod bukvami in smrekami")[0]) == "bukev,smreka"
    assert R.parse_habitat(R.prep("na travniku")[0]) is None


# ---------------------------------------------------------------- relative dates
@pytest.mark.parametrize("text,posted,expected", [
    ("danes zjutraj", SUN, (0, 0)),
    ("včeraj sem bil v gozdu", SUN, (-1, 0)),
    ("predvčerajšnjim smo šli", SUN, (-2, 1)),
    ("v soboto sem nabral", SUN, (-1, 1)),
    ("v nedeljo sem nabral", MON, (-1, 1)),
    ("v nedeljo sem nabral", SUN, (-7, 1)),           # same weekday as the post: a week ago
    ("pred tednom dni", SUN, (-7, 3)),
    ("pred tremi dnevi", SUN, (-3, 2)),
    ("jutri grem v soboto", SUN, (0, 1)),              # future, not an observation date
])
def test_resolve_date(text, posted, expected):
    assert R.resolve_date(R.prep(text)[0], posted) == expected


# ---------------------------------------------------------------- classifier on invented sentences
def test_outing_found():
    x = cls("Danes sem bil na Pohorju in nabral 12 jurčkov, nekaj lisičk.")
    assert x.report_kind == "outing"
    assert x.jurcki_outcome in ("found", "few", "many")
    assert x.quantity_value == 12 and x.quantity_unit == "count"
    assert x.other_mushrooms_outcome == "found"
    assert x.date_offset_days == 0


def test_outing_absence():
    x = cls("Včeraj sem bil v gozdu, ni bilo niti enega jurčka, samo lisičke.")
    assert x.report_kind == "outing"
    assert x.jurcki_outcome == "none"
    assert x.date_offset_days == -1
    assert x.quantity_value is None


def test_not_reports():
    assert cls("Prosim za determinacijo, katera goba je to? Hvala.", "Determiniranje 2019").report_kind == "none"
    assert cls("Recept za jurčkovo juho: najprej ocvremo čebulo.", title="Kulinarika", board="Kulinarika").report_kind == "none"
    assert cls("Lep pozdrav vsem, hvala za čestitke!").report_kind == "none"
    assert cls("Cena jurčkov na tržnici je 40 evrov za kilogram.", title="Cene gob").jurcki_outcome == "not_mentioned"


def test_status_and_elevation():
    x = cls("Na Gorenjskem rastejo jurčki, pognali so po dežju.")
    assert x.report_kind in ("status", "outing")
    y = cls("Danes polna košara jurčkov, ogromno, kar 40 kosov na 900 m.")
    assert y.report_kind == "outing" and y.jurcki_outcome == "many" and y.elev_m == 900


def test_empty_text_is_none():
    assert cls("").report_kind == "none"
    assert cls("   ").report_kind == "none"


# ---------------------------------------------------------------- locations
def test_rule_location_drops_origin():
    pytest.importorskip("pandas")
    m = R.rule_location("Pozdrav iz Maribora, hvala vsem.")
    assert m is None or m.context != "origin"
    m = R.rule_location("Danes sem nabiral jurčke na Pohorju.")
    assert m is not None and "Pohorje" in m.name and 46 < m.lat < 47


def test_geocode_place_bare_name():
    m = R.geocode_place("Pokljuka")
    assert m is not None and 13.5 < m.lon < 14.5
    assert R.geocode_place("Nekraj Kinemanj") is None
    assert R.geocode_place("") is None


def test_geocode_place_aliases():
    m = R.geocode_place("Klopni vrh Šumik")          # missing from the gazetteer -> whole range, flagged
    assert m is not None and m.name == "Pohorje" and "alias" in m.flags
    assert R.geocode_place("Klopnim vrhom").name == "Pohorje"
    k = R.geocode_place("Kope")                       # not the village 'Kopa' in the south
    assert k is not None and k.lat > 46.4
    assert R.geocode_place("Kapla").lat > 46.5


def test_region_of():
    assert R.region_of(15.6, 46.55) == "Podravska"
    assert R.region_of(14.5, 46.05) == "Osrednjeslovenska"
    assert R.region_of(float("nan"), 46.0) == ""


def test_split_is_stable_and_balanced():
    ids = [f"id{i}" for i in range(3000)]
    s = [R.split_id(i) for i in ids]
    assert s == [R.split_id(i) for i in ids]
    for b in (0, 1, 2):
        assert 800 < s.count(b) < 1200


# ---------------------------------------------------------------- label storage + build, on tiny synthetic parquet files
def _write_sources(tmp_path):
    posts = pd.DataFrame({
        "id": ["p1", "p2", "p3", "p4"],
        "source": "gobe_forum",
        "thread_id": ["t1", "t1", "t2", "t1"],
        "thread_title": ["Gobarjenje 2019", "Gobarjenje 2019", "Determiniranje 2019", "Gobarjenje 2019"],
        "board": "Nabiranje gob",
        "post_dt": pd.to_datetime(["2019-09-15T10:00:00Z", "2019-09-16T10:00:00Z", "2019-09-17T10:00:00Z",
                                   "2019-09-18T10:00:00Z"], utc=True),
        "text": ["Včeraj sem na Pohorju nabral dva jurčka.",
                 "Danes sem bil v gozdu, jurčkov ni bilo niti enega, samo lisičke.",
                 "Prosim za determinacijo, katera goba je to?",
                 "Hvala za čestitke, lep pozdrav."],
        "has_quote": False, "mentions_jurcek": [True, True, False, False], "mentions_mushroom": True})
    posts.to_parquet(tmp_path / "posts.parquet")
    pd.DataFrame(columns=["id", "source", "title", "lead", "text", "published", "section", "mentions_jurcek",
                          "mentions_mushroom"]).to_parquet(tmp_path / "news.parquet")
    return tmp_path / "posts.parquet", tmp_path / "news.parquet"


def _label(post_id, **kw):
    row = dict(post_id=post_id, labeler="llm", report_kind="outing", jurcki_outcome="few", other_mushrooms_outcome="unclear",
               quantity_value=2.0, quantity_unit="count", loc_mode="text", loc_text="Pohorje", elev_m=None, habitat=None,
               date_offset_days=-1, date_uncertainty_days=0, paraphrase="Two boletes picked on the mountain yesterday.")
    row.update(kw)
    return row


def test_read_labels_last_wins(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    (d / "batch_000.jsonl").write_text(json.dumps(_label("p1", jurcki_outcome="found")) + "\n")
    (d / "batch_001.jsonl").write_text(json.dumps(_label("p1")) + "\n" + json.dumps(_label("p9", report_kind="none")) + "\n")
    lab = R.read_labels(d)
    assert sorted(lab.post_id) == ["p1", "p9"]
    assert lab.set_index("post_id").loc["p1", "jurcki_outcome"] == "few"
    assert R.read_labels(tmp_path / "nothing").empty


def test_load_sources_and_candidates(tmp_path):
    pp, npth = _write_sources(tmp_path)
    df = R.load_sources(pp, npth)
    assert set(df.post_id) == {"p1", "p2", "p3", "p4"}
    assert str(df.post_dt.dt.tz) == "UTC"
    assert df.date_posted.iloc[0] == pd.Timestamp("2019-09-15")
    cand = R.select_candidates(df)
    assert {"p1", "p2"} <= set(cand.post_id)


def test_build_reports_merges_labels_and_rules(tmp_path):
    pp, npth = _write_sources(tmp_path)
    d = tmp_path / "labels"
    d.mkdir()
    (d / "batch_000.jsonl").write_text(json.dumps(_label("p1")) + "\n")
    out = tmp_path / "field_reports.parquet"
    rep = R.build_reports(d, pp, npth, out_path=out)
    assert list(rep.columns) == R.OUT_COLUMNS
    assert pd.read_parquet(out).shape == rep.shape
    r1 = rep.set_index("post_id").loc["p1"]
    assert r1.labeler == "llm" and r1.report_kind == "outing" and bool(r1.is_field_report)
    assert r1.date_observed == pd.Timestamp("2019-09-14") and r1.date_uncertainty_days == 0
    assert r1.loc_source == "label_text" and r1.loc_name and 46 < r1.loc_lat < 47
    assert r1.region != ""
    # the unlabelled absence post comes from the rules
    assert "p2" in set(rep.post_id)
    r2 = rep.set_index("post_id").loc["p2"]
    assert r2.labeler == "rules" and r2.jurcki_outcome == "none"
    # no post text leaks into the output
    for col in rep.columns:
        assert "text" not in col or col in ("loc_text",)
    assert not rep.drop(columns=["paraphrase"]).astype(str).apply(lambda c: c.str.contains("lisičke|Prosim za")).any().any()
    # rules-only mode ignores the labels
    rep2 = R.build_reports(d, pp, npth, use_labels=False)
    assert set(rep2.labeler) == {"rules"}


def test_cli_missing_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "POSTS_PATH", tmp_path / "no.parquet")
    monkeypatch.setattr(R, "NEWS_PATH", tmp_path / "no2.parquet")
    assert R.main(["-q"]) == 2
    assert "no input" in capsys.readouterr().err


def test_summarize_runs():
    rep = pd.DataFrame({
        "post_id": ["a", "b", "c"], "source": "gobe_forum",
        "date_observed": pd.to_datetime(["2018-09-10", "2018-09-12", "2019-07-01"]),
        "report_kind": "outing", "jurcki_outcome": ["many", "none", "few"], "labeler": ["llm", "llm", "rules"],
        "loc_lon": [15.6, float("nan"), 14.2], "loc_lat": [46.5, float("nan"), 46.3], "loc_radius_m": [8000, float("nan"), 3000],
        "loc_name": ["Pohorje", None, "Bled"], "loc_kind": ["mountain", None, "village"], "loc_source": ["label_text", None, "rules"],
        "region": ["Podravska", "", "Gorenjska"], "elev_m": pd.array([None, 900, None], dtype="Float64"),
        "quantity_value": [3.0, None, None], "habitat": [None, None, "bukev"],
        "paraphrase": ["Many boletes on the mountain.", "Nothing at all.", ""]})
    txt = R.summarize(rep)
    assert "field reports: 3" in txt and "absence share" in txt and "usable location: 2 of 3" in txt


def test_shorten_paraphrase():
    short = "Two boletes picked on the mountain."
    assert R.shorten_paraphrase(short) == short
    assert R.shorten_paraphrase("") == "" and R.shorten_paraphrase(None) == ""
    long = ("Two hours on northern slopes at 350-700 m: violet lepistas, parasols, milkcaps, "
            "bay boletes, one milkcap; no boletes at all.")
    out = R.shorten_paraphrase(long)
    assert len(out.split()) <= 15 and out.endswith(".")
    assert out.endswith("no boletes at all.")                       # the result clause survives
    assert R.shorten_paraphrase("word " * 40).count(" ") <= 15
