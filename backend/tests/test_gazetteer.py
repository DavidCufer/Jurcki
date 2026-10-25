"""Gazetteer + matcher tests.  No network: they use data/processed/gazetteer_si.parquet if present,
otherwise the GeoNames cache in data/raw/geonames, and fall back to the curated regions only."""
import json

import numpy as np
import pandas as pd
import pytest

from jurcki import config
from jurcki.geo import eval_set, evaluate, gazetteer
from jurcki.geo.matcher import Gazetteer, best_location, geocode
from jurcki.geo.regions_curated import CURATED
from jurcki.geo.slovene import fold, head_forms, name_token_forms, tokenize

FULL = gazetteer.GAZ_PATH.exists() or (gazetteer.GN_DIR / "dump" / "SI.txt").exists()
needs_full = pytest.mark.skipif(not FULL, reason="GeoNames SI dump / gazetteer parquet not available")


@pytest.fixture(scope="module")
def gz():
    if FULL:
        return Gazetteer(gazetteer.load_gazetteer())
    return Gazetteer(gazetteer._curated_frame())


# --- text helpers ------------------------------------------------------------------------------
def test_fold_keeps_length_and_strips_diacritics():
    s = "Črnomelj, Šmarna Gora, Žiri, Đurđevac, ČŠŽ"
    f = fold(s)
    assert len(f) == len(s)
    assert f.startswith("crnomelj, smarna gora, ziri, durdevac")


def test_tokenize_flags():
    t = tokenize("Bil sem na Pohorju. Pohorje je lepo.")
    assert [x.t for x in t][:4] == ["bil", "sem", "na", "pohorju"]
    assert t[3].cap and not t[3].sent_start
    assert t[4].sent_start and t[4].cap
    assert t[0].sent_start


@pytest.mark.parametrize("name,forms", [
    ("pohorje", {"pohorju", "pohorja", "pohorjem"}),
    ("snežnik", {"sneznikom", "sneznika", "sezniku"} - {"sezniku"} | {"sneznikom"}),
    ("rogla", {"rogli", "roglo", "rogle"}),
    ("gorjanci", {"gorjancih", "gorjancev", "gorjancem", "gorjance"}),
    ("haloze", {"halozah", "haloz", "halozam"}),
    ("notranjska", {"notranjskem", "notranjskih", "notranjsko", "notranjski"}),
    ("bohor", {"bohorju", "bohorjem"}),
    ("kocevje", {"kocevju", "kocevjem", "kocevja"}),
])
def test_head_forms(name, forms):
    assert forms <= head_forms(fold(name))


def test_multiword_forms_agree_per_token():
    f = name_token_forms(["trnovski", "gozd"])
    assert "trnovskem" in f[0] and "gozdu" in f[1]
    f = name_token_forms(["ribnica", "na", "pohorju"])
    assert "ribnici" in f[0] and f[2] == {"pohorju"}
    f = name_token_forms(["slovenj", "gradec"])
    assert f[0] == {"slovenj"} and "gradcu" in f[1]


# --- gazetteer ---------------------------------------------------------------------------------
def test_curated_regions_are_inside_slovenia_bbox_and_sane():
    minx, miny, maxx, maxy = config.BBOX_WGS84
    names = set()
    for c in CURATED:
        assert minx - 0.2 <= c["lon"] <= maxx + 0.2 and miny - 0.1 <= c["lat"] <= maxy + 0.1, c["name"]
        assert 1 <= c["b_km"] <= c["a_km"] <= 60, c["name"]
        assert 0 <= c["az"] <= 180
        assert c["basis"]
        assert c["name"] not in names
        names.add(c["name"])
    for must in ("Pohorje", "Kočevski Rog", "Snežnik", "Javorniki", "Trnovski gozd", "Nanos", "Hrušica", "Pokljuka",
                 "Jelovica", "Mežakla", "Menina planina", "Gorjanci", "Bohor", "Haloze", "Slovenske gorice", "Goričko",
                 "Kozjansko", "Krimsko hribovje", "Bloke", "Notranjska", "Dolenjska", "Gorenjska", "Štajerska",
                 "Koroška", "Primorska", "Bela krajina", "Prekmurje", "Kras", "Brkini", "Idrijsko", "Cerkljansko",
                 "Škofjeloško hribovje", "Polhograjsko hribovje", "Velika planina", "Logarska dolina", "Solčavsko",
                 "Uršlja gora", "Boč", "Donačka gora", "Rogla", "Areh", "Kozjak", "Paški Kozjak", "Banjška planota"):
        assert must in names, must


def test_field_report_gap_places_are_curated():
    by = {c["name"]: c for c in CURATED}
    for n in ("Smrekovec", "Resevna", "Košenjak", "Kobansko", "Sorško polje", "Jošt", "Uskovnica", "Murska šuma",
              "Ljubljansko barje", "Istra", "Šumik", "Osankarica", "Črno jezero", "Klopni vrh"):
        assert n in by, n
        assert "estimated" in by[n]["basis"] or "GeoNames" in by[n]["basis"]
    # coarse sanity: sub-areas of the Pohorje lie inside the Pohorje ellipse's bounding box, Jošt is above Kranj
    for n in ("Šumik", "Osankarica", "Črno jezero", "Klopni vrh"):
        assert 46.40 < by[n]["lat"] < 46.55 and 15.2 < by[n]["lon"] < 15.6 and by[n]["a_km"] <= 4
    assert abs(by["Jošt"]["lat"] - 46.245) < 0.02 and abs(by["Jošt"]["lon"] - 14.30) < 0.02
    assert by["Istra"]["lat"] < 45.7 and by["Ljubljansko barje"]["lat"] < 46.05


def test_ellipse_polygon_closed_and_sized():
    ring = gazetteer.ellipse_polygon(46.0, 14.5, 10_000, 4_000, 90)
    assert ring[0] == ring[-1]
    lon = np.array([p[0] for p in ring])
    lat = np.array([p[1] for p in ring])
    width_km = (lon.max() - lon.min()) * 111.2 * np.cos(np.radians(46))
    height_km = (lat.max() - lat.min()) * 111.2
    assert width_km == pytest.approx(20, rel=0.05) and height_km == pytest.approx(8, rel=0.1)   # az 90 => E-W major axis


@needs_full
def test_gazetteer_frame():
    g = gazetteer.load_gazetteer()
    assert len(g) > 6000
    assert {"gid", "name", "norm_name", "kind", "lon", "lat", "radius_m", "source"} <= set(g.columns)
    assert g.radius_m.between(500, 40_000).all()
    assert g.lon.between(13.0, 17.0).all() and g.lat.between(45.2, 47.0).all()
    assert (g.source == "curated").sum() == len(CURATED)
    assert {"place", "mountain", "forest", "valley", "region"} <= set(g.kind)
    row = g[(g.norm_name == "kocevje") & (g.fcode == "PPLA")].iloc[0]
    assert abs(row.lat - 45.64) < 0.03 and abs(row.lon - 14.86) < 0.03
    # towns are bigger than villages
    lj = g[g.norm_name == "ljubljana"].iloc[0].radius_m
    assert lj > 6000


@needs_full
def test_regions_geojson():
    p = gazetteer.REGIONS_PATH
    if not p.exists():
        gazetteer.save_regions_geojson(p)
    gj = json.loads(p.read_text())
    assert len(gj["features"]) == len(CURATED)
    f = gj["features"][0]
    assert f["geometry"]["type"] == "Polygon" and f["properties"]["approximate"] is True


# --- matcher -----------------------------------------------------------------------------------
@pytest.mark.parametrize("text,name", [
    ("Včeraj sem bil na Pohorju.", "Pohorje"),
    ("pod Snežnikom smo nabrali jurčke", "Snežnik"),
    ("V Gorjancih je bilo suho", "Gorjanci"),
    ("na Notranjskem so jurčki", "Notranjska"),
    ("v notranjskih gozdovih", "Notranjska"),
    ("proti Rogli", "Rogla"),
    ("v kocevskem rogu", "Kočevski Rog"),
    ("na mezakli", "Mežakla"),
    ("V Trnovskem gozdu", "Trnovski gozd"),
    ("na Bohorju", "Bohor"),
])
def test_curated_inflections_and_missing_diacritics(gz, text, name):
    ms = geocode(text, index=gz)
    assert [m.name for m in ms] == [name], ms


@needs_full
@pytest.mark.parametrize("text,name", [
    ("okolica Kočevja je polna gob", "Kočevje"),
    ("Okolica Ljubljane", "Ljubljana"),
    ("bil sem v Črnomlju", "Črnomelj"),
    ("pri Zrečah", "Zreče"),
    ("okoli Kranjske Gore", "Kranjska Gora"),
    ("v Ribnici na Pohorju", "Ribnica na Pohorju"),
    ("v Novem mestu", "Novo Mesto"),
])
def test_town_inflections(text, name):
    ms = geocode(text)
    assert ms and ms[0].name == name, ms


@needs_full
@pytest.mark.parametrize("text", [
    "Na gori je bilo veliko gob, v dolini pa nič.",
    "Šel sem čez most in naprej po polju.",
    "Dol je bilo suho, gor pa mokro.",
    "Pri mostu sem parkiral.",
    "V hribih je bilo vroče.",
    "Včeraj sem bil v gozdu, bil je dež.",
    "Pečico segrejte na 200 stopinj. Cela vrsta gob.",
    "Suhe jurčke namočimo v vodi.",
    "Včeraj sva z Jurijem dodala nasekljan koper in prižgala luč.",
])
def test_common_words_do_not_match(text):
    assert best_location(text) is None, geocode(text)


@needs_full
def test_origin_is_not_the_find_location():
    assert best_location("Sem iz Maribora, jurčkov ne iščem.") is None
    m = best_location("Pozdrav iz Ljubljane! Včeraj na Pohorju polno jurčkov.")
    assert m and m.name == "Pohorje"


@needs_full
def test_best_location_prefers_specific_consistent_mention():
    m = best_location("Na Pohorju, bolj natančno na Rogli, sem našel 5 jurčkov.")
    assert m and m.name == "Rogla"
    m = best_location("Na Gorenjskem sem našel jurčke blizu Bleda.")
    assert m and m.name == "Bled"


@needs_full
def test_region_only_returns_region_with_ellipse():
    m = best_location("Na Dolenjskem je že sezona.")
    assert m and m.name == "Dolenjska" and m.kind == "region" and m.ellipse and m.radius_m > 20_000


@needs_full
def test_around_widens_radius():
    a = geocode("v Kočevju")[0]
    b = geocode("okolica Kočevja")[0]
    assert b.radius_m > 1.5 * a.radius_m and b.context == "around"


@needs_full
def test_spans_point_into_original_text():
    t = "Šel sem v Črnomelj, potem pod Snežnikom."
    for m in geocode(t):
        assert fold(t[m.span[0]:m.span[1]]).startswith(fold(m.text)[:3])
        assert t[m.span[0]:m.span[1]] == m.text


def test_empty_and_no_place():
    assert geocode("") == []
    assert best_location("Ali veste, kdaj bodo letos jurčki?") is None


# --- labelled evaluation -----------------------------------------------------------------------
@needs_full
def test_eval_set_sizes():
    assert len(eval_set.SET_A) + len(eval_set.SET_B) >= 80


@needs_full
def test_eval_dev_set_quality():
    res, _ = evaluate.run(eval_set.SET_A)
    assert res["mention_precision"] >= 0.95 and res["mention_recall"] >= 0.95
    assert res["best_precision"] >= 0.95 and res["best_recall"] >= 0.95


@needs_full
def test_eval_second_set_quality():
    res, _ = evaluate.run(eval_set.SET_B)
    assert res["mention_precision"] >= 0.9 and res["mention_recall"] >= 0.9
    assert res["best_precision"] >= 0.88 and res["best_recall"] >= 0.9
