"""Geocode Slovenian free text to places from the gazetteer.

>>> from jurcki.geo import geocode, best_location
>>> best_location("Včeraj sem bil na Pohorju, pod Roglo je bilo polno jurčkov.")

How it works
------------
1. The text is folded (lower-case, no diacritics -- forum users often drop
   č/š/ž) with offsets preserved, and tokenised.
2. Every gazetteer name (and alias) is expanded per token into Slovenian case
   forms (``slovene.name_token_forms``): "Pohorje" -> Pohorja/Pohorju/Pohorjem,
   "Notranjska" -> notranjskem/notranjskih/..., "Kočevski Rog" -> kočevskem
   rogu ...  Matching is by token-form sets, so no stemmer is needed.
3. Candidates are scored: kind of place, ambiguity (same name in several
   places), capitalisation, and preposition context ("na/v/pod/okolica ...",
   "sem iz ..." = where the author lives, which is penalised).
4. Overlaps are resolved by preferring the longest match.
5. Names that are also ordinary words (gora, dol, vas, most, log ...) and all
   very short names are "strict": they only match when capitalised, not at a
   sentence start, and with a preposition right before them.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from .gazetteer import load_gazetteer
from .slovene import NAME_PREPS, _is_adjective, fold, name_token_forms, tokenize

# Names (folded, single token) that are also everyday Slovenian words or hopelessly generic.
STRICT_WORDS = set("""
gora gore dol dolina vas brdo breg log polje hrib laze most sela selo rovt rovte ravne mlaka jama studenec
dobrava gozd planina loka kal kot ledine njive zavrh zaplana vrh grm grmada lipa bela mala velika nova stara
zgornja spodnja zgornje spodnje zgornji spodnji sveti sveta sveto kamen skala pec sredica dob mir sora mirna
tabor gradec zalog sv jesen pod nad ob pri med breze breza bukovje hrastje javor smrek smreka bor borovje
gorenja dolenja gorenje dolenje srednja srednje rudnik rob rep rak ribnik jezero potok reka struga draga dragi plac trg cesta pot hrast jelka grad
kraj mesto zid sejem brod brodi lisice lisica lovec lovci vrata klanec sedlo stene stena zaboj gobe
kamnik ig bled mislinja sava drava soca krka kolpa dravinja savinja
nanosi ocvirk mlin tabor rakovnik zavod zabava
bloke kras
""".split()) - {"kamnik", "bled", "mislinja", "sava", "drava", "soca", "krka", "kolpa", "dravinja", "savinja", "ig"}
# (the river words / Kamnik / Bled / Ig are removed again: rivers are not in the gazetteer,
#  and towns are handled by the generic rules)

LOC_STRONG = {"na", "v", "ob", "pri", "pod", "nad", "med", "po", "skozi", "okoli", "okrog", "blizu", "proti",
              "okolica", "okolici", "okolico", "okolice", "bliznji", "blizini", "mimo", "cez", "iznad", "izpod",
              "podnozju", "vznozju", "pobocju", "obmocju", "obmocje", "predelu", "delu", "koncu", "smer", "smeri",
              "do"}
AROUND = {"okoli", "okrog", "okolica", "okolici", "okolico", "okolice", "blizu", "blizini", "bliznji", "podnozju",
          "vznozju", "pobocju", "obrobju"}
ORIGIN_PREPS = {"iz", "z", "s", "od"}
ORIGIN_MARKERS = {"sem", "smo", "doma", "zivim", "zivimo", "stanujem", "prihajam", "pozdrav", "pozdravljeni",
                  "pozdravi", "domov", "domacin", "ziv", "iz"}
DIRECTIONS = {"severno", "juzno", "vzhodno", "zahodno", "severni", "juzni", "vzhodni", "zahodni", "okolica",
              "zgornje", "spodnje", "srednje", "stari", "novi", "hotel", "gostilna", "dom", "koca", "planina"}
ORG_WORDS = {"drustva", "drustvu", "drustvo", "zavod", "zavoda", "zavodu", "klub", "kluba", "zveza", "zveze", "sluzba",
             "sluzbe", "enota", "enote", "uprava", "uprave", "upravi", "centra", "centru", "bolnisnica", "bolnisnici",
             "podjetje", "podjetja", "d", "o"}
FILLERS = {"naso", "nasem", "nasi", "nasim", "lepem", "lepi", "tem", "istem", "isti", "ravno", "tudi", "vse",
           "zgornjem", "spodnjem", "samem", "sami", "celotnem", "siroki", "siroko", "cisto", "malo", "nekje",
           "kar", "se", "pa", "bolj", "visje", "nizje", "okolici", "okolico", "okolica", "okolice", "blizini"}

_PRIO = {"CUR": 10000, "PPLC": 9000, "PPLA": 1500, "PPLA2": 1400, "ADM1": 1300, "MTS": 3000, "MT": 2500,
         "PK": 2300, "FRST": 2800, "PLAT": 2800, "VAL": 2200, "VALS": 2200, "RGN": 2800, "AREA": 2600,
         "PLN": 2600, "HLLS": 2400, "RDGE": 2300, "MNTS": 2300, "LK": 1800, "HLL": 1200, "PPL": 1000,
         "PPLL": 900, "PPLX": 500}
_BASE_CONF = {"CUR": 0.92, "PPLC": 0.95, "PPLA": 0.88, "PPLA2": 0.85, "ADM1": 0.75, "MTS": 0.85, "MT": 0.78,
              "PK": 0.75, "FRST": 0.82, "PLAT": 0.82, "VAL": 0.8, "VALS": 0.8, "RGN": 0.8, "AREA": 0.8,
              "PLN": 0.8, "HLLS": 0.78, "RDGE": 0.75, "MNTS": 0.75, "LK": 0.7, "HLL": 0.65, "PPL": 0.7,
              "PPLL": 0.66, "PPLX": 0.6}


@dataclass
class Match:
    name: str
    lon: float
    lat: float
    radius_m: float
    kind: str
    confidence: float
    span: tuple[int, int]
    text: str = ""
    gid: str = ""
    context: str = "none"          # locative | around | origin | none
    source: str = ""
    ellipse: tuple[float, float, float] | None = None   # (a_m, b_m, azimuth_deg) for curated regions
    support: int = 0               # best_location: number of consistent other mentions
    flags: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _km(lat1, lon1, lat2, lon2) -> float:
    return math.hypot((lat1 - lat2) * 111.2, (lon1 - lon2) * 111.2 * math.cos(math.radians((lat1 + lat2) / 2)))


def _prio(row) -> float:
    extra = min(row["elevation"], 3000) / 10 if row["kind"] == "mountain" else min(row["population"] / 10.0, 3000)
    return _PRIO.get(row["fcode"], 800) + extra


class Gazetteer:
    """Index built from the gazetteer frame: form lookup + per-name ambiguity."""

    def __init__(self, gaz: pd.DataFrame):
        self.ents = gaz.reset_index(drop=True)
        rows = self.ents.to_dict("records")
        self.rows = rows
        prio = [_prio(r) for r in rows]
        # keys: tuple of folded tokens -> list of (entity idx, alias weight)
        keys: dict[tuple, list] = {}
        self.origs: dict[tuple, list[str]] = {}
        for i, r in enumerate(rows):
            names = [(r["norm_name"], 1.0, r["name"])]
            if r["alt_names"]:
                for a in r["alt_names"].split("|"):
                    if not a:
                        continue
                    if r["source"] == "curated":
                        nm, _, w = a.rpartition(":")
                        names.append((nm, float(w), nm))
                    else:
                        names.append((a, 0.9, a))
            for nm, w, raw in names:
                toks = tuple(t for t in _split(nm))
                if not toks or len(toks) > 7:
                    continue
                keys.setdefault(toks, []).append((i, w))
                if toks not in self.origs:
                    self.origs[toks] = [o for _, o in _split_pairs(raw)]
        self.variants = []         # (forms, entity idx, weight, ambiguity, key)
        self.index: dict[str, list[int]] = {}
        for key, lst in keys.items():
            lst.sort(key=lambda x: -prio[x[0]] * (x[1] if rows[x[0]]["source"] == "curated" else x[1] ** 8))
            pi, w = lst[0]
            p = rows[pi]
            # ambiguity: other entities with this exact name that are >8 km away
            far = [(j, ww) for j, ww in lst[1:] if _km(p["lat"], p["lon"], rows[j]["lat"], rows[j]["lon"]) > 8]
            amb, n_alt = 1.0, len(far)
            flags = []
            if far:
                best_alt = max(prio[j] * (ww if rows[j]["source"] == "curated" else ww ** 8) for j, ww in far)
                ratio = prio[pi] / max(best_alt, 1.0)
                amb = 0.9 if ratio >= 2.4 else 0.75 if ratio >= 1.4 else 0.45
                if n_alt >= 4:
                    amb *= 0.8
                flags.append(f"ambiguous({n_alt + 1})")
            origs = self.origs.get(key, list(key))
            if len(origs) != len(key):
                origs = list(key)
            forms = name_token_forms(list(key), origs,
                                     allow_short_gen=p["source"] == "curated" or p["fcode"] in ("PPLA", "PPLC", "ADM1"))
            strict = bool(p["strict"]) or (len(key) == 1 and (key[0] in STRICT_WORDS or len(key[0]) <= 3)) or (
                len(key) > 1 and all(t in STRICT_WORDS for t in key))
            vid = len(self.variants)
            self.variants.append((forms, pi, w, amb, strict, tuple(flags), key))
            for f in forms[0]:
                self.index.setdefault(f, []).append(vid)


def _split(s: str) -> list[str]:
    return [t for t, _ in _split_pairs(s)]


def _split_pairs(s: str) -> list[tuple[str, str]]:
    """(folded token, lower-case original token with diacritics) pairs."""
    f = fold(s)
    low = s.lower()
    return [(m.group(), low[m.start():m.end()]) for m in re.finditer(r"[a-z0-9]+", f)]


@lru_cache(maxsize=1)
def _default_index() -> Gazetteer:
    return Gazetteer(load_gazetteer())


def _kind(row) -> str:
    if row["kind"] == "place":
        return "town" if (row["fcode"] in ("PPLA", "PPLC", "PPLA2") or row["population"] >= 3000) else "village"
    return row["kind"]


def _gap_ok(text: str, a, b) -> bool:
    return text[a.end:b.start].strip(" \t-–") == ""


_FROM_MARKERS = {"sem", "smo", "je", "doma", "prihajam", "prihajamo", "pozdrav", "pozdravi", "pozdravljeni",
                 "zdravo", "pozdravljen", "sva"}
_LIVE_MARKERS = {"zivim", "zivimo", "stanujem", "stanujemo", "doma", "bivam", "bivamo", "domacin"}


def _context(toks, i, j):
    """Return (context label, confidence factor, radius factor, has_preposition_before)."""
    k = i - 1
    preps = []            # indices of prepositions / "around" words found directly before the name
    steps = 0
    while k >= 0 and steps < 4:
        t = toks[k]
        if t.t in LOC_STRONG or t.t in ORIGIN_PREPS or t.t in NAME_PREPS:
            preps.append(k)
        elif t.t in FILLERS:
            pass
        else:
            break
        if t.sent_start:
            break
        k -= 1
        steps += 1
    if not preps:
        return "none", 0.85, 1.0, False
    outer = preps[-1]
    first = toks[preps[0]].t
    around = any(toks[p].t in AROUND for p in preps)
    prev = toks[outer - 1].t if outer > 0 and not toks[outer].sent_start else ""
    outer_t = toks[outer].t
    if (outer_t == "iz" and prev in _FROM_MARKERS) or (outer_t in ("v", "na", "pri", "iz") and prev in _LIVE_MARKERS):
        return "origin", 0.35, 1.0, True
    if any(toks[p].t in ORIGIN_PREPS for p in preps) and not around:
        return "from", 0.7, 1.2, True
    if around:
        return "around", 1.0, 1.8, True
    if first in ("pod", "nad", "proti", "smer", "smeri", "mimo", "cez", "iznad", "izpod"):
        return "locative", 1.0, 1.3, True
    if first == "do":
        return "locative", 0.8, 1.0, True
    return "locative", 1.0, 1.0, True


def _case_reliable(toks) -> bool:
    n = 0
    for t in toks:
        if t.cap and not t.sent_start and len(t.t) >= 3 and t.t not in NAME_PREPS:
            n += 1
    return n >= 1


def geocode(text: str, min_confidence: float = 0.3, index: Gazetteer | None = None) -> list[Match]:
    """All place mentions in ``text`` (longest-match, non-overlapping), in text order."""
    gz = index or _default_index()
    toks = tokenize(text)
    if not toks:
        return []
    reliable = _case_reliable(toks)
    has_dia = any(c in text.lower() for c in "čšžćđ")
    cands: list[tuple[int, float, int, int, Match, int]] = []
    for i, tk in enumerate(toks):
        for vid in gz.index.get(tk.t, ()):
            forms, pi, w, amb, strict, flags, key = gz.variants[vid]
            n = len(forms)
            j = i + n
            if j > len(toks):
                continue
            ok = True
            for d in range(1, n):
                if toks[i + d].t not in forms[d] or not _gap_ok(text, toks[i + d - 1], toks[i + d]):
                    ok = False
                    break
            if not ok:
                continue
            row = gz.rows[pi]
            ctx, cf, rf, has_prep = _context(toks, i, j)
            head = toks[i]
            capped = head.cap and not head.sent_start
            fl = list(flags)
            curated = row["source"] == "curated"
            major = curated or row["fcode"] in ("PPLA", "PPLC", "PPLA2", "ADM1")
            if strict:
                if not (capped and has_prep and ctx != "origin" and reliable and key[0] not in NAME_PREPS):
                    continue
            # --- capitalisation policy ------------------------------------------------------------
            if capped:
                capf = 1.0
            elif head.cap:                               # capitalised only because it starts a sentence
                if not has_prep and not major and n == 1:
                    continue                             # "Pečico segrejte ...", "Kostanj nadrobite ..."
                capf = 0.9
            else:                                        # lower case
                if curated:
                    capf = 0.8 if reliable else 0.92
                elif reliable:
                    continue                             # "nasekljan koper", "oranžna luč", "na policah"
                elif (major and has_prep) or (has_prep and row["kind"] in ("mountain", "forest", "plateau", "valley", "area", "region", "lake")
                               and row["fcode"] != "HLL"):
                    capf = 0.88
                else:
                    continue
                fl.append("lowercase")
            # single-token adjective-like minor names ("Gorska reševalna služba", "v Gorskem Kotarju")
            if (n == 1 and not major and row["kind"] in ("place", "mountain") and _is_adjective(key[0])
                    and j < len(toks) and _gap_ok(text, toks[j - 1], toks[j])):
                continue
            # diacritics: "Cela vrsta" must not become Čela when the text otherwise writes č/š/ž
            dia = 1.0
            if has_dia:
                name_dia = {c for o in gz.origs.get(key, [""]) for c in o if c in "čšž"}
                if name_dia and not (set(text[toks[i].start:toks[j - 1].end].lower()) & set("čšž")):
                    dia = 0.85 if curated else 0.4
                    fl.append("no-diacritics")
            base = _BASE_CONF.get(row["fcode"], 0.7)
            conf = base * w * float(row["weight"]) * amb * capf * cf * dia
            if n == 1 and len(key[0]) <= 4 and not major:
                conf *= 0.9
            if ctx == "none" and not major:
                conf *= 0.7                  # a bare minor place name (no preposition) is often a surname/word
            conf = max(0.0, min(1.0, conf))
            if conf < min_confidence:
                continue
            ell = None
            if row["source"] == "curated":
                ell = (float(row["a_m"]), float(row["b_m"]), float(row["az_deg"]))
                if ctx == "around":
                    rf = min(rf, 1.4)  # curated regions are already area-sized
            m = Match(name=row["name"], lon=float(row["lon"]), lat=float(row["lat"]),
                      radius_m=float(row["radius_m"]) * rf, kind=_kind(row), confidence=round(conf, 3),
                      span=(toks[i].start, toks[j - 1].end), text=text[toks[i].start:toks[j - 1].end],
                      gid=row["gid"], context=ctx, source=row["source"], ellipse=ell, flags=tuple(fl))
            cands.append((n, conf, toks[i].start, vid, m, i))
    # longest first, then confidence; drop overlaps
    cands.sort(key=lambda c: (-c[0], -(c[1] + (0.15 if c[4].source == "curated" else 0.0)), c[2]))
    taken: list[tuple[int, int]] = []
    chosen: list[tuple[Match, int]] = []
    for _, _, _, _, m, ti in cands:
        s, e = m.span
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        chosen.append((m, ti))
    out: list[Match] = []
    for m, ti in chosen:
        # "Gobarsko društvo Lisička Maribor", "GRS Kranj", "Perutnina Ptuj": a capitalised word glued in
        # front of a bare place name makes it part of an organisation / person name, not a find location
        if m.context == "none" and m.source != "curated" and ti > 0:
            prev = toks[ti - 1]
            if (((prev.cap and prev.t not in NAME_PREPS and prev.t not in DIRECTIONS) or prev.t in ORG_WORDS)
                    and text[prev.end:toks[ti].start].strip(" \t)") == ""
                    and not any(prev.start < te and ts < prev.end for ts, te in taken)):
                m.confidence = round(m.confidence * 0.55, 3)
                m.flags += ("org-prefix",)
                if m.confidence < min_confidence:
                    continue
        out.append(m)
    out.sort(key=lambda m: m.span[0])
    return out


def _contains(big: Match, small: Match) -> bool:
    if big.radius_m < small.radius_m * 1.3:
        return False
    d = _km(big.lat, big.lon, small.lat, small.lon) * 1000
    return d <= big.radius_m * 1.05 + small.radius_m * 0.3


def best_location(text: str, min_confidence: float = 0.45, index: Gazetteer | None = None) -> Match | None:
    """The single most plausible location for the text, or None.

    Mentions of the same place are merged.  Mentions that are consistent with
    each other (a small place inside a named region) support one another and
    the more specific one is returned; "I live in X" mentions only win if
    nothing else is there.
    """
    ms = [m for m in geocode(text, min_confidence=min_confidence, index=index)]
    if not ms:
        return None
    non_origin = [m for m in ms if m.context != "origin"]
    if non_origin:
        ms = non_origin
    # merge repeated mentions of the same gazetteer entry
    merged: dict[str, Match] = {}
    for m in ms:
        if m.gid in merged:
            old = merged[m.gid]
            old.confidence = round(min(1.0, max(old.confidence, m.confidence) + 0.04), 3)
            old.support += 1
        else:
            merged[m.gid] = m
    cand = list(merged.values())
    for a in cand:
        for b in cand:
            if a is b:
                continue
            if _contains(a, b) or _contains(b, a) or _km(a.lat, a.lon, b.lat, b.lon) * 1000 < 0.6 * (a.radius_m + b.radius_m):
                a.support += 1

    def score(m: Match) -> float:
        spec = 1.0 - 0.12 * math.log10(max(m.radius_m, 1500) / 1500)
        return m.confidence * (1 + 0.15 * m.support) * spec

    best = max(cand, key=score)
    best.confidence = round(min(1.0, best.confidence + 0.03 * best.support), 3)
    return best


if __name__ == "__main__":  # pragma: no cover
    import sys
    for line in sys.stdin:
        for m in geocode(line.strip()):
            print(m)
