"""Slovenian text helpers: diacritic folding, tokenising, inflected name forms.

Everything here is pure Python and offline.  Folding maps every character to
exactly one ASCII character, so offsets in the folded string are valid offsets
in the original text (needed for match spans).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_SPECIAL = {"đ": "d", "ð": "d", "ł": "l", "ß": "s", "æ": "a", "ø": "o", "œ": "o"}


@lru_cache(maxsize=4096)
def _fold_char(ch: str) -> str:
    lo = ch.lower()
    lo = lo[0] if lo else ch
    if lo in _SPECIAL:
        return _SPECIAL[lo]
    nfd = unicodedata.normalize("NFD", lo)
    return nfd[0] if nfd else lo


def fold(text: str) -> str:
    """Lower-case and strip diacritics, preserving string length."""
    return "".join(_fold_char(c) for c in text)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SKIP_BACK = set(" \t\"'„“”‘’(»«[-–—*•_<>/")
_SENT_END = set(".!?:;\n\r")


class Token:
    __slots__ = ("t", "start", "end", "cap", "sent_start", "orig")

    def __init__(self, t, start, end, cap, sent_start, orig):
        self.t, self.start, self.end, self.cap, self.sent_start, self.orig = t, start, end, cap, sent_start, orig

    def __repr__(self):  # pragma: no cover
        return f"Token({self.t!r},{self.start}-{self.end}{',C' if self.cap else ''}{',S' if self.sent_start else ''})"


def tokenize(text: str) -> list[Token]:
    """Folded tokens with offsets, a capitalisation flag and a sentence-start flag."""
    folded = fold(text)
    out = []
    for m in _TOKEN_RE.finditer(folded):
        s, e = m.span()
        # sentence start: walk back over spaces/quotes; start of text or after . ! ? : or newline
        j = s - 1
        while j >= 0 and text[j] in _SKIP_BACK and text[j] not in "\n\r":
            j -= 1
        sent = j < 0 or text[j] in _SENT_END
        out.append(Token(m.group(), s, e, text[s].isupper(), sent, text[s:e]))
    return out


# ---------------------------------------------------------------------------
# Inflection of place names
# ---------------------------------------------------------------------------
PREPOSITIONS = {"na", "v", "ob", "pri", "pod", "nad", "med", "po", "za", "pred", "od", "do", "iz", "s", "z", "k",
                "ko", "u"}
# prepositions that may sit inside an official name ("Radlje ob Dravi", "Smihel pod Nanosom")
NAME_PREPS = {"na", "v", "ob", "pri", "pod", "nad", "med", "za", "pred", "po", "pri"}

# endings of an adjective stem, superset over gender / case / number
AD_SUF = ("i", "a", "o", "e", "ega", "emu", "em", "im", "ih", "imi", "ga")
_ADJ_STEMS = {"velik", "mal", "star", "nov", "bel", "crn", "rdec", "zelen", "visok", "nizk", "dolg", "kratk",
              "lep", "svet", "sveti", "sv", "suh", "mokr", "ravn", "strm", "kamnit", "gozdn", "skofj", "dolenj",
              "gorenj", "notranj", "zgornj", "spodnj", "gornj", "dolnj", "srednj", "prednj", "zadnj", "ozk",
              "siv", "rjav", "rumen", "modr", "vel", "tretj", "prvi"}
_ADJ_END = re.compile(r"(sk|ck|nj)[aeio]$|(sk|ck)i$|nji$")


def _is_adjective(t: str) -> bool:
    if len(t) < 4:
        return t == "sv"
    if _ADJ_END.search(t):
        return True
    if t[-1] in "iaoe" and t[:-1] in _ADJ_STEMS:
        return True
    return False


def adjective_forms(t: str) -> set[str]:
    if t == "sv":
        return {"sv"} | {"svet" + s for s in AD_SUF}
    stem = t[:-1] if t[-1] in "aeio" else t
    return {t} | {stem + s for s in AD_SUF}


def head_forms(t: str, lo: str | None = None, allow_short_gen: bool = False) -> set[str]:
    """Surface forms (folded) of a name's head token across Slovenian cases."""
    lo = lo or t
    n = len(t)
    if n < 3:
        return {t}
    if _is_adjective(t):
        return adjective_forms(t)
    last = t[-1]
    out = {t}
    if t.endswith("je") and n > 4:                      # Pohorje, Kočevje, Posavje (neuter)
        s = t[:-1]
        out |= {s + "a", s + "u", s + "em", s + "ah", s + "ju", s + "ega", s + "emu", s + "im"}
    elif last == "e":                                   # plural feminine: Haloze, Karavanke, Alpe, Jesenice
        b = t[:-1]
        out |= {b + "am", b + "ah", b + "ami"}
        if len(b) > 4 or allow_short_gen:
            out.add(b)                                  # genitive plural (Karavank, Haloz)
    elif last == "o":                                   # neuter: Brdo, Šmartno
        s = t[:-1]
        out |= {s + "a", s + "u", s + "om", s + "em", s + "e", s + "ega", s + "emu", s + "im"}   # Cerkno: v Cerknem, Laško: v Laškem
    elif last == "a":                                   # feminine: Rogla, Pokljuka, Ribnica
        s = t[:-1]
        out |= {s + "e", s + "i", s + "o"}
    elif last == "i":                                   # plural masculine: Gorjanci, Brkini, Javorniki
        s = t[:-1]
        out |= {s + "e", s + "ev", s + "ov", s + "em", s + "om", s + "ih"}
    elif last == "u":
        pass
    elif t == "vas":                                    # feminine i-stem: v vasi, Gorenja vas -> Gorenji vasi
        out |= {"vasi", "vasjo", "vasmi", "vaseh", "vasem"}
    else:                                               # masculine consonant stem: Snežnik, Nanos, Kranj
        out |= {t + "a", t + "u", t + "om", t + "em"}
        if t.endswith("r") and n > 3:                   # Bohor -> Bohorju, Bohorjem (soft stem)
            out |= {t + "ja", t + "ju", t + "jem"}
        if t.endswith("elj") and n > 5:                 # Črnomelj -> Črnomlju
            s = t[:-3] + "lj"
            out |= {s + "a", s + "u", s + "em", s + "ju"}
        if t.endswith(("ec", "ek")) and n > 4:          # fleeting e: Žalec -> Žalca
            s = t[:-2] + t[-1]
            out |= {s + "a", s + "u", s + "om", s + "em"}
        if lo[-1:] in ("č", "š", "ž") and n > 2:         # feminine i-stems: Peč -> Peči, Pečjo
            out |= {t + "i", t + "jo", t + "ji"}
    return out


def name_token_forms(tokens: list[str], origs: list[str] | None = None,
                     allow_short_gen: bool = False) -> list[set[str]]:
    """Per-token form sets for a multi-token name.

    Tokens from the first preposition on stay fixed ("Smihel pod Nanosom");
    the last token before that is the head noun; earlier vowel-final tokens are
    modifiers (adjectives) that inflect; consonant-final modifiers are fixed
    ("Slovenj Gradec").
    """
    origs = origs or tokens
    k = len(tokens)
    prep_at = next((i for i, t in enumerate(tokens) if i > 0 and t in NAME_PREPS), k)
    head = prep_at - 1
    forms: list[set[str]] = []
    for i, (t, o) in enumerate(zip(tokens, origs)):
        if i >= prep_at:
            forms.append({t})
        elif i == head:
            forms.append(head_forms(t, o, allow_short_gen))
        elif t[-1] in "aeio" or _is_adjective(t):
            forms.append(adjective_forms(t) if (_is_adjective(t) or len(t) >= 3) else {t})
        else:
            forms.append({t})
    return forms
