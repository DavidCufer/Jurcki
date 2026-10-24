"""Precision / recall of the geocoder on the labelled sets in eval_set.py.

    python -m jurcki.geo.evaluate [--verbose]
"""
from __future__ import annotations

import sys

from . import eval_set
from .matcher import best_location, geocode
from .slovene import fold


def run(cases, verbose=False, min_conf_best=0.45):
    tp = fp = fn = 0          # mention level
    b_tp = b_fp = b_fn = b_tn = 0   # best_location level
    fails = []
    for text, mentions, best in cases:
        pred = {fold(m.name) for m in geocode(text)}
        exp = {fold(x) for x in mentions}
        tp += len(pred & exp)
        fp += len(pred - exp)
        fn += len(exp - pred)
        if pred != exp:
            fails.append(("mention", text, sorted(exp), sorted(pred)))
        b = best_location(text, min_confidence=min_conf_best)
        acc = {fold(x) for x in best}
        if b is None:
            if acc:
                b_fn += 1
                fails.append(("best-miss", text, sorted(acc), None))
            else:
                b_tn += 1
        elif fold(b.name) in acc:
            b_tp += 1
        else:
            b_fp += 1
            if acc:
                b_fn += 1
            fails.append(("best-wrong", text, sorted(acc), f"{b.name} ({b.confidence})"))
    P = lambda a, b: a / (a + b) if a + b else float("nan")
    res = dict(n=len(cases),
               mention_precision=P(tp, fp), mention_recall=P(tp, fn), mention_tp_fp_fn=(tp, fp, fn),
               best_precision=P(b_tp, b_fp), best_recall=P(b_tp, b_fn), best_tp_fp_fn_tn=(b_tp, b_fp, b_fn, b_tn))
    if verbose:
        for f in fails:
            print(*f, sep=" | ")
    return res, fails


if __name__ == "__main__":
    v = "--verbose" in sys.argv
    for name in ("SET_A", "SET_B"):
        cases = getattr(eval_set, name)
        if cases:
            print(name, run(cases, v)[0])
