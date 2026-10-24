"""Geocoding of Slovenian free text (forum posts, news) to a place + radius.

    from jurcki.geo import geocode, best_location
"""


def __getattr__(name):          # lazy, so ``python -m jurcki.geo.gazetteer`` does not import the matcher twice
    if name in ("Match", "geocode", "best_location"):
        from . import matcher
        return getattr(matcher, name)
    raise AttributeError(name)
