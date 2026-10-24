"""Tests for gbif.clean on a synthetic frame (no network)."""
import pandas as pd

from jurcki.sources.gbif import clean, hash_recorders, summarize


def _row(gbifid, **kw):
    base = dict(
        gbifid=str(gbifid), species="Boletus edulis", scientificname="Boletus edulis Bull.",
        decimallatitude=46.05, decimallongitude=14.5, coordinateuncertaintyinmeters=10.0,
        year=2020, month=9, day=15, countrycode="SI", occurrencestatus="PRESENT",
        datasetkey="ds1", institutioncode="X",
    )
    base.update(kw)
    return base


def test_clean_filters_and_dedups():
    rows = [
        _row(1),                                              # kept
        _row(2, decimallatitude=46.0501, coordinateuncertaintyinmeters=50.0),  # dup of 1
        _row(3, occurrencestatus="ABSENT"),                   # absent
        _row(4, month=None, day=None),                        # no full date
        _row(5, decimallatitude=None),                        # no coords
        _row(6, coordinateuncertaintyinmeters=6000.0),        # too imprecise
        _row(7, coordinateuncertaintyinmeters=None, decimallatitude=46.5),  # unknown unc kept
        _row(8, species="Boletus luridus", day=16),           # other species, kept
        _row(9, species=None, scientificname="Boletus", day=17),  # genus only, kept
        _row(10, decimallatitude=0.0, decimallongitude=0.0),  # null island
    ]
    out = clean(pd.DataFrame(rows))
    assert sorted(out["gbifid"]) == ["1", "7", "8", "9"]
    assert out.set_index("gbifid").loc["1", "coordinateuncertaintyinmeters"] == 10.0
    flags = out.set_index("gbifid")["is_jurcki"]
    assert flags["1"] and flags["7"] and not flags["8"] and not flags["9"]
    assert str(out["date"].iloc[0].date()) == "2020-09-15"


def test_summarize_counts():
    out = clean(pd.DataFrame([_row(1), _row(2, species="Boletus aereus", day=1)]))
    s = summarize(out)
    assert s["species"]["n"].sum() == 2
    assert s["year"].loc[2020, "jurcki"] == 2


def test_hash_recorders_hides_names():
    import pyarrow as pa
    col = pa.chunked_array([pa.array([[{"array_element": "Ana Novak"}], None])])
    h = hash_recorders(col)
    assert h[1] is None and len(h[0]) == 12 and "Ana" not in h[0]
