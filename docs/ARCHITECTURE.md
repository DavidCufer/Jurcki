# Jurčki — architecture

Private app that predicts where and when Boletus (jurčki: *B. edulis*,
*B. aereus*, *B. reticulatus*, *B. pinophilus*) fruit in Slovenia.

## Model

`score(cell, day) = habitat(cell) × fruiting_trigger(cell, day)`

* **habitat** (static, per 250 m cell): elevation, slope, aspect, terrain
  position/wetness, tree cover, forest type, soil pH / geology, distance to
  forest edge. Starts as literature rules, then a presence/background model
  trained on known finds.
* **fruiting_trigger** (dynamic): lagged rain sums (accumulated over ~2–4
  weeks), mean temperature over the previous ~20 days (optimum ≈ 13 °C), heat
  spikes, frost, soil moisture, temperature adjusted for cell elevation.

Literature anchors:
* Bielefeld beech-forest monitoring 2015–2024: peak fruiting at ~13 °C mean
  temperature over the preceding 20 days, and fruiting rises with rain over
  a 26-day window.
* Salerni et al. 2023: sudden Tmax spikes suppress yield, measured over
  the 21 days after an event.
* Pinna et al. 2010: wider soil temperature range leads to earlier fruiting.
* Pinar Grande: warm, wet autumns give high yields; cold, dry ones give low
  yields.

## Ground truth / backtesting

1. GBIF occurrences (includes iNaturalist), read from the AWS open-data
   parquet snapshot. These give point, date and species.
2. Forum and news posts (slo-tech "Gobarjenje", gobarska društva, regional
   news). The text is geocoded with a gazetteer to region-level points.

Backtest by year: train on every other year, predict the held-out year
day by day, and score the reported finds against the predicted surface.
Metrics are AUC against background points and the hit rate in the top-k%
of cells.

## Layout

```
backend/jurcki/
  config.py        grid + paths
  grid.py          model grid helpers, Slovenia mask
  sources/         one module per data source (dem, landcover, weather, gbif, forums…)
  features.py      stack static + dynamic features
  model/           prior.py (rules), train.py, backtest.py
  export.py        outputs for the app
android/           Kotlin + Compose + MapLibre viewer
data/processed/    small derived files (committed)
data/output/       app-facing outputs (committed or published)
```

## App data contract (`data/output/`)

* `meta.json` contains
  `{"generated": ISO8601, "days": ["2026-10-24", ...], "bounds": [minLon, minLat, maxLon, maxLat], "legend": [[0.0,"#..."],...]}`
* `score_YYYY-MM-DD.png` is an RGBA heat map in EPSG:4326 covering `bounds`.
  It is transparent where the score is ~0 or outside Slovenia.
* `hotspots_YYYY-MM-DD.geojson` holds the top cells as points, with
  properties `score`, `habitat`, `trigger` and `elev`.
* `reports.geojson` holds recent finds from forums and GBIF, with properties
  `date`, `source`, `precision_m` and `text`.
