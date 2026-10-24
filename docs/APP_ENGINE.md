# On-phone engine spec

The Android app computes the maps itself (no server). Python is the training / backtest lab: it
writes a **model package** and **parity test vectors**; the Kotlin engine implements this spec and is
verified against the vectors.

* Machine-readable twin of this document: `android/app/src/main/assets/model/model.json`
  (parameters, season table, reference, lattice points, request templates, colour table).
  **Read all numbers from `model.json`, do not copy them from this text.**
* Reference implementation of the spec in plain Python: `backend/jurcki/app_engine.py`
  (tests prove it equals the production code: `backend/tests/test_app_package.py`).
* Builder: `python -m jurcki.app_package` (`backend/jurcki/app_package.py`).

```
score(pixel, day) = clamp( habitat(pixel) * season_factor(month(day)) * trigger(pixel, day) / reference , 0, 1 )
```

Data flow of one refresh:

```
Open-Meteo (3 requests, 120 coarse points) -> daily series per point -> gap filling
  -> 5 coarse features per point and day (trailing windows)
  -> per pixel: bilinear interpolation of the features + lapse-rate shift (elevation raster)
  -> trigger(pixel, day) -> score -> RGBA bitmap (one image quad) + hotspots
```

## 1. Package files (`assets/model/`)

| file | content |
|---|---|
| `habitat.u8.bin` | uint8, habitat 0..1 stored as `round(h * 250)` (0..250), **255 = nodata** (outside Slovenia) |
| `elev.i16.bin` | int16 little endian, metres, **-32768 = nodata** (outside Slovenia) |
| `mask.u8.bin` | uint8, 1 = Slovenia land, 0 = outside |
| `model.json` | everything below in machine-readable form |
| `stats.json` | texts and numbers of the "model / success" screen (sl + en) |
| `finds.json` | dated finds 2010+ with backtest values |

All three rasters: `nx * ny` values, **row-major, row 0 = north**, no header, gzip-compressed
(`java.util.zip.GZIPInputStream`). Decode: `habitat = byte / 250.0` (treat nodata inside the mask as 0);
`elev = short.toDouble()`. `habitat` and `elev` are nodata exactly where `mask == 0`.
Total size is about 0.9 MB.

## 2. Raster geometry (Web-Mercator aligned)

From `model.json` `raster`: `nx = 1009`, `ny = 661`, `px = 360.0` (Mercator metres per pixel, = 250 m on
the ground at 46.1 N), `x0 = 1487228.3969981347`, `y_top = 5924260.199618344`, `r_earth = 6378137.0`,
`bounds = [minLon 13.36, minLat 45.41, maxLon 16.6230404, maxLat 46.8907519]` (outer pixel edges).
Columns are even in longitude and rows even in Mercator y (the same convention as the score PNGs),
so one image quad with the four `bounds` corners covers the map exactly.

```kotlin
const val R = 6378137.0
fun mercX(lon: Double) = R * Math.toRadians(lon)
fun mercY(lat: Double) = R * ln(tan(PI / 4 + Math.toRadians(lat) / 2))
// lon/lat -> fractional pixel; pixel (r, c) = (floor(row), floor(col)); centres at +0.5
fun col(lon: Double) = (mercX(lon) - x0) / px
fun row(lat: Double) = (yTop - mercY(lat)) / px
// pixel centre -> lon/lat
fun lonOf(c: Int) = Math.toDegrees((x0 + (c + 0.5) * px) / R)
fun latOf(r: Int) = Math.toDegrees(2 * atan(exp((yTop - (r + 0.5) * px) / R)) - PI / 2)
```

Tolerance for row/col: 0.01 px (vector `geometry.json`).

## 3. Weather: requests

Source: Open-Meteo, daily variables only, **no `hourly` block**. All requests use
`timezone=Europe/Ljubljana`, so every value is a local calendar day (00:00-24:00).

| model name | Open-Meteo variable | unit |
|---|---|---|
| tmean | `temperature_2m_mean` | degC |
| tmax | `temperature_2m_max` | degC |
| tmin | `temperature_2m_min` | degC |
| precip | `precipitation_sum` | mm / day |

`daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum`

### 3.1 Coarse point list

`model.json` `weather.lattice`: 120 points = the ERA5 0.25 degree lattice, `lat0 = 45.25`, `lon0 = 13.25`,
`step = 0.25`, `n_lat = 8` (45.25 .. 47.0), `n_lon = 15` (13.25 .. 16.75). `points[k] = [lat, lon, elevation_m]`,
**latitude ascending (outer), longitude ascending (inner)**: `k = i_lat * 15 + i_lon`.
`elevation_m` is the mean DEM height of the 0.25 degree box around the point in whole metres.

### 3.2 Forecast + recent days (default; today and the next days)

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude=<lat,lat,...>&longitude=<lon,lon,...>&elevation=<m,m,...>
    &daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum
    &timezone=Europe/Ljubljana&past_days=62&forecast_days=7
```

* Do not pass `models` (default best match), as production does.
* Batches of 50 points: 3 GETs (points 0-49, 50-99, 100-119). `latitude`, `longitude`, `elevation` are
  comma-separated lists of equal length. `model.json` `weather.forecast_recent.requests` holds the three
  exact query maps (the production client's output without `hourly`); `parity/weather_full.json` repeats them.
* Result window: 69 days, `today - 62 .. today + 6` (`today` = Europe/Ljubljana date). The app shows
  `today - 2 .. today + 6` (9 days).

### 3.3 History (a past day older than the forecast window; needs 60 days before it)

```
GET https://archive-api.open-meteo.com/v1/archive
    ?latitude=...&longitude=...&elevation=...&daily=<same four>&timezone=Europe/Ljubljana
    &start_date=<D_first - 60 days>&end_date=<D_last>&models=era5
```

`models=era5` is the 0.25 degree ERA5 reanalysis the trigger was trained on. Because the lattice points
are exactly the ERA5 grid points, the nearest grid cell is the point itself. The archive lags real time by
a few days: use 3.2 for the last week. **This archive request is derived from the production client but
could not be tested live from the build environment (api hosts are blocked there).**

### 3.4 Response

Several points: a JSON **array** of location objects in request order (one object if one point).

```json
[{"latitude":45.25,"longitude":13.25,"elevation":0.0,"timezone":"Europe/Ljubljana",
  "daily":{"time":["2025-07-29","..."],"temperature_2m_mean":[18.4,null],"temperature_2m_max":[...],
           "temperature_2m_min":[...],"precipitation_sum":[...]}}, ...]
```

* Match locations to lattice points **by position**, not by echoed coordinates.
* `elevation` of the response = the elevation the temperatures are valid for = **orography** of that
  point (we passed `elevation_m`; if the field is missing use `elevation_m`).
* HTTP 200 with `{"error":true,"reason":...}` is an error; 429 = rate limit (wait ~60 s); retry 5xx with
  back-off. Cache the result (e.g. one refresh per 3 hours and after midnight).

## 4. Daily series and gap filling

For each of the four variables and each point (`T` days):

1. Days = contiguous calendar days from the earliest to the latest `time` returned by any point;
   missing day or `null` -> NaN.
2. NaN **inside** the series -> linear interpolation in the day index between the neighbouring valid days.
   NaN **after** the last valid value -> that value (carry forward). NaN **before** the first valid value ->
   the first valid value (carry backward). Applies to precipitation too.
3. A point and variable with no valid value at all -> copy the series of the nearest point that has data
   (Euclidean distance in lattice index space `(i_lat, i_lon)`; for equal distances the production
   code (scipy) picks an implementation-defined one, so avoid relying on ties; in practice use any).
4. The orography grid `orog[i][j]` is filled from the responses; all must be present.

(vector `weather_small.json` checks 1-3, including a point with no data, 3-day interior gaps, leading and trailing gaps.)

## 5. Coarse features (per point, per day)

Parameters (`model.json` `trigger.params`, tuned on 267 finds, 2010-2026):

| name | value | unit | meaning |
|---|---|---|---|
| `t_opt` | 17.609 | degC | optimum 20-day mean temperature |
| `t_sd` | 6.835 | degC | width of the temperature response |
| `t_window` | 20 | days | temperature averaging window |
| `p_window` | 26 | days | rain accumulation window |
| `p_half` | 81.12 | mm | 26-day rain with 50 % rain response |
| `pulse_min` | 10.464 | mm / 3 days | rain that counts as a soaking rain |
| `pulse_lag` | [14, 19] | days | a flush appears 14..19 days after a soaking rain |
| `heat_tmax` | 28.80 | degC | 10-day maximum above which the trigger drops |
| `heat_window` | 10 | days | |
| `frost_tmin` | -0.687 | degC | 7-day minimum below which the trigger drops |
| `frost_window` | 7 | days | |

**Windows are trailing and include the current day**: `window(t, w) = days t-w+1 .. t`. At the start of the
series the window is **truncated to the days that exist** (means divide by the number of days present;
sum / max / min over the days present); there is never NaN. With the 62 past days this only matters for
the first ~26 days of the series, which are never displayed.

```
t20[t]    = mean(tmean[t-t_window+1 .. t])
p26[t]    = sum(precip[t-p_window+1 .. t])
tmax10[t] = max(tmax[t-heat_window+1 .. t])
tmin7[t]  = min(tmin[t-frost_window+1 .. t])
# pulse: 1 if a soaking rain ended between lo and hi days ago
rain3[t]  = sum(precip[t-2 .. t])                      # truncated at the start
soak[t]   = if (rain3[t] >= pulse_min) 1 else 0
lo, hi    = pulse_lag
recent[t] = max(soak[t-(hi-lo) .. t])                  # window of hi-lo+1 days, truncated at the start
pulse[t]  = if (t >= lo) recent[t - lo] else 0
```

(So `pulse[t] = max(soak[t-hi .. t-lo])` with truncation at the series start.) Features are computed on the
**unshifted** Open-Meteo series (the elevation shift comes in section 6). Vector `weather_small.json`.

## 6. Downscaling to a pixel

Per pixel with centre `(lon, lat)` and elevation `elev` (from the raster); `lapse = -0.0065` K/m:

```
fi = clamp((lat - lat0) / step, 0, n_lat - 1);   fj = clamp((lon - lon0) / step, 0, n_lon - 1)
i0 = min(floor(fi), n_lat - 2);                  j0 = min(floor(fj), n_lon - 2)
di = fi - i0;  dj = fj - j0
bil(F) = F[i0][j0]*(1-di)*(1-dj) + F[i0][j0+1]*(1-di)*dj + F[i0+1][j0]*di*(1-dj) + F[i0+1][j0+1]*di*dj
dz     = lapse * (elev - bil(orog))
t20_px = bil(t20) + dz;  tmax10_px = bil(tmax10) + dz;  tmin7_px = bil(tmin7) + dz
p26_px = bil(p26);       pulse_px  = bil(pulse)
```

Outside the lattice the nearest edge value is used (clamp, no extrapolation; Slovenia is inside). The
lapse shift is the same for every day in a window, so shifting the features equals shifting the daily series.

**The features are interpolated, then the trigger is evaluated per pixel.** `pulse` is a 0/1 flag at the
coarse points and becomes a fraction 0..1 between points (this is what production does; do not threshold).

## 7. Trigger and score

```
temp    = exp(-0.5 * ((t20 - t_opt) / t_sd)^2)
rain    = 0.6 * p26 / (p26 + p_half) + 0.4 * pulse
heat    = clamp(1 - (tmax10 - heat_tmax) / 5, 0.2, 1.0)
frost   = clamp((tmin7 - frost_tmin) / 3 + 1, 0.0, 1.0)
trigger = clamp(temp * rain * heat * frost, 0, 1)

season  = season_factor[month(day)]            # local calendar day; table in model.json score.season_factor
raw     = habitat * season * trigger
score   = clamp(raw / reference, 0, 1)         # reference = 0.75773, a FIXED constant (not per-day)
```

Season factor: Jan-Mar 0, Apr 0.05, May 0.25, Jun 0.55, Jul 0.6, Aug 0.85, Sep 1.0, Oct 0.9, Nov 0.35, Dec 0.
Pixels outside the mask have no score (transparent, never a hotspot). Vector `cells.json`.

## 8. Colours

`model.json` `colormap`: legend (6 stops), `min_score 0.04`, `alpha_full_at 0.40`, `alpha_max 0.92`, 64 levels.
**Use `colormap.lut` (64 x RGBA) directly**; it is the exact result of the rule:

```
level = clamp(rint(score * 63), 0, 63)          # rint = round half to even (Math.rint), not Math.round
s = level / 63
rgb   = per channel linear interpolation of the legend stops at s, rounded half to even
alpha = if (s < 0.04) 0 else clamp((s - 0.04) / (0.40 - 0.04), 0, 1) * 0.92;   a255 = rint(alpha * 255)
```

Score NaN / outside the mask -> fully transparent. Bitmap = `nx * ny` ARGB, placed as one image quad with
the corners (minLon, maxLat), (maxLon, maxLat), (maxLon, minLat), (minLon, minLat). Vector `misc.json`.

## 9. Hotspots

Inputs: `score`, `habitat`, `trigger`, `elev` of the pixels of one day.

```
rank = score + 0.001 * habitat * trigger          # Float32 arithmetic in production; ties among clipped scores
candidates = pixels with mask == 1 and rank >= 0.15
sort by rank descending; ties by ascending row, then ascending column (stable sort of the row-major scan)
for c in sorted order: if c suppressed -> skip;  pick c;  stop at 300 picks;
    suppress every candidate d (including c) with sqrt((row_d-row_c)^2 + (col_d-col_c)^2) < 6.0   # strictly less
```

Output per pick (order = pick order, descending rank): pixel-centre lon/lat, `score`, `habitat`, `trigger`
(0..1, 3 decimals), `elev` (rounded metres). The threshold is applied to `rank`, not to `score`.
Vector `misc.json` `hotspots`.

## 10. Numeric tolerances

Production runs the bilinear step in float32 and the rest in float64; the vectors were generated from it.
A Kotlin Double implementation matches to: daily series 1e-5, coarse features 1e-4, pixel features 1e-3,
trigger 1e-4, score 2e-4, pixel row/col 0.01 (`model.json` `tolerances`). A Float implementation should still
be inside these except possibly the score (use Double for the windowed sums).

## 11. Parity vectors (`android/app/src/test/resources/parity/`)

| file | checks |
|---|---|
| `weather_small.json` | Open-Meteo-style JSON for a 2 x 3 lattice block, 69 days, with nulls and two points without data. Expected: daily series after gap filling, orography, the five coarse features, trigger. Section 3.4, 4, 5, 7 |
| `weather_full.json` | the same style for all 120 points, plus the 3 request query maps (section 3.2) |
| `cells.json` | 20 pixels (17 real, 3 synthetic edge cases) x 2 days (`2025-09-30` Sep, `2025-10-02` Oct): downscaled features, trigger, raw, score from `weather_full.json`. Section 6, 7 |
| `geometry.json` | lon/lat -> row/col and pixel -> lon/lat (section 2) |
| `misc.json` | score scaling, colour LUT / colours, hotspot selection on a 28 x 36 grid (sections 7-9) |

`today` of the weather vectors is `2025-09-29` (62 past days + today + 6).

## 12. Porting notes and ambiguities

* **Pulse non-linearity**: threshold at the coarse point, then interpolate (section 6). Thresholding after
  interpolation would give different maps.
* **Trailing windows with truncation** at the series start (not NaN, not padded with zeros).
* **Gap filling** touches precipitation too (an interpolated rain value instead of 0).
* **Rounding**: `rint` (half to even) for the colour level, RGB and alpha; use the LUT.
* **Pixel centres** are used for everything (lon/lat of the pixel, nearest-neighbour is not used for weather).
* `season_factor` uses the month of the day being scored, so a forecast across a month boundary changes factor.
* Habitat nodata inside the mask does not occur; treat as 0 anyway.
* Open-Meteo forecast data (best match, NWP) differ from the ERA5 the trigger was tuned on (rain amounts in
  particular); production already did the same. Not quantified here (no network).
* Open-Meteo calls (`forecast`, `archive`) were not tested live from the build environment; check the first live
  response against the section 3.4 schema and the vector files.
* The habitat raster is a resampling of the 250 m EPSG:3794 model grid (bilinear, Slovenian land cells only,
  quantised to 1/250). Compared with production on the same ERA5 weather (3 days, all land pixels): mean
  absolute score difference 0.0003-0.0011, 99th percentile 0.003-0.006, correlation > 0.9999
  (`model.json` `validation`).

## 13. What the Kotlin engine must implement

1. Load `model.json` and the three gzip rasters; geometry functions (section 2).
2. Request builder for the forecast (3 batches) and history queries; parse the array response (section 3).
3. Daily series assembly and gap filling (section 4).
4. Five coarse features per point and day with truncated trailing windows, incl. the pulse steps (section 5).
5. Per-pixel bilinear interpolation, orography/lapse shift and trigger (sections 6-7); score with the month's season factor.
6. Colour LUT bitmap and the single image quad (section 8).
7. Hotspot selection (section 9) and the tap sheet (score, habitat, trigger, elev).
8. `stats.json` screen and `finds.json` markers (`p_trigger`, `p_season_trigger` are held-out temporal percentiles, 0.5 = chance).
9. JVM unit tests loading every parity file.

## 14. Rebuilding the package

```
source backend/.venv/bin/activate
python -m jurcki.app_package           # about 1 minute (re-runs the trigger backtest, deterministic)
pytest -q backend/tests/test_app_package.py
```

Rebuild after changing the habitat raster, `trigger_params.json` or `score_reference.json`
(`python -m jurcki.predict --build-reference` first); the builder refuses to run if the reference was
computed with other trigger parameters.
