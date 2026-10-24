# Jurčki – Android app

Private Kotlin / Jetpack Compose app that shows where and when jurčki (*Boletus*) are likely to fruit in
Slovenia, on a MapLibre map. **The phone computes everything itself**: there is no server, no base URL,
no token – the app works right after installation (it only needs internet for the weather).

* Package `si.jurcki.app`, minSdk 26, target/compileSdk 36
* UI language: Slovenian (default, `res/values/`) and English (`res/values-en/`)
* Spec of the engine: [`docs/APP_ENGINE.md`](../docs/APP_ENGINE.md) (the Kotlin code mirrors
  `backend/jurcki/app_engine.py` and is tested against vectors made by the Python lab)

## How it works

```
score(pixel, day) = clamp( habitat(pixel) * season_factor(month(day)) * trigger(pixel, day) / reference, 0, 1 )
```

1. `assets/model/` holds the **model package**: `model.json` (parameters, season table, 120-point weather
   lattice, colour table), three gzip rasters (habitat, elevation, land mask; 1009 x 661 pixels of ~250 m),
   `stats.json` (the "Model / uspešnost" screen) and `finds.json` (dated finds with backtest values).
2. **Forecast mode**: 3 Open-Meteo `forecast` requests (120 points, `past_days=62`, `forecast_days=7`) ->
   daily series -> gap filling -> 5 coarse features per point (20-day mean temperature, 26-day rain, soaking-rain
   pulse, 10-day max, 7-day min) -> per pixel bilinear interpolation + lapse-rate shift with the elevation raster
   -> trigger -> score -> one ARGB bitmap (`ImageSource` quad) + up to 300 hotspots. Days today-2 .. today+6;
   days after today are marked `*` (model data).
3. **History mode** ("Zgodovina"): pick a date 2010-01-01 .. today; the app computes that day +- 3 days from the
   ERA5 reanalysis (`archive-api.open-meteo.com`, `models=era5`, 60 days of lead-in). The last ~week is not in
   the archive yet, so those dates use the forecast API (the panel says so). Finds of `finds.json` within
   +-7 days of the selected day are shown as purple diamonds (tap: date, species, source, precision and the
   backtest values: habitat and the held-out percentile "the day of the find was among the top X % of the season
   days"); "Vse najdbe" shows all finds of all years as small dots for habitat context.
4. Weather responses are cached on disk (`cacheDir/weather/`): forecast responses are reused for 3 hours
   (and not across midnight, Europe/Ljubljana), history (archive) responses are kept for good. If a download
   fails and an old copy exists it is shown (the panel turns red: "Brez povezave"). Clear cache: info screen.
5. All computing runs on `Dispatchers.Default`; per day only the finished bitmap + hotspots are kept.

## Screens

* **Map**: full-screen MapLibre map (OpenFreeMap "liberty" or OpenTopoMap terrain), score overlay with
  opacity slider, day chips, hotspot circles (tap -> sheet with score, habitat, trigger, elevation, navigate
  button), layer chips, legend, "my location".
* **Model / uspešnost** (info button): headline numbers (habitat AUC literature vs model, trigger AUC season only /
  literature / tuned leave-one-year-out, hit rates), plain-language explanation of AUC 0.5 / 0.7 / 1.0, per-year
  table and bars, parameter table with explanations, weekend effect, data counts, glossary. All from `stats.json`.
* **O aplikaciji** (settings button): how it works, data sources, attribution, "clear cached weather".

## Refreshing the model package

The package is produced by the Python lab and committed to the repo:

```bash
source backend/.venv/bin/activate
python -m jurcki.app_package          # ~1 min; writes android/app/src/main/assets/model/* and the parity vectors
pytest -q backend/tests/test_app_package.py
```

Rebuild after changing the habitat raster, `trigger_params.json` or `score_reference.json`. Then run the unit
tests (below): they compare the Kotlin engine with the new vectors.

## Build

### GitHub Actions (recommended)

`.github/workflows/android.yml` runs `./gradlew --no-daemon testDebugUnitTest assembleDebug` on every push that
touches `android/**` (and on manual *Run workflow*). If a unit test fails the report is uploaded as
`unit-test-report`; the APK is still built (`--continue`) and uploaded as `jurcki-debug-apk`. Download it and
install it (`adb install -r app-debug.apk` or copy it to the phone and allow "unknown sources").

### Locally

Requirements: JDK 17 and the Android SDK (platform 36, build-tools 35.0.0). Either open `android/`
in Android Studio, or:

```bash
cd android
export ANDROID_HOME=$HOME/Android/Sdk        # or write sdk.dir=... to local.properties
./gradlew testDebugUnitTest                  # JVM unit tests, no device and no network
./gradlew assembleDebug                      # -> app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The Gradle wrapper pins Gradle 8.14.3 (`gradle/wrapper/`). Dependency versions are in
`gradle/libs.versions.toml`.

## Tests (`app/src/test/java/si/jurcki/app/engine/`)

Plain JVM tests (JUnit 4.13.2, `org.json:json` because Android's `org.json` is a stub in local unit tests).
They read the package from `src/main/assets/model` and the vectors from `src/test/resources/parity/`:

| test | checks |
| --- | --- |
| `EngineParityTest` | geometry, daily assembly + gap filling, coarse features incl. pulse, request builders, downscaling, trigger, score at 17 real + 3 synthetic cells (also through the full-raster loop), score scaling, colour LUT and its rule, hotspot selection |
| `EngineUnitTest` | gap-filling edge cases, truncated windows, pulse, raster decoding, hotspot ties / strict separation, history planning, `finds.json` and `stats.json` parsing |
| `WeatherClientTest` | OkHttp client + disk cache with a fake HTTP layer: caching, forced refresh, stale copy when offline, error mapping, archive vs forecast choice |

## Code layout (`app/src/main/java/si/jurcki/app/`)

| File | Purpose |
| --- | --- |
| `engine/ModelPackage.kt` | `model.json` + gzip rasters -> `ModelPackage`, colour map, lattice, parameters |
| `engine/Geometry.kt` | lon/lat <-> pixel (Web-Mercator aligned raster) |
| `engine/Weather.kt` | Open-Meteo request builder, response parser, daily assembly, gap filling |
| `engine/WeatherClient.kt` | OkHttp download (retry / back-off / 429 wait) + on-disk cache |
| `engine/WeatherService.kt` | forecast vs ERA5 history, day windows (`HistoryPlan`) |
| `engine/Features.kt` | 5 coarse features per point and day, trigger formula |
| `engine/Engine.kt` | downscaling to pixels, trigger, score, `dayMap` (score + colours + hotspots) |
| `engine/Hotspots.kt` | hotspot selection (rank, de-clustering) |
| `engine/Render.kt` | score -> ARGB through the LUT (pure); `BitmapRender.kt` is the only Android part |
| `engine/Finds.kt` | `finds.json` parser, "top X %" helper |
| `StatsData.kt` | `stats.json` parser (sl / en texts) |
| `MainViewModel.kt` | UI state, loading, modes, day cache (no DI framework) |
| `MapScreen.kt` | MapLibre `MapView` in Compose, layers, panel, date picker, bottom sheets, location |
| `StatsScreen.kt` | "Model / uspešnost" screen (cards, tables, Compose bars) |
| `AboutScreen.kt` | info / attribution / clear cache |
| `SettingsStore.kt` | DataStore: overlay opacity and base map |
| `Models.kt` | UI data classes, GeoJSON for the map layers |

## Notes / limitations

* Map library: `org.maplibre.gl:android-sdk-opengl` (OpenGL ES build). The plain `android-sdk` artifact of the 13.x
  line is the Vulkan build and declares `uses-feature vulkan required`.
* MapLibre caches viewed tiles itself. If the base style cannot be loaded (offline, nothing cached) the app falls back
  to a plain background so the score overlay, hotspots and finds still show; toggle Ulice/Teren or restart to retry.
* Open-Meteo is free for non-commercial use without a key but rate-limited, and a request is weighted by locations x
  days (a full refresh is 3 requests of 50 points x 69 days). On HTTP 429 the app waits 60 s (up to twice) and shows
  the countdown; the forecast is only re-downloaded after 3 hours, history ranges are rounded to 14-day blocks so that
  stepping through dates re-uses the cache, and the date-step buttons wait 0.6 s after the last tap.
* The Open-Meteo `forecast` data (best-match NWP) differ somewhat from ERA5, on which the trigger was tuned (rain amounts
  in particular); the history mode therefore shows the "truer" picture than the forecast.
* The archive request (`archive-api.open-meteo.com`, `models=era5`) is derived from the production client and could not be
  tested live from the build environment: check the first live history response against docs/APP_ENGINE.md section 3.4.
* AGP 9.x needs Gradle 9.6+ and a different Kotlin setup, so the project stays on AGP 8.13.
