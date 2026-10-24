# Jurčki – Android viewer

Private Kotlin / Jetpack Compose app that shows the predicted jurček (*Boletus*) hot spots for
Slovenia on a MapLibre map. It only *reads* the backend output described in
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) ("App data contract"): `meta.json`,
`score_YYYY-MM-DD.png`, `hotspots_YYYY-MM-DD.geojson`, `reports.geojson`.

* Package `si.jurcki.app`, minSdk 26, target/compileSdk 36
* UI language: Slovenian (default, `res/values/`) and English (`res/values-en/`)

## Features

* Full-screen map centred on Slovenia (46.12 N, 14.82 E, zoom 8)
  * Base map: OpenFreeMap "liberty" (default, no API key) or OpenTopoMap terrain (raster, attribution shown)
* Score PNG as a MapLibre `ImageSource` (corners from `meta.json` `bounds`) + `RasterLayer` with an opacity slider
* Day chips for every day in `meta.json` (`days`)
* Hotspots: circles coloured by score (legend from `meta.json`); tap -> bottom sheet with score, habitat, trigger,
  elevation and a "Navigate" button (`geo:` intent)
* Reports (finds): purple diamond markers; tap -> date / source / precision / text
* "My location" button (runtime permission, blue dot via MapLibre `LocationComponent`, plain `LocationManager`)
* Legend gradient, built from `meta.json` `legend`
* Settings: data base URL and optional GitHub token (`Authorization: Bearer <token>`), stored in DataStore
* Files are cached on disk (`filesDir/data-cache/<hash of base URL>/`); if the network fails the last cache is shown
  (status line turns red: "Brez povezave" / "Offline"). All days are prefetched after a refresh.

## Build

### GitHub Actions (recommended)

`.github/workflows/android.yml` builds `assembleDebug` on every push that touches `android/**`
(and on manual *Run workflow*). Download the `jurcki-debug-apk` artifact from the run page and
install it (`adb install -r app-debug.apk` or copy it to the phone and allow "unknown sources").

### Locally

Requirements: JDK 17 and the Android SDK (platform 36, build-tools 35.0.0). Either open `android/`
in Android Studio, or:

```bash
cd android
export ANDROID_HOME=$HOME/Android/Sdk        # or write sdk.dir=... to local.properties
./gradlew assembleDebug                      # -> app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The Gradle wrapper pins Gradle 8.14.3 (`gradle/wrapper/`). Dependency versions are in
`gradle/libs.versions.toml`.

## Run with the sample data

`data/output/sample/` holds a synthetic dataset that follows the contract (4 days from 2026-10-24,
5 reports). Regenerate it with `python data/output/sample/make_sample.py`.

```bash
cd data/output/sample && python3 -m http.server 8000 --bind 0.0.0.0
```

In the app: Settings -> base URL `http://10.0.2.2:8000/` (Android emulator -> host machine) or
`http://<LAN-IP>:8000/` for a real phone on the same Wi-Fi; token empty -> "Shrani in osveži".
Plain `http://` only works in **debug** builds (`app/src/debug/AndroidManifest.xml` enables cleartext).
The sample days start at 2026-10-24; the app selects today's date if it is listed, otherwise the
first day after today, otherwise the last day.

## Real data from the private GitHub repo

Default base URL: `https://raw.githubusercontent.com/DavidCufer/Jurcki/data/output/` (a placeholder – the backend
must publish `data/output/*` to that branch/path, or change the URL in Settings). Because the repository is
private, create a *fine-grained personal access token* (Repository access: `Jurcki` only; permission
*Contents: Read-only*) and paste it into Settings. It is stored unencrypted in the app's private DataStore and sent
only as `Authorization: Bearer <token>` to the configured URL. `raw.githubusercontent.com` caches responses for a
few minutes, so a refresh right after a publish can still return the old file.

## Code layout (`app/src/main/java/si/jurcki/app/`)

| File | Purpose |
| --- | --- |
| `MainActivity.kt` | Entry point, theme, switches between map and settings |
| `MapScreen.kt` | MapLibre `MapView` in Compose, layers, day chips, panel, legend, bottom sheets, location |
| `SettingsScreen.kt` | Base URL + token form |
| `MainViewModel.kt` | UI state, refresh / day loading / offline logic (no DI framework) |
| `DataRepository.kt` | OkHttp downloads + on-disk cache |
| `SettingsStore.kt` | DataStore wrapper |
| `Models.kt` | Data classes, `org.json` parsing, legend colour interpolation, GeoJSON for the map |

## Notes / limitations

* Map library: `org.maplibre.gl:android-sdk-opengl` (OpenGL ES build). The plain `android-sdk` artifact of the 13.x
  line is the Vulkan build and declares `uses-feature vulkan required`.
* MapLibre caches viewed tiles itself. If the base style cannot be loaded (offline, nothing cached) the app falls back
  to a plain background so the forecast, hotspots and finds still show; toggle Ulice/Teren or restart to retry.
* The score PNG is placed with MapLibre's Web-Mercator `ImageSource` quad. A PNG sampled on a regular lat/lon grid is
  therefore stretched slightly (< 1 km across Slovenia) – irrelevant for a 250 m heat map, but the backend could
  also export the PNG in EPSG:3857 row spacing.
* Dependency versions could not be resolved from Google Maven in the authoring environment (blocked); they were
  taken from the official release-notes pages and Maven Central. If CI complains about a missing artifact, adjust
  `gradle/libs.versions.toml`. AGP 9.x needs Gradle 9.6+ and a different Kotlin setup, so the project stays on AGP 8.13.
