@file:OptIn(ExperimentalMaterial3Api::class)

package si.jurcki.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color as AndroidColor
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.location.Location
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SelectableDates
import androidx.compose.material3.Slider
import androidx.compose.material3.SmallFloatingActionButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.delay
import org.maplibre.android.MapLibre
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.geometry.LatLngQuad
import org.maplibre.android.location.LocationComponentActivationOptions
import org.maplibre.android.location.modes.CameraMode
import org.maplibre.android.location.modes.RenderMode
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.expressions.Expression
import org.maplibre.android.style.layers.CircleLayer
import org.maplibre.android.style.layers.Property
import org.maplibre.android.style.layers.PropertyFactory.circleColor
import org.maplibre.android.style.layers.PropertyFactory.circleOpacity
import org.maplibre.android.style.layers.PropertyFactory.circleRadius
import org.maplibre.android.style.layers.PropertyFactory.circleStrokeColor
import org.maplibre.android.style.layers.PropertyFactory.circleStrokeWidth
import org.maplibre.android.style.layers.PropertyFactory.iconAllowOverlap
import org.maplibre.android.style.layers.PropertyFactory.iconIgnorePlacement
import org.maplibre.android.style.layers.PropertyFactory.iconImage
import org.maplibre.android.style.layers.PropertyFactory.iconSize
import org.maplibre.android.style.layers.PropertyFactory.rasterOpacity
import org.maplibre.android.style.layers.PropertyFactory.visibility
import org.maplibre.android.style.layers.RasterLayer
import org.maplibre.android.style.layers.SymbolLayer
import org.maplibre.android.style.sources.GeoJsonSource
import org.maplibre.android.style.sources.ImageSource
import org.maplibre.geojson.Feature
import org.maplibre.geojson.Point
import si.jurcki.app.engine.Find
import si.jurcki.app.engine.HistoryPlan
import si.jurcki.app.engine.Hotspot
import si.jurcki.app.engine.LegendStop
import si.jurcki.app.engine.topPercent
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle
import java.util.Locale
import kotlin.math.roundToInt

private const val TAG = "JurckiMap"

// OpenFreeMap vector style: free, no API key.
private const val STYLE_STREETS = "https://tiles.openfreemap.org/styles/liberty"

// OpenTopoMap raster tiles wrapped in a minimal style (attribution is required by their terms).
private const val STYLE_TOPO_JSON = """{
  "version": 8,
  "sources": {
    "topo": {
      "type": "raster",
      "tiles": ["https://tile.opentopomap.org/{z}/{x}/{y}.png"],
      "tileSize": 256,
      "maxzoom": 17,
      "attribution": "© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)"
    }
  },
  "layers": [ { "id": "topo", "type": "raster", "source": "topo" } ]
}"""

private const val STYLE_PLAIN_JSON = """{
  "version": 8,
  "sources": {},
  "layers": [ { "id": "bg", "type": "background", "paint": { "background-color": "#E8EBE0" } } ]
}"""

private const val SRC_SCORE = "jurcki-score-src"
private const val LYR_SCORE = "jurcki-score"
private const val SRC_HOT = "jurcki-hotspots-src"
private const val LYR_HOT = "jurcki-hotspots"
private const val SRC_NEAR = "jurcki-finds-near-src"
private const val LYR_NEAR = "jurcki-finds-near"
private const val SRC_ALL = "jurcki-finds-all-src"
private const val LYR_ALL = "jurcki-finds-all"
private const val IMG_FIND = "jurcki-find-icon"
private const val EMPTY_FC = """{"type":"FeatureCollection","features":[]}"""

private val SLOVENIA_CENTER = LatLng(46.12, 14.82)
private const val SLOVENIA_ZOOM = 8.0

// ---------------------------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------------------------

@SuppressLint("MissingPermission")
@Composable
fun MapScreen(state: UiState, vm: MainViewModel, onOpenSettings: () -> Unit, onOpenStats: () -> Unit) {
    val context = LocalContext.current
    val density = LocalDensity.current.density
    val mapView = rememberMapView()

    var map by remember { mutableStateOf<MapLibreMap?>(null) }
    var style by remember { mutableStateOf<Style?>(null) }
    var panelOpen by rememberSaveable { mutableStateOf(true) }
    var showPicker by rememberSaveable { mutableStateOf(false) }

    var locationGranted by remember { mutableStateOf(hasLocationPermission(context)) }
    var locationActive by remember { mutableStateOf(false) }
    var locateRequest by remember { mutableIntStateOf(0) }
    var handledLocate by remember { mutableIntStateOf(0) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        val ok = result.values.any { it }
        locationGranted = ok
        if (ok) {
            locateRequest++
        } else {
            Toast.makeText(context, R.string.location_permission_denied, Toast.LENGTH_SHORT).show()
        }
    }

    // --- map creation ------------------------------------------------------------------------
    LaunchedEffect(mapView) {
        mapView.getMapAsync { m ->
            m.setCameraPosition(
                CameraPosition.Builder().target(SLOVENIA_CENTER).zoom(SLOVENIA_ZOOM).build(),
            )
            m.uiSettings.setRotateGesturesEnabled(false)
            m.uiSettings.setTiltGesturesEnabled(false)
            m.uiSettings.setCompassEnabled(false)
            m.addOnMapClickListener { latLng -> handleTap(m, latLng, density, vm) }
            map = m
        }
    }

    // --- base style (re-applied when the user toggles streets/terrain) -------------------------
    val baseMap = state.settings.baseMap
    LaunchedEffect(map, baseMap) {
        val m = map ?: return@LaunchedEffect
        style = null
        locationActive = false
        val builder = when (baseMap) {
            BaseMap.STREETS -> Style.Builder().fromUri(STYLE_STREETS)
            BaseMap.TOPO -> Style.Builder().fromJson(STYLE_TOPO_JSON)
        }
        // Offline and the style is not cached: fall back to a plain background so that the
        // score overlay, hotspots and finds still show up.
        var fellBack = false
        val failListener = MapView.OnDidFailLoadingMapListener { message ->
            if (!fellBack && style == null) {
                fellBack = true
                Log.w(TAG, "Base style failed to load ($message), using plain background")
                m.setStyle(Style.Builder().fromJson(STYLE_PLAIN_JSON)) { loaded -> style = loaded }
            }
        }
        mapView.addOnDidFailLoadingMapListener(failListener)
        try {
            m.setStyle(builder) { loaded -> style = loaded }
            awaitCancellation()
        } finally {
            mapView.removeOnDidFailLoadingMapListener(failListener)
        }
    }

    // --- overlays: every effect (re)creates the layers if the style was replaced ---------------
    val bounds = state.bounds

    LaunchedEffect(style, state.overlay, bounds) {
        val st = style ?: return@LaunchedEffect
        guarded {
            ensureLayers(st, density)
            applyScore(st, state.overlay, bounds, state.settings.opacity, state.showScore)
        }
    }
    LaunchedEffect(style, state.settings.opacity) {
        val st = style ?: return@LaunchedEffect
        guarded { st.getLayer(LYR_SCORE)?.setProperties(rasterOpacity(state.settings.opacity)) }
    }
    LaunchedEffect(style, state.hotspots, state.colorMap) {
        val st = style ?: return@LaunchedEffect
        val cm = state.colorMap ?: return@LaunchedEffect
        guarded {
            ensureLayers(st, density)
            val src: GeoJsonSource? = st.getSourceAs(SRC_HOT)
            src?.setGeoJson(hotspotsToGeoJson(state.hotspots, cm))
        }
    }
    LaunchedEffect(style, state.nearFinds) {
        val st = style ?: return@LaunchedEffect
        guarded {
            ensureLayers(st, density)
            val src: GeoJsonSource? = st.getSourceAs(SRC_NEAR)
            src?.setGeoJson(findsToGeoJson(state.nearFinds))
        }
    }
    LaunchedEffect(style, state.allFinds) {
        val st = style ?: return@LaunchedEffect
        guarded {
            ensureLayers(st, density)
            val src: GeoJsonSource? = st.getSourceAs(SRC_ALL)
            src?.setGeoJson(findsToGeoJson(state.allFinds))
        }
    }
    LaunchedEffect(style, state.showScore, state.showHotspots, state.showNearFinds, state.showAllFinds) {
        val st = style ?: return@LaunchedEffect
        guarded {
            ensureLayers(st, density)
            setVisible(st, LYR_SCORE, state.showScore)
            setVisible(st, LYR_HOT, state.showHotspots)
            setVisible(st, LYR_NEAR, state.showNearFinds)
            setVisible(st, LYR_ALL, state.showAllFinds)
        }
    }

    // --- location (blue dot via LocationComponent, plain Android LocationManager engine) ------
    LaunchedEffect(style, locationGranted) {
        val m = map
        val st = style
        if (m == null || st == null || !locationGranted) return@LaunchedEffect
        try {
            val lc = m.locationComponent
            lc.activateLocationComponent(
                LocationComponentActivationOptions.builder(context, st)
                    .useDefaultLocationEngine(true)
                    .build(),
            )
            lc.setLocationComponentEnabled(true)
            lc.setCameraMode(CameraMode.NONE)
            lc.setRenderMode(RenderMode.NORMAL)
            locationActive = true
        } catch (e: Exception) {
            Log.w(TAG, "Could not activate location component", e)
        }
    }
    LaunchedEffect(locateRequest, locationActive) {
        if (locateRequest <= handledLocate || !locationActive) return@LaunchedEffect
        val m = map ?: return@LaunchedEffect
        handledLocate = locateRequest
        var loc: Location? = m.locationComponent.lastKnownLocation
        if (loc == null) {
            Toast.makeText(context, R.string.location_searching, Toast.LENGTH_SHORT).show()
            var tries = 0
            while (loc == null && tries < 40) {
                delay(500)
                loc = m.locationComponent.lastKnownLocation
                tries++
            }
        }
        if (loc != null) {
            m.animateCamera(
                CameraUpdateFactory.newLatLngZoom(LatLng(loc.latitude, loc.longitude), 13.0),
            )
        } else {
            Toast.makeText(context, R.string.location_unavailable, Toast.LENGTH_SHORT).show()
        }
    }

    val onLocate = {
        if (locationGranted) {
            locateRequest++
        } else {
            permissionLauncher.launch(
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
            )
        }
    }

    // --- UI ------------------------------------------------------------------------------------
    Box(Modifier.fillMaxSize()) {
        AndroidView(factory = { mapView }, modifier = Modifier.fillMaxSize())

        // Top: day chips + stats + settings
        Row(
            Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                modifier = Modifier.weight(1f),
                shape = RoundedCornerShape(28.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.94f),
                shadowElevation = 3.dp,
            ) {
                LazyRow(
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    items(state.days) { chip ->
                        FilterChip(
                            selected = chip.date == state.selectedDay,
                            onClick = { vm.selectDay(chip.date) },
                            label = { Text(dayLabel(chip.date, state.mode) + if (chip.forecast) " *" else "") },
                        )
                    }
                }
            }
            SmallFloatingActionButton(onClick = onOpenStats) {
                Icon(Icons.Default.Info, contentDescription = stringResource(R.string.action_stats))
            }
            SmallFloatingActionButton(onClick = onOpenSettings) {
                Icon(Icons.Default.Settings, contentDescription = stringResource(R.string.action_settings))
            }
        }

        // Right: refresh + my location
        Column(
            Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            SmallFloatingActionButton(onClick = { vm.refresh() }) {
                Icon(Icons.Default.Refresh, contentDescription = stringResource(R.string.action_refresh))
            }
            SmallFloatingActionButton(onClick = { onLocate() }) {
                Icon(Icons.Default.LocationOn, contentDescription = stringResource(R.string.action_my_location))
            }
        }

        // Bottom: attribution + control panel
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(8.dp),
        ) {
            Text(
                text = stringResource(
                    if (baseMap == BaseMap.TOPO) R.string.attribution_topo else R.string.attribution_streets,
                ),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier
                    .align(Alignment.End)
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.7f), RoundedCornerShape(4.dp))
                    .padding(horizontal = 4.dp, vertical = 1.dp),
            )
            Spacer(Modifier.height(4.dp))
            ControlPanel(
                state = state,
                vm = vm,
                open = panelOpen,
                onToggleOpen = { panelOpen = !panelOpen },
                onPickDate = { showPicker = true },
            )
        }

        if (state.loading || state.computing) {
            LinearProgressIndicator(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .fillMaxWidth(),
            )
        }
    }

    if (showPicker) {
        HistoryDatePickerDialog(
            initial = runCatching { LocalDate.parse(state.historyDate) }.getOrDefault(HistoryPlan.today()),
            onPick = { d ->
                showPicker = false
                vm.setHistoryDate(d)
            },
            onDismiss = { showPicker = false },
        )
    }

    // --- bottom sheet for tapped hotspot / find ------------------------------------------------
    state.selection?.let { selection ->
        ModalBottomSheet(
            onDismissRequest = { vm.clearSelection() },
            sheetState = rememberModalBottomSheetState(),
        ) {
            when (selection) {
                is Selection.OfHotspot -> HotspotSheet(selection.hotspot, state)
                is Selection.OfFind -> FindSheet(selection.find)
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------------------------

@Composable
private fun ControlPanel(
    state: UiState,
    vm: MainViewModel,
    open: Boolean,
    onToggleOpen: () -> Unit,
    onPickDate: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.95f),
        shadowElevation = 4.dp,
    ) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    val progress = state.progress
                    if (progress != null) {
                        Text(progressText(progress), style = MaterialTheme.typography.labelMedium)
                    } else if (state.computing) {
                        Text(stringResource(R.string.progress_computing), style = MaterialTheme.typography.labelMedium)
                    }
                    val err = state.error
                    if (err != null) {
                        Text(
                            errorText(err),
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                    val note = state.note
                    if (note != null && progress == null) {
                        Text(
                            noteText(note),
                            style = MaterialTheme.typography.labelMedium,
                            color = if (note.stale) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    val chip = state.days.firstOrNull { it.date == state.selectedDay }
                    if (chip != null && chip.forecast) {
                        Text(
                            stringResource(R.string.day_is_forecast),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    val season = state.season
                    val mean = state.meanScore
                    if (season != null && mean != null && progress == null) {
                        Text(
                            stringResource(R.string.day_stats, season, mean),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                if (state.error != null && state.ready && !state.loading) {
                    TextButton(onClick = { vm.refresh() }) { Text(stringResource(R.string.action_retry)) }
                }
                IconButton(onClick = onToggleOpen) {
                    Icon(
                        imageVector = if (open) Icons.Default.KeyboardArrowDown else Icons.Default.KeyboardArrowUp,
                        contentDescription = stringResource(if (open) R.string.action_collapse else R.string.action_expand),
                    )
                }
            }

            // Mode: forecast / history
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                FilterChip(
                    selected = state.mode == Mode.FORECAST,
                    onClick = { vm.setMode(Mode.FORECAST) },
                    label = { Text(stringResource(R.string.mode_forecast)) },
                )
                FilterChip(
                    selected = state.mode == Mode.HISTORY,
                    onClick = { vm.setMode(Mode.HISTORY) },
                    label = { Text(stringResource(R.string.mode_history)) },
                )
            }

            if (state.mode == Mode.HISTORY) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    IconButton(onClick = { vm.shiftHistoryDate(-7) }) { Text("‹‹") }
                    IconButton(onClick = { vm.shiftHistoryDate(-1) }) { Text("‹") }
                    OutlinedButton(onClick = onPickDate, modifier = Modifier.weight(1f)) {
                        Text(formatDate(state.historyDate))
                    }
                    IconButton(onClick = { vm.shiftHistoryDate(1) }) { Text("›") }
                    IconButton(onClick = { vm.shiftHistoryDate(7) }) { Text("››") }
                }
            }

            if (open) {
                Text(
                    stringResource(R.string.opacity_label, (state.settings.opacity * 100).roundToInt()),
                    style = MaterialTheme.typography.labelMedium,
                )
                Slider(
                    value = state.settings.opacity,
                    onValueChange = { vm.setOpacity(it) },
                    onValueChangeFinished = { vm.commitOpacity() },
                    valueRange = 0.1f..1f,
                )
                Row(
                    Modifier.horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    FilterChip(
                        selected = state.showScore,
                        onClick = { vm.setShowScore(!state.showScore) },
                        label = { Text(stringResource(R.string.layer_forecast)) },
                    )
                    FilterChip(
                        selected = state.showHotspots,
                        onClick = { vm.setShowHotspots(!state.showHotspots) },
                        label = { Text(stringResource(R.string.layer_hotspots)) },
                    )
                    if (state.mode == Mode.HISTORY) {
                        FilterChip(
                            selected = state.showNearFinds,
                            onClick = { vm.setShowNearFinds(!state.showNearFinds) },
                            label = { Text(stringResource(R.string.layer_finds_near, state.nearFinds.size)) },
                        )
                    }
                    FilterChip(
                        selected = state.showAllFinds,
                        onClick = { vm.setShowAllFinds(!state.showAllFinds) },
                        label = { Text(stringResource(R.string.layer_finds_all)) },
                    )
                    Spacer(Modifier.width(8.dp))
                    FilterChip(
                        selected = state.settings.baseMap == BaseMap.STREETS,
                        onClick = { vm.setBaseMap(BaseMap.STREETS) },
                        label = { Text(stringResource(R.string.base_map_streets)) },
                    )
                    FilterChip(
                        selected = state.settings.baseMap == BaseMap.TOPO,
                        onClick = { vm.setBaseMap(BaseMap.TOPO) },
                        label = { Text(stringResource(R.string.base_map_topo)) },
                    )
                }
                if (state.legend.isNotEmpty()) {
                    Spacer(Modifier.height(8.dp))
                    Legend(state.legend)
                }
            }
        }
    }
}

@Composable
private fun Legend(legend: List<LegendStop>) {
    val min = legend.first().value
    val max = legend.last().value
    val range = (max - min).takeIf { it > 0.0 } ?: 1.0
    val stops = legend
        .map { (((it.value - min) / range).toFloat()) to Color(it.argb) }
        .toTypedArray()
    Column(Modifier.fillMaxWidth()) {
        Text(stringResource(R.string.legend_title), style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.height(2.dp))
        Box(
            Modifier
                .fillMaxWidth()
                .height(12.dp)
                .clip(RoundedCornerShape(6.dp))
                .background(Brush.horizontalGradient(*stops)),
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                "${stringResource(R.string.legend_low)} (${"%.1f".format(Locale.US, min)})",
                style = MaterialTheme.typography.labelSmall,
            )
            Text(
                "${stringResource(R.string.legend_high)} (${"%.1f".format(Locale.US, max)})",
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
private fun HistoryDatePickerDialog(initial: LocalDate, onPick: (LocalDate) -> Unit, onDismiss: () -> Unit) {
    val today = remember { HistoryPlan.today() }
    val startMs = remember(initial) { initial.atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli() }
    val pickerState = rememberDatePickerState(
        initialSelectedDateMillis = startMs,
        yearRange = HistoryPlan.MIN_DATE.year..today.year,
        selectableDates = object : SelectableDates {
            override fun isSelectableDate(utcTimeMillis: Long): Boolean {
                val d = Instant.ofEpochMilli(utcTimeMillis).atZone(ZoneOffset.UTC).toLocalDate()
                return !d.isBefore(HistoryPlan.MIN_DATE) && !d.isAfter(today)
            }

            override fun isSelectableYear(year: Int): Boolean = year >= HistoryPlan.MIN_DATE.year && year <= today.year
        },
    )
    DatePickerDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(
                onClick = {
                    val ms = pickerState.selectedDateMillis
                    if (ms != null) {
                        onPick(Instant.ofEpochMilli(ms).atZone(ZoneOffset.UTC).toLocalDate())
                    } else {
                        onDismiss()
                    }
                },
            ) { Text(stringResource(R.string.action_ok)) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text(stringResource(R.string.action_cancel)) }
        },
    ) {
        DatePicker(state = pickerState)
    }
}

@Composable
private fun HotspotSheet(h: Hotspot, state: UiState) {
    val context = LocalContext.current
    val cm = state.colorMap
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp)
            .padding(bottom = 16.dp)
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            val argb = cm?.argb(h.score) ?: 0
            Box(
                Modifier
                    .size(18.dp)
                    .background(Color(argb or (0xFF shl 24)), CircleShape),
            )
            Spacer(Modifier.width(10.dp))
            Text(stringResource(R.string.hotspot_title), style = MaterialTheme.typography.headlineSmall)
        }
        state.selectedDay?.let { InfoRow(stringResource(R.string.field_date), formatDate(it)) }
        InfoRow(stringResource(R.string.field_score), "%.2f".format(Locale.US, h.score))
        InfoRow(stringResource(R.string.field_habitat), "%.2f".format(Locale.US, h.habitat))
        InfoRow(stringResource(R.string.field_trigger), "%.2f".format(Locale.US, h.trigger))
        InfoRow(stringResource(R.string.field_elevation), stringResource(R.string.unit_meters, h.elev.roundToInt()))
        InfoRow(stringResource(R.string.field_coordinates), coords(h.lat, h.lon))
        Text(
            stringResource(R.string.hotspot_explain),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(4.dp))
        Button(onClick = { navigateTo(context, h.lat, h.lon) }) {
            Text(stringResource(R.string.action_navigate))
        }
    }
}

@Composable
private fun FindSheet(f: Find) {
    val context = LocalContext.current
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp)
            .padding(bottom = 16.dp)
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(stringResource(R.string.find_title), style = MaterialTheme.typography.headlineSmall)
        InfoRow(stringResource(R.string.field_date), formatDate(f.date))
        if (f.species.isNotEmpty()) InfoRow(stringResource(R.string.field_species), f.species)
        f.source?.let { InfoRow(stringResource(R.string.field_source), it) }
        f.country?.let { InfoRow(stringResource(R.string.field_country), it) }
        f.precisionM?.let {
            InfoRow(stringResource(R.string.field_precision), "~" + stringResource(R.string.unit_meters, it))
        }
        InfoRow(stringResource(R.string.field_coordinates), coords(f.lat, f.lon))

        HorizontalDivider(Modifier.padding(vertical = 4.dp))
        Text(stringResource(R.string.find_backtest_title), style = MaterialTheme.typography.titleSmall)
        f.habitat?.let {
            InfoRow(stringResource(R.string.field_habitat), "%.2f".format(Locale.US, it))
            Text(
                stringResource(R.string.find_habitat_explain),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        val ps = f.pSeasonTrigger
        val pt = f.pTrigger
        if (ps != null) {
            Text(
                stringResource(R.string.find_backtest_season, topPercent(ps)),
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        if (pt != null) {
            Text(
                stringResource(R.string.find_backtest_weather, topPercent(pt)),
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        if (ps == null && pt == null) {
            Text(
                stringResource(R.string.find_not_in_backtest),
                style = MaterialTheme.typography.bodyMedium,
            )
        } else {
            Text(
                stringResource(R.string.find_backtest_note),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(Modifier.height(4.dp))
        Button(onClick = { navigateTo(context, f.lat, f.lon) }) {
            Text(stringResource(R.string.action_navigate))
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.width(12.dp))
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun progressText(p: Progress): String = when (p.kind) {
    ProgressKind.LOADING_MODEL -> stringResource(R.string.progress_model)
    ProgressKind.WEATHER -> stringResource(R.string.progress_weather, minOf(p.done + 1, p.total), p.total)
    ProgressKind.WAIT_RATE_LIMIT -> stringResource(R.string.progress_wait, p.waitSeconds)
    ProgressKind.COMPUTING -> stringResource(R.string.progress_computing)
}

@Composable
private fun errorText(e: AppError): String {
    val detail = e.detail ?: ""
    return when (e.kind) {
        ErrorKind.NO_NETWORK -> stringResource(R.string.error_no_network) + if (detail.isNotEmpty()) " ($detail)" else ""
        ErrorKind.RATE_LIMIT -> stringResource(R.string.error_rate_limit)
        ErrorKind.API -> stringResource(R.string.error_api, detail)
        ErrorKind.BAD_RESPONSE -> stringResource(R.string.error_bad_response, detail)
        ErrorKind.PACKAGE -> stringResource(R.string.error_package, detail)
        ErrorKind.OTHER -> stringResource(R.string.error_other, detail)
    }
}

@Composable
private fun noteText(n: DataNote): String {
    val time = formatDateTime(n.fetchedAtMillis)
    return when {
        n.stale -> stringResource(R.string.note_stale, time)
        n.source == DataSource.ERA5 -> stringResource(R.string.note_era5)
        n.source == DataSource.FORECAST_FOR_HISTORY -> stringResource(R.string.note_forecast_for_history, time)
        else -> stringResource(R.string.note_forecast, time)
    }
}

@Composable
private fun dayLabel(day: String, mode: Mode): String {
    val date = try {
        LocalDate.parse(day)
    } catch (e: Exception) {
        return day
    }
    val today = LocalDate.now()
    return when {
        mode == Mode.FORECAST && date == today -> stringResource(R.string.day_today)
        mode == Mode.FORECAST && date == today.plusDays(1) -> stringResource(R.string.day_tomorrow)
        else -> date.format(DateTimeFormatter.ofPattern("EEE d. M.", Locale.getDefault()))
    }
}

// ---------------------------------------------------------------------------------------------
// MapView lifecycle
// ---------------------------------------------------------------------------------------------

@Composable
private fun rememberMapView(): MapView {
    val context = LocalContext.current
    val mapView = remember {
        MapLibre.getInstance(context)
        MapView(context)
    }
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    DisposableEffect(lifecycle, mapView) {
        var destroyed = false
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_CREATE -> mapView.onCreate(null)
                Lifecycle.Event.ON_START -> mapView.onStart()
                Lifecycle.Event.ON_RESUME -> mapView.onResume()
                Lifecycle.Event.ON_PAUSE -> mapView.onPause()
                Lifecycle.Event.ON_STOP -> mapView.onStop()
                Lifecycle.Event.ON_DESTROY -> {
                    destroyed = true
                    mapView.onDestroy()
                }
                else -> Unit
            }
        }
        lifecycle.addObserver(observer)
        onDispose {
            lifecycle.removeObserver(observer)
            if (!destroyed) {
                val current = lifecycle.currentState
                if (current.isAtLeast(Lifecycle.State.RESUMED)) mapView.onPause()
                if (current.isAtLeast(Lifecycle.State.STARTED)) mapView.onStop()
                mapView.onDestroy()
            }
        }
    }
    return mapView
}

// ---------------------------------------------------------------------------------------------
// Map helpers (no Compose below this line)
// ---------------------------------------------------------------------------------------------

private inline fun guarded(block: () -> Unit) {
    try {
        block()
    } catch (e: Exception) {
        Log.w(TAG, "Map update failed", e)
    }
}

/** Creates the hotspot and find sources/layers if the (new) style does not have them yet. */
private fun ensureLayers(style: Style, density: Float) {
    if (style.getSource(SRC_ALL) == null) {
        // All finds of any year: small translucent dots (habitat context), hidden by default.
        style.addSource(GeoJsonSource(SRC_ALL, EMPTY_FC))
        style.addLayer(
            CircleLayer(LYR_ALL, SRC_ALL).withProperties(
                circleRadius(
                    Expression.interpolate(
                        Expression.linear(),
                        Expression.zoom(),
                        Expression.stop(6, 2f),
                        Expression.stop(10, 4f),
                        Expression.stop(14, 6f),
                    ),
                ),
                circleColor("#6A1B9A"),
                circleOpacity(0.6f),
                circleStrokeWidth(0.8f),
                circleStrokeColor("#FFFFFF"),
                visibility(Property.NONE),
            ),
        )
    }
    if (style.getSource(SRC_HOT) == null) {
        style.addSource(GeoJsonSource(SRC_HOT, EMPTY_FC))
        style.addLayer(
            CircleLayer(LYR_HOT, SRC_HOT).withProperties(
                circleRadius(
                    Expression.interpolate(
                        Expression.linear(),
                        Expression.zoom(),
                        Expression.stop(6, 4f),
                        Expression.stop(10, 8f),
                        Expression.stop(14, 14f),
                    ),
                ),
                circleColor(Expression.toColor(Expression.get("color"))),
                circleOpacity(0.9f),
                circleStrokeWidth(1.5f),
                circleStrokeColor("#FFFFFF"),
            ),
        )
    }
    if (style.getSource(SRC_NEAR) == null) {
        // Finds within +-7 days of the selected day (history mode): purple diamonds.
        style.addImage(IMG_FIND, createFindIcon(density))
        style.addSource(GeoJsonSource(SRC_NEAR, EMPTY_FC))
        style.addLayer(
            SymbolLayer(LYR_NEAR, SRC_NEAR).withProperties(
                iconImage(IMG_FIND),
                iconSize(1f),
                iconAllowOverlap(true),
                iconIgnorePlacement(true),
            ),
        )
    }
}

/** Adds/updates the score bitmap as an ImageSource placed by the 4 corners of `bounds`. */
private fun applyScore(style: Style, bitmap: Bitmap?, bounds: Bounds?, opacity: Float, visible: Boolean) {
    if (bitmap == null || bounds == null) {
        style.removeLayer(LYR_SCORE)
        style.removeSource(SRC_SCORE)
        return
    }
    val quad = LatLngQuad(
        LatLng(bounds.maxLat, bounds.minLon), // top-left
        LatLng(bounds.maxLat, bounds.maxLon), // top-right
        LatLng(bounds.minLat, bounds.maxLon), // bottom-right
        LatLng(bounds.minLat, bounds.minLon), // bottom-left
    )
    val existing: ImageSource? = style.getSourceAs(SRC_SCORE)
    if (existing != null) {
        existing.setCoordinates(quad)
        existing.setImage(bitmap)
        return
    }
    style.addSource(ImageSource(SRC_SCORE, quad, bitmap))
    val layer = RasterLayer(LYR_SCORE, SRC_SCORE).withProperties(
        rasterOpacity(opacity),
        visibility(if (visible) Property.VISIBLE else Property.NONE),
    )
    // Keep place names readable: insert the heat map below the first label layer of the base style.
    val firstLabel = style.layers.firstOrNull { it is SymbolLayer && it.id != LYR_NEAR }?.id
    style.addLayerBelow(layer, firstLabel ?: LYR_ALL)
}

private fun setVisible(style: Style, layerId: String, visible: Boolean) {
    style.getLayer(layerId)?.setProperties(visibility(if (visible) Property.VISIBLE else Property.NONE))
}

/** Purple diamond with white outline, so finds look different from the round hotspot circles. */
private fun createFindIcon(density: Float): Bitmap {
    val size = (26 * density).roundToInt()
    val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bmp)
    val half = size / 2f
    val inset = 2f * density
    val path = Path().apply {
        moveTo(half, inset)
        lineTo(size - inset, half)
        lineTo(half, size - inset)
        lineTo(inset, half)
        close()
    }
    val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    fill.color = 0xFF6A1B9A.toInt()
    fill.style = Paint.Style.FILL
    val stroke = Paint(Paint.ANTI_ALIAS_FLAG)
    stroke.color = AndroidColor.WHITE
    stroke.style = Paint.Style.STROKE
    stroke.strokeWidth = 2f * density
    stroke.strokeJoin = Paint.Join.ROUND
    canvas.drawPath(path, fill)
    canvas.drawPath(path, stroke)
    return bmp
}

/** Picks the find/hotspot nearest to the tap within a finger-sized rectangle. */
private fun handleTap(map: MapLibreMap, tap: LatLng, density: Float, vm: MainViewModel): Boolean {
    val pt = map.projection.toScreenLocation(tap)
    val r = 24f * density
    val rect = RectF(pt.x - r, pt.y - r, pt.x + r, pt.y + r)

    val nearIdx = nearestIdx(map.queryRenderedFeatures(rect, LYR_NEAR), tap)
    if (nearIdx != null) {
        vm.selectNearFind(nearIdx)
        return true
    }
    val hotspotIdx = nearestIdx(map.queryRenderedFeatures(rect, LYR_HOT), tap)
    if (hotspotIdx != null) {
        vm.selectHotspot(hotspotIdx)
        return true
    }
    val allIdx = nearestIdx(map.queryRenderedFeatures(rect, LYR_ALL), tap)
    if (allIdx != null) {
        vm.selectAnyFind(allIdx)
        return true
    }
    return false
}

private fun nearestIdx(features: List<Feature>, tap: LatLng): Int? {
    var best: Int? = null
    var bestDist = Double.MAX_VALUE
    for (f in features) {
        val idx = f.getNumberProperty("idx")?.toInt() ?: continue
        val p = f.geometry() as? Point
        val d = if (p == null) 0.0 else {
            val dLat = p.latitude() - tap.latitude
            val dLon = (p.longitude() - tap.longitude) * Math.cos(Math.toRadians(tap.latitude))
            dLat * dLat + dLon * dLon
        }
        if (d < bestDist) {
            bestDist = d
            best = idx
        }
    }
    return best
}

private fun hasLocationPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

private fun coords(lat: Double, lon: Double): String = String.format(Locale.US, "%.5f, %.5f", lat, lon)

/** Opens any installed navigation/maps app via a geo: intent. */
private fun navigateTo(context: Context, lat: Double, lon: Double) {
    val uri = Uri.parse(String.format(Locale.US, "geo:%.6f,%.6f?q=%.6f,%.6f", lat, lon, lat, lon))
    try {
        context.startActivity(Intent(Intent.ACTION_VIEW, uri))
    } catch (e: ActivityNotFoundException) {
        Toast.makeText(context, R.string.no_navigation_app, Toast.LENGTH_SHORT).show()
    }
}

private fun formatDate(iso: String): String =
    try {
        LocalDate.parse(iso).format(DateTimeFormatter.ofLocalizedDate(FormatStyle.MEDIUM))
    } catch (e: Exception) {
        iso
    }

private fun formatDateTime(millis: Long): String =
    try {
        Instant.ofEpochMilli(millis).atZone(ZoneId.systemDefault())
            .format(DateTimeFormatter.ofLocalizedDateTime(FormatStyle.SHORT))
    } catch (e: Exception) {
        "-"
    }
