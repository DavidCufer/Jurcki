package si.jurcki.app

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import si.jurcki.app.engine.ColorMap
import si.jurcki.app.engine.Engine
import si.jurcki.app.engine.Find
import si.jurcki.app.engine.FindsParser
import si.jurcki.app.engine.HistoryPlan
import si.jurcki.app.engine.Hotspot
import si.jurcki.app.engine.LegendStop
import si.jurcki.app.engine.ModelPackage
import si.jurcki.app.engine.WeatherBundle
import si.jurcki.app.engine.WeatherClient
import si.jurcki.app.engine.WeatherException
import si.jurcki.app.engine.WeatherField
import si.jurcki.app.engine.WeatherProgress
import si.jurcki.app.engine.WeatherService
import si.jurcki.app.engine.WeatherSource
import si.jurcki.app.engine.argbToBitmap
import java.io.File
import java.time.LocalDate

data class UiState(
    val settings: AppSettings = AppSettings(),
    val mode: Mode = Mode.FORECAST,
    /** The model package (rasters + parameters) is loaded. */
    val ready: Boolean = false,
    val modelCreated: String = "",
    val bounds: Bounds? = null,
    val colorMap: ColorMap? = null,
    val legend: List<LegendStop> = emptyList(),
    val days: List<DayChip> = emptyList(),
    val selectedDay: String? = null,
    val overlay: Bitmap? = null,
    val hotspots: List<Hotspot> = emptyList(),
    /** Season factor and mean score (land pixels) of the selected day. */
    val season: Double? = null,
    val meanScore: Double? = null,
    /** Chosen centre date of history mode (ISO). */
    val historyDate: String = HistoryPlan.today().minusYears(1).toString(),
    val loading: Boolean = false,
    val computing: Boolean = false,
    val progress: Progress? = null,
    val error: AppError? = null,
    val note: DataNote? = null,
    val allFinds: List<Find> = emptyList(),
    /** Finds within +-7 days of the selected day (history mode only). */
    val nearFinds: List<Find> = emptyList(),
    val showScore: Boolean = true,
    val showHotspots: Boolean = true,
    val showNearFinds: Boolean = true,
    val showAllFinds: Boolean = false,
    val selection: Selection? = null,
    val stats: StatsData? = null,
)

/** Computed map of one day, ready to display. */
private class DayView(
    val bitmap: Bitmap,
    val hotspots: List<Hotspot>,
    val season: Double,
    val meanScore: Double,
)

/** One weather download with its coarse features and the days computed so far. */
private class Session(
    val field: WeatherField,
    val chips: List<DayChip>,
    val note: DataNote,
    val loadedAtMillis: Long,
    val loadedOn: LocalDate = HistoryPlan.today(),
) {
    val views = HashMap<String, DayView>()

    /** Day that was shown last in this session (restored when switching back to its mode). */
    var selected: String? = null
}

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val store = SettingsStore(app)

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var engine: Engine? = null
    private var service: WeatherService? = null
    private var forecastSession: Session? = null
    private var historySession: Session? = null

    private var loadJob: Job? = null
    private var dayJob: Job? = null
    private var shiftJob: Job? = null
    private var loadGen = 0
    private var dayGen = 0

    init {
        viewModelScope.launch {
            val settings = store.flow.first()
            _state.update { it.copy(settings = settings, loading = true, progress = Progress(ProgressKind.LOADING_MODEL)) }
            try {
                val loaded = withContext(Dispatchers.Default) { loadPackage() }
                engine = loaded.engine
                service = WeatherService(
                    WeatherClient(WeatherClient.defaultHttpClient(), File(app.cacheDir, "weather")),
                    loaded.engine.pkg,
                )
                val g = loaded.engine.pkg.geo
                _state.update {
                    it.copy(
                        ready = true,
                        modelCreated = loaded.engine.pkg.createdAt,
                        bounds = Bounds(g.minLon, g.minLat, g.maxLon, g.maxLat),
                        colorMap = loaded.engine.pkg.colorMap,
                        legend = loaded.engine.pkg.colorMap.legend,
                        allFinds = loaded.finds,
                        stats = loaded.stats,
                        loading = false,
                        progress = null,
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _state.update {
                    it.copy(loading = false, progress = null, error = AppError(ErrorKind.PACKAGE, e.message ?: e.javaClass.simpleName))
                }
                return@launch
            }
            loadForecast(force = false)
        }
    }

    private class Loaded(val engine: Engine, val finds: List<Find>, val stats: StatsData?)

    private fun loadPackage(): Loaded {
        val assets = getApplication<Application>().assets
        val pkg = ModelPackage.load { name -> assets.open("model/$name") }
        val finds = assets.open("model/finds.json").use { FindsParser.parse(String(it.readBytes(), Charsets.UTF_8)) }
        val stats = try {
            assets.open("model/stats.json").use { StatsData.parse(String(it.readBytes(), Charsets.UTF_8)) }
        } catch (e: Exception) {
            null // the stats screen is optional
        }
        return Loaded(Engine(pkg), finds, stats)
    }

    // ---- loading -------------------------------------------------------------------------------

    /** Forced reload of the current mode (forecast: new download). */
    fun refresh() {
        if (!_state.value.ready || _state.value.loading) return
        when (_state.value.mode) {
            Mode.FORECAST -> loadForecast(force = true)
            Mode.HISTORY -> loadHistory(LocalDate.parse(_state.value.historyDate))
        }
    }

    /** Called when the app comes to the foreground: reload the forecast if it is older than 3 h or from another day. */
    fun onForeground() {
        val s = forecastSession ?: return
        if (!_state.value.ready || _state.value.mode != Mode.FORECAST || _state.value.loading) return
        val age = System.currentTimeMillis() - s.loadedAtMillis
        val dayChanged = s.loadedOn != HistoryPlan.today()
        if (age > WeatherClient.FORECAST_MAX_AGE_MS || dayChanged) loadForecast(force = false)
    }

    private fun loadForecast(force: Boolean) {
        val eng = engine ?: return
        val svc = service ?: return
        startLoad {
            val bundle = withContext(Dispatchers.IO) { svc.forecast(force) { p -> setWeatherProgress(p) } }
            setProgress(Progress(ProgressKind.COMPUTING))
            val today = HistoryPlan.today()
            val session = withContext(Dispatchers.Default) {
                val field = eng.field(bundle.weather)
                val dates = HistoryPlan.forecastDates(field.dates, today, eng.pkg.shownPast, eng.pkg.shownAhead)
                if (dates.isEmpty()) throw WeatherException(WeatherException.Kind.BAD_RESPONSE, "No days to show")
                val chips = dates.map { DayChip(it, LocalDate.parse(it).isAfter(today)) }
                Session(field, chips, noteOf(bundle, Mode.FORECAST), System.currentTimeMillis())
            }
            forecastSession = session
            if (_state.value.mode == Mode.FORECAST) {
                val keep = _state.value.selectedDay
                val preferred = if (keep != null && session.chips.any { it.date == keep }) keep else today.toString()
                showSession(session, preferred)
            }
        }
    }

    private fun loadHistory(center: LocalDate) {
        val eng = engine ?: return
        val svc = service ?: return
        startLoad {
            val today = HistoryPlan.today()
            val (first, last) = HistoryPlan.window(center, today)
            val bundle = withContext(Dispatchers.IO) { svc.history(first, last, today) { p -> setWeatherProgress(p) } }
            setProgress(Progress(ProgressKind.COMPUTING))
            val session = withContext(Dispatchers.Default) {
                val field = eng.field(bundle.weather)
                val dates = HistoryPlan.historyDates(field.dates, first, last)
                if (dates.isEmpty()) throw WeatherException(WeatherException.Kind.BAD_RESPONSE, "No days to show")
                val chips = dates.map { DayChip(it, LocalDate.parse(it).isAfter(today)) }
                Session(field, chips, noteOf(bundle, Mode.HISTORY), System.currentTimeMillis())
            }
            historySession = session
            if (_state.value.mode == Mode.HISTORY) {
                val preferred = if (session.chips.any { it.date == center.toString() }) center.toString() else session.chips.last().date
                showSession(session, preferred)
            }
        }
    }

    private fun noteOf(b: WeatherBundle, mode: Mode): DataNote {
        val source = when {
            b.source == WeatherSource.ERA5 -> DataSource.ERA5
            mode == Mode.HISTORY -> DataSource.FORECAST_FOR_HISTORY
            else -> DataSource.FORECAST
        }
        return DataNote(source, b.fetchedAtMillis, b.stale, b.fromCache)
    }

    /** Runs [block] as the (single) loading job; errors become [UiState.error], old data stays visible. */
    private fun startLoad(block: suspend () -> Unit) {
        loadJob?.cancel()
        dayJob?.cancel()
        val gen = ++loadGen
        _state.update { it.copy(loading = true, computing = false, error = null, progress = Progress(ProgressKind.WEATHER, 0, 3)) }
        loadJob = viewModelScope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (e: WeatherException) {
                if (gen == loadGen) _state.update { it.copy(error = AppError(errorKindOf(e.kind), e.message)) }
            } catch (e: Exception) {
                if (gen == loadGen) {
                    _state.update { it.copy(error = AppError(ErrorKind.OTHER, e.message ?: e.javaClass.simpleName)) }
                }
            } finally {
                if (gen == loadGen) _state.update { it.copy(loading = false, progress = null) }
            }
        }
    }

    private fun errorKindOf(k: WeatherException.Kind): ErrorKind = when (k) {
        WeatherException.Kind.NO_NETWORK -> ErrorKind.NO_NETWORK
        WeatherException.Kind.RATE_LIMIT -> ErrorKind.RATE_LIMIT
        WeatherException.Kind.API -> ErrorKind.API
        WeatherException.Kind.BAD_RESPONSE -> ErrorKind.BAD_RESPONSE
    }

    private fun setWeatherProgress(p: WeatherProgress) {
        setProgress(
            if (p.waitSeconds > 0) Progress(ProgressKind.WAIT_RATE_LIMIT, waitSeconds = p.waitSeconds)
            else Progress(ProgressKind.WEATHER, p.done, p.total),
        )
    }

    private fun setProgress(p: Progress) = _state.update { it.copy(progress = p) }

    // ---- showing a day --------------------------------------------------------------------------

    private suspend fun showSession(s: Session, preferred: String) {
        _state.update {
            it.copy(
                days = s.chips,
                note = s.note,
                selectedDay = preferred,
                selection = null,
            )
        }
        displayDay(s, preferred)
    }

    /** Computes (or takes from the session cache) the map of [date] and publishes it. */
    private suspend fun displayDay(s: Session, date: String) {
        s.selected = date
        val v: DayView = s.views[date] ?: computeView(s, date)
        _state.update {
            if (it.selectedDay != date) {
                it
            } else {
                it.copy(
                    overlay = v.bitmap,
                    hotspots = v.hotspots,
                    season = v.season,
                    meanScore = v.meanScore,
                    nearFinds = nearFindsFor(it.mode, date, it.allFinds),
                    selection = null,
                )
            }
        }
    }

    private suspend fun computeView(s: Session, date: String): DayView {
        val eng = engine ?: throw IllegalStateException("model not loaded")
        _state.update { it.copy(computing = true) }
        try {
            val v = withContext(Dispatchers.Default) {
                val m = eng.dayMap(s.field, s.field.indexOf(date))
                val g = eng.pkg.geo
                DayView(argbToBitmap(m.argb, g.nx, g.ny), m.hotspots, m.season, m.meanScore)
            }
            s.views[date] = v
            return v
        } finally {
            _state.update { it.copy(computing = false) }
        }
    }

    private fun nearFindsFor(mode: Mode, date: String, all: List<Find>): List<Find> =
        if (mode == Mode.HISTORY) FindsParser.near(all, LocalDate.parse(date), 7) else emptyList()

    fun selectDay(date: String) {
        val s = currentSession() ?: return
        if (s.chips.none { it.date == date }) return
        dayJob?.cancel()
        val gen = ++dayGen
        _state.update { it.copy(selectedDay = date, selection = null) }
        dayJob = viewModelScope.launch {
            try {
                displayDay(s, date)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                if (gen == dayGen) {
                    _state.update { it.copy(error = AppError(ErrorKind.OTHER, e.message ?: e.javaClass.simpleName)) }
                }
            }
        }
    }

    private fun currentSession(): Session? =
        if (_state.value.mode == Mode.FORECAST) forecastSession else historySession

    // ---- mode and history date --------------------------------------------------------------------

    fun setMode(mode: Mode) {
        if (!_state.value.ready || mode == _state.value.mode) return
        loadJob?.cancel()
        dayJob?.cancel()
        loadGen++
        val existing = if (mode == Mode.FORECAST) forecastSession else historySession
        _state.update {
            val base = it.copy(
                mode = mode, selection = null, error = null, loading = false, computing = false, progress = null,
                nearFinds = emptyList(),
            )
            if (existing == null) base.withoutDisplay() else base
        }
        if (existing != null) {
            val wanted = if (mode == Mode.HISTORY) _state.value.historyDate else HistoryPlan.today().toString()
            val preferred = existing.selected?.takeIf { d -> existing.chips.any { it.date == d } }
                ?: wanted.takeIf { d -> existing.chips.any { it.date == d } }
                ?: existing.chips.last().date
            dayJob = viewModelScope.launch {
                try {
                    showSession(existing, preferred)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    _state.update { it.copy(error = AppError(ErrorKind.OTHER, e.message ?: e.javaClass.simpleName)) }
                }
            }
            if (mode == Mode.FORECAST) onForeground()
        } else if (mode == Mode.FORECAST) {
            loadForecast(force = false)
        } else {
            loadHistory(LocalDate.parse(_state.value.historyDate))
        }
    }

    private fun UiState.withoutDisplay(): UiState = copy(
        days = emptyList(), selectedDay = null, overlay = null, hotspots = emptyList(),
        season = null, meanScore = null, nearFinds = emptyList(), note = null,
    )

    /** History mode: show [date] +- 3 days (clamped to 2010-01-01 .. today). */
    fun setHistoryDate(date: LocalDate) {
        shiftJob?.cancel()
        val d = clampHistoryDate(date)
        _state.update { it.copy(historyDate = d.toString(), mode = Mode.HISTORY).withoutDisplay() }
        historySession = null
        loadHistory(d)
    }

    /** Date-step buttons: the date label changes at once, the map loads when the taps stop. */
    fun shiftHistoryDate(days: Long) {
        val d = clampHistoryDate(LocalDate.parse(_state.value.historyDate).plusDays(days))
        _state.update { it.copy(historyDate = d.toString()) }
        shiftJob?.cancel()
        shiftJob = viewModelScope.launch {
            delay(600)
            setHistoryDate(d)
        }
    }

    private fun clampHistoryDate(date: LocalDate): LocalDate {
        val today = HistoryPlan.today()
        return when {
            date.isBefore(HistoryPlan.MIN_DATE) -> HistoryPlan.MIN_DATE
            date.isAfter(today) -> today
            else -> date
        }
    }

    // ---- settings and toggles ------------------------------------------------------------------------

    fun setOpacity(value: Float) {
        _state.update { it.copy(settings = it.settings.copy(opacity = value)) }
    }

    fun commitOpacity() {
        val v = _state.value.settings.opacity
        viewModelScope.launch { store.setOpacity(v) }
    }

    fun setBaseMap(value: BaseMap) {
        if (value == _state.value.settings.baseMap) return
        _state.update { it.copy(settings = it.settings.copy(baseMap = value)) }
        viewModelScope.launch { store.setBaseMap(value) }
    }

    fun setShowScore(v: Boolean) = _state.update { it.copy(showScore = v) }
    fun setShowHotspots(v: Boolean) = _state.update { it.copy(showHotspots = v, selection = null) }
    fun setShowNearFinds(v: Boolean) = _state.update { it.copy(showNearFinds = v, selection = null) }
    fun setShowAllFinds(v: Boolean) = _state.update { it.copy(showAllFinds = v, selection = null) }

    fun clearWeatherCache() {
        val svc = service ?: return
        viewModelScope.launch { svc.clearCache() }
    }

    // ---- selection ----------------------------------------------------------------------------------

    fun selectHotspot(idx: Int) {
        val h = _state.value.hotspots.getOrNull(idx) ?: return
        _state.update { it.copy(selection = Selection.OfHotspot(h)) }
    }

    fun selectNearFind(idx: Int) {
        val f = _state.value.nearFinds.getOrNull(idx) ?: return
        _state.update { it.copy(selection = Selection.OfFind(f)) }
    }

    fun selectAnyFind(idx: Int) {
        val f = _state.value.allFinds.getOrNull(idx) ?: return
        _state.update { it.copy(selection = Selection.OfFind(f)) }
    }

    fun clearSelection() = _state.update { it.copy(selection = null) }
}
