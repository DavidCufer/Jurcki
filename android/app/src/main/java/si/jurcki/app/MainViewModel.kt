package si.jurcki.app

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.LocalDate

enum class Status { LOADING_FIRST, OK, OFFLINE, NO_DATA }

data class UiState(
    val settings: AppSettings = AppSettings(),
    val meta: Meta? = null,
    val selectedDay: String? = null,
    val overlay: Bitmap? = null,
    val hotspots: List<Hotspot> = emptyList(),
    val reports: List<Report> = emptyList(),
    val loading: Boolean = false,
    val status: Status = Status.LOADING_FIRST,
    /** Short technical detail of the last failure (shown next to the status). */
    val detail: String? = null,
    val dayError: String? = null,
    val selection: Selection? = null,
    val showForecast: Boolean = true,
    val showHotspots: Boolean = true,
    val showReports: Boolean = true,
) {
    val legend: List<LegendStop> get() = meta?.legend ?: DEFAULT_LEGEND
}

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val store = SettingsStore(app)
    private val repo = DataRepository(app)

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var refreshJob: Job? = null
    private var dayJob: Job? = null

    init {
        viewModelScope.launch {
            val settings = store.flow.first()
            _state.update { it.copy(settings = settings) }
            // Show the cached data immediately, then try to update it.
            repo.cachedMeta(settings)?.let { cached ->
                val day = pickDay(cached, null)
                _state.update { it.copy(meta = cached, selectedDay = day, status = Status.OFFLINE) }
                _state.update { it.copy(reports = repo.loadReports(settings, force = false)) }
                loadDayInto(day, force = false)
            }
            refresh()
        }
    }

    // ---- data -------------------------------------------------------------------------------

    fun refresh() {
        refreshJob?.cancel()
        refreshJob = viewModelScope.launch {
            _state.update { it.copy(loading = true) }
            val s = _state.value.settings
            val res = repo.refreshMeta(s)
            val meta = res.value
            if (meta == null) {
                _state.update {
                    it.copy(loading = false, status = Status.NO_DATA, detail = res.error?.let { e -> describe(e) })
                }
                return@launch
            }
            val day = pickDay(meta, _state.value.selectedDay)
            _state.update {
                it.copy(
                    meta = meta,
                    selectedDay = day,
                    status = if (res.fromCache) Status.OFFLINE else Status.OK,
                    detail = res.error?.let { e -> describe(e) },
                )
            }
            _state.update { it.copy(reports = repo.loadReports(s, force = true)) }
            loadDayInto(day, force = true)
            _state.update { it.copy(loading = false) }

            for (d in meta.days) if (d != day) repo.prefetchDay(s, d)
        }
    }

    fun selectDay(day: String) {
        if (day == _state.value.selectedDay) return
        _state.update { it.copy(selectedDay = day, selection = null) }
        dayJob?.cancel()
        dayJob = viewModelScope.launch { loadDayInto(day, force = false) }
    }

    private suspend fun loadDayInto(day: String, force: Boolean) {
        val d = repo.loadDay(_state.value.settings, day, force)
        _state.update {
            if (it.selectedDay != day) it
            else it.copy(overlay = d.overlay, hotspots = d.hotspots, dayError = d.error)
        }
    }

    private fun pickDay(meta: Meta, current: String?): String {
        if (current != null && current in meta.days) return current
        val today = LocalDate.now().toString()
        return meta.days.firstOrNull { it == today }
            ?: meta.days.firstOrNull { it > today }
            ?: meta.days.last()
    }

    private fun describe(e: Throwable): String =
        if (e is HttpStatusException) "HTTP ${e.code}" else (e.message ?: e.javaClass.simpleName)

    // ---- settings ---------------------------------------------------------------------------

    fun saveConnection(baseUrl: String, token: String) {
        val url = DataRepository.normalizeBaseUrl(baseUrl)
        val tok = token.trim()
        _state.update {
            it.copy(
                settings = it.settings.copy(baseUrl = url, token = tok),
                meta = null,
                selectedDay = null,
                overlay = null,
                hotspots = emptyList(),
                reports = emptyList(),
                selection = null,
                status = Status.LOADING_FIRST,
                detail = null,
            )
        }
        viewModelScope.launch {
            store.setConnection(url, tok)
            val settings = _state.value.settings
            // Another base URL may already have a cache: show it while refreshing.
            repo.cachedMeta(settings)?.let { cached ->
                val day = pickDay(cached, null)
                _state.update { it.copy(meta = cached, selectedDay = day, status = Status.OFFLINE) }
                loadDayInto(day, force = false)
            }
            refresh()
        }
    }

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

    fun setShowForecast(v: Boolean) = _state.update { it.copy(showForecast = v) }
    fun setShowHotspots(v: Boolean) = _state.update { it.copy(showHotspots = v, selection = null) }
    fun setShowReports(v: Boolean) = _state.update { it.copy(showReports = v, selection = null) }

    // ---- selection --------------------------------------------------------------------------

    fun selectHotspot(idx: Int) {
        val h = _state.value.hotspots.getOrNull(idx) ?: return
        _state.update { it.copy(selection = Selection.OfHotspot(h)) }
    }

    fun selectReport(idx: Int) {
        val r = _state.value.reports.getOrNull(idx) ?: return
        _state.update { it.copy(selection = Selection.OfReport(r)) }
    }

    fun clearSelection() = _state.update { it.copy(selection = null) }
}
