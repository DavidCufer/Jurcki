package si.jurcki.app.engine

import java.time.LocalDate
import java.time.ZoneId

enum class WeatherSource { FORECAST, ERA5 }

/** Weather of one download, parsed and gap-filled. */
class WeatherBundle(
    val weather: DailyWeather,
    val source: WeatherSource,
    val fromCache: Boolean,
    val stale: Boolean,
    val fetchedAtMillis: Long,
)

/** Which days and which API a history request uses. */
object HistoryPlan {
    /** First date of the archive that the model was tuned on. */
    val MIN_DATE: LocalDate = LocalDate.of(2010, 1, 1)

    /** The archive lags real time by a few days: newer days come from the forecast API (`past_days`). */
    const val ARCHIVE_LAG_DAYS = 7L

    /** Days shown on each side of the chosen date. */
    const val HALF_RANGE = 3L

    fun today(): LocalDate = LocalDate.now(ZoneId.of(OpenMeteo.TIMEZONE))

    /** Displayed days: chosen date +- [HALF_RANGE], never after [today]. */
    fun window(center: LocalDate, today: LocalDate): Pair<LocalDate, LocalDate> {
        val last = minOf(center.plusDays(HALF_RANGE), today)
        val first = minOf(center.minusDays(HALF_RANGE), last)
        return first to last
    }

    fun useArchive(last: LocalDate, today: LocalDate): Boolean = !last.isAfter(today.minusDays(ARCHIVE_LAG_DAYS))

    /** Archive request range for the days [first]..[last]: whole 14-day blocks, never later than the archive lag allows. */
    fun archiveRange(first: LocalDate, last: LocalDate, today: LocalDate, leadInDays: Int): Pair<LocalDate, LocalDate> {
        val block = 14L
        val s = Math.floorDiv(first.toEpochDay(), block) * block
        val e = Math.floorDiv(last.toEpochDay(), block) * block + block - 1
        val start = LocalDate.ofEpochDay(s - leadInDays)
        val end = minOf(LocalDate.ofEpochDay(e), today.minusDays(ARCHIVE_LAG_DAYS))
        return start to end
    }

    /** Dates of [all] (ISO) within today-[past] .. today+[ahead]: the day chips of forecast mode. */
    fun forecastDates(all: List<String>, today: LocalDate, past: Int, ahead: Int): List<String> {
        val from = today.minusDays(past.toLong())
        val to = today.plusDays(ahead.toLong())
        return all.filter {
            val d = LocalDate.parse(it)
            !d.isBefore(from) && !d.isAfter(to)
        }
    }

    /** Dates of [all] (ISO) within [first]..[last]: the day chips of history mode. */
    fun historyDates(all: List<String>, first: LocalDate, last: LocalDate): List<String> =
        all.filter {
            val d = LocalDate.parse(it)
            !d.isBefore(first) && !d.isAfter(last)
        }
}

/** Gets the weather for the engine: forecast + recent days, or the ERA5 history around a date. */
class WeatherService(private val client: WeatherClient, private val pkg: ModelPackage) {

    /** Forecast API: today-62 .. today+6. Refreshes when the cache is older than 3 h or from another day. */
    suspend fun forecast(force: Boolean, onProgress: (WeatherProgress) -> Unit): WeatherBundle {
        val reqs = OpenMeteo.forecastRequests(pkg.lattice, pkg.forecastPastDays, pkg.forecastDays, pkg.batchSize)
        val res = client.fetchAll(reqs, WeatherClient.FORECAST_MAX_AGE_MS, force, onProgress)
        return bundle(res, WeatherSource.FORECAST)
    }

    /**
     * History for the days [first]..[last]: ERA5 archive if the window ends before the archive lag, otherwise the
     * forecast API, which holds the last 62 days. The archive range is rounded to 14-day blocks (and starts
     * `historyDaysNeeded` days before the first block), so stepping from day to day re-uses the cached responses.
     */
    suspend fun history(
        first: LocalDate,
        last: LocalDate,
        today: LocalDate,
        onProgress: (WeatherProgress) -> Unit,
    ): WeatherBundle {
        if (!HistoryPlan.useArchive(last, today)) return forecast(force = false, onProgress = onProgress)
        val (start, end) = HistoryPlan.archiveRange(first, last, today, pkg.historyDaysNeeded)
        val reqs = OpenMeteo.archiveRequests(pkg.lattice, start, end, pkg.batchSize)
        val res = client.fetchAll(reqs, null, false, onProgress)
        return bundle(res, WeatherSource.ERA5)
    }

    private fun bundle(res: FetchResult, source: WeatherSource): WeatherBundle {
        val locations = res.bodies.flatMap { OpenMeteo.parseLocations(it) }
        val lat = pkg.lattice
        val w = OpenMeteo.assembleDaily(locations, lat.nLat, lat.nLon, lat.pointElev)
        return WeatherBundle(w, source, res.fromCache, res.stale, res.fetchedAtMillis)
    }

    suspend fun clearCache() = client.clearCache()
}
