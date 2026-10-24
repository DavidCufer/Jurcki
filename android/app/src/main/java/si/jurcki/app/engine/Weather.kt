package si.jurcki.app.engine

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.time.LocalDate

/** Failure while getting or reading weather data; [kind] tells the UI which message to show. */
class WeatherException(val kind: Kind, message: String, cause: Throwable? = null) : Exception(message, cause) {
    enum class Kind { NO_NETWORK, RATE_LIMIT, API, BAD_RESPONSE }
}

/** One Open-Meteo GET: base [url] plus query [params]. */
data class WeatherRequest(val url: String, val params: Map<String, String>) {
    /** Full URL. Values only contain URL-safe characters (digits, '.', '-', ',', '/', letters). */
    fun fullUrl(): String = url + "?" + params.entries.joinToString("&") { it.key + "=" + it.value }
}

/**
 * Daily series after parsing and gap filling: arrays are `[point][day]`, lattice order
 * (latitude ascending outer, longitude ascending inner). [orog] = orography (m) per point.
 */
class DailyWeather(
    val dates: List<String>,
    val tmean: Array<DoubleArray>,
    val tmax: Array<DoubleArray>,
    val tmin: Array<DoubleArray>,
    val precip: Array<DoubleArray>,
    val orog: DoubleArray,
)

/** Open-Meteo request builder, response parser and daily assembly (docs/APP_ENGINE.md sections 3 and 4). */
object OpenMeteo {
    const val FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    const val ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    const val DAILY = "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum"
    const val TIMEZONE = "Europe/Ljubljana"

    /** Forecast + recent days: 3 GETs of [batchSize] points, `past_days` + `forecast_days`. */
    fun forecastRequests(lattice: Lattice, pastDays: Int, forecastDays: Int, batchSize: Int): List<WeatherRequest> =
        batches(lattice, batchSize).map { (from, to) ->
            val p = LinkedHashMap<String, String>()
            addPoints(p, lattice, from, to)
            p["daily"] = DAILY
            p["timezone"] = TIMEZONE
            p["past_days"] = pastDays.toString()
            p["forecast_days"] = forecastDays.toString()
            WeatherRequest(FORECAST_URL, p)
        }

    /** ERA5 archive for [start]..[end] (inclusive), 3 GETs of [batchSize] points. */
    fun archiveRequests(lattice: Lattice, start: LocalDate, end: LocalDate, batchSize: Int): List<WeatherRequest> =
        batches(lattice, batchSize).map { (from, to) ->
            val p = LinkedHashMap<String, String>()
            addPoints(p, lattice, from, to)
            p["daily"] = DAILY
            p["timezone"] = TIMEZONE
            p["start_date"] = start.toString()
            p["end_date"] = end.toString()
            p["models"] = "era5"
            WeatherRequest(ARCHIVE_URL, p)
        }

    private fun batches(lattice: Lattice, batchSize: Int): List<Pair<Int, Int>> {
        val out = ArrayList<Pair<Int, Int>>()
        var from = 0
        while (from < lattice.size) {
            val to = minOf(from + batchSize, lattice.size)
            out += from to to
            from = to
        }
        return out
    }

    private fun addPoints(p: MutableMap<String, String>, lattice: Lattice, from: Int, to: Int) {
        p["latitude"] = (from until to).joinToString(",") { lattice.pointLat[it].toString() }
        p["longitude"] = (from until to).joinToString(",") { lattice.pointLon[it].toString() }
        p["elevation"] = (from until to).joinToString(",") { Math.round(lattice.pointElev[it]).toString() }
    }

    /**
     * Response body -> list of location objects. A JSON array has one object per requested
     * point (request order), a single object is one location. `{"error":true,"reason":..}` throws.
     */
    fun parseLocations(body: String): List<JSONObject> {
        val t = body.trim()
        try {
            if (t.startsWith("[")) {
                val arr = JSONArray(t)
                val out = ArrayList<JSONObject>(arr.length())
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    checkError(o)
                    out += o
                }
                return out
            }
            val o = JSONObject(t)
            checkError(o)
            return listOf(o)
        } catch (e: JSONException) {
            throw WeatherException(WeatherException.Kind.BAD_RESPONSE, "Unreadable weather response: ${e.message}", e)
        }
    }

    private fun checkError(o: JSONObject) {
        if (o.optBoolean("error", false)) {
            throw WeatherException(WeatherException.Kind.API, o.optString("reason", "unknown error"))
        }
    }

    /**
     * Locations (lattice order) -> gap-free daily series (section 4):
     * contiguous days from the earliest to the latest `time` of any location, null/missing -> NaN,
     * NaN inside a series -> linear, after the last / before the first valid value -> that value,
     * a point without any value -> series of the nearest point that has data (per variable).
     * [fallbackElev] = elevation passed in the request, used if the response has no `elevation`.
     */
    fun assembleDaily(locations: List<JSONObject>, nLat: Int, nLon: Int, fallbackElev: DoubleArray): DailyWeather {
        val n = nLat * nLon
        if (locations.size != n) {
            throw WeatherException(
                WeatherException.Kind.BAD_RESPONSE,
                "${locations.size} locations in the response, expected $n",
            )
        }
        try {
            // epoch day of every `time` entry of every location
            val days = ArrayList<LongArray>(n)
            var minDay = Long.MAX_VALUE
            var maxDay = Long.MIN_VALUE
            for (loc in locations) {
                val t = loc.getJSONObject("daily").getJSONArray("time")
                val d = LongArray(t.length()) { LocalDate.parse(t.getString(it)).toEpochDay() }
                for (v in d) {
                    if (v < minDay) minDay = v
                    if (v > maxDay) maxDay = v
                }
                days += d
            }
            if (minDay > maxDay) {
                throw WeatherException(WeatherException.Kind.BAD_RESPONSE, "The weather response has no days")
            }
            val dayCount = (maxDay - minDay).toInt() + 1
            val first = minDay
            val dates = (0 until dayCount).map { LocalDate.ofEpochDay(first + it).toString() }

            fun variable(name: String): Array<DoubleArray> {
                val series = Array(n) { DoubleArray(dayCount) { Double.NaN } }
                for (k in 0 until n) {
                    val vals = locations[k].getJSONObject("daily").optJSONArray(name) ?: continue
                    val d = days[k]
                    for (i in 0 until minOf(d.size, vals.length())) {
                        if (vals.isNull(i)) continue
                        series[k][(d[i] - first).toInt()] = vals.getDouble(i)
                    }
                }
                val valid = BooleanArray(n)
                for (k in 0 until n) valid[k] = fillGaps(series[k])
                fillMissingPoints(series, valid, nLat, nLon)
                return series
            }

            val tmean = variable("temperature_2m_mean")
            val tmax = variable("temperature_2m_max")
            val tmin = variable("temperature_2m_min")
            val precip = variable("precipitation_sum")

            val orog = DoubleArray(n)
            for (k in 0 until n) {
                val e = if (locations[k].has("elevation") && !locations[k].isNull("elevation")) {
                    locations[k].optDouble("elevation", Double.NaN)
                } else {
                    Double.NaN
                }
                orog[k] = if (e.isNaN()) fallbackElev[k] else e
            }
            return DailyWeather(dates, tmean, tmax, tmin, precip, orog)
        } catch (e: JSONException) {
            throw WeatherException(WeatherException.Kind.BAD_RESPONSE, "Unexpected weather response: ${e.message}", e)
        } catch (e: java.time.format.DateTimeParseException) {
            throw WeatherException(WeatherException.Kind.BAD_RESPONSE, "Unexpected date in the weather response", e)
        }
    }

    /**
     * In place: NaN inside the series -> linear interpolation in the day index, after the last valid
     * value -> that value, before the first valid value -> the first valid value.
     * Returns false (series untouched) if there is no valid value at all.
     */
    fun fillGaps(a: DoubleArray): Boolean {
        var first = -1
        for (t in a.indices) {
            if (!a[t].isNaN()) {
                first = t
                break
            }
        }
        if (first < 0) return false
        for (t in 0 until first) a[t] = a[first]
        var prev = first
        for (t in first + 1 until a.size) {
            if (a[t].isNaN()) continue
            if (t - prev > 1) {
                val span = (t - prev).toDouble()
                val v0 = a[prev]
                val v1 = a[t]
                for (k in prev + 1 until t) a[k] = v0 + (v1 - v0) * ((k - prev) / span)
            }
            prev = t
        }
        for (t in prev + 1 until a.size) a[t] = a[prev]
        return true
    }

    /**
     * Points without any valid value copy the series of the nearest valid point (Euclidean distance
     * in lattice index space; ties: first in lattice order).
     */
    fun fillMissingPoints(series: Array<DoubleArray>, valid: BooleanArray, nLat: Int, nLon: Int) {
        if (valid.all { it }) return
        if (valid.none { it }) {
            throw WeatherException(WeatherException.Kind.BAD_RESPONSE, "The weather response has no valid values")
        }
        for (k in series.indices) {
            if (valid[k]) continue
            val ik = k / nLon
            val jk = k % nLon
            var best = -1
            var bestD = Int.MAX_VALUE
            for (g in series.indices) {
                if (!valid[g]) continue
                val di = g / nLon - ik
                val dj = g % nLon - jk
                val d = di * di + dj * dj
                if (d < bestD) {
                    bestD = d
                    best = g
                }
            }
            series[k] = series[best].copyOf()
        }
    }
}
