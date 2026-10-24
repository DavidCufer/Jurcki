package si.jurcki.app.engine

import org.json.JSONObject
import java.time.LocalDate

/**
 * One dated find of `assets/model/finds.json` with its backtest values.
 * [pTrigger] / [pSeasonTrigger]: held-out temporal percentile of the find day among the other days of the
 * season at the same place (0.5 = chance, 1 = best day); null = the find is not part of the backtest.
 */
data class Find(
    val idx: Int,
    val date: String,
    val epochDay: Long,
    val lon: Double,
    val lat: Double,
    val species: String,
    val country: String?,
    val source: String?,
    val precisionM: Int?,
    val habitat: Double?,
    val pTrigger: Double?,
    val pSeasonTrigger: Double?,
)

object FindsParser {
    fun parse(json: String): List<Find> {
        val arr = JSONObject(json).getJSONArray("finds")
        val out = ArrayList<Find>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val date = o.getString("date")
            out += Find(
                idx = out.size,
                date = date,
                epochDay = LocalDate.parse(date).toEpochDay(),
                lon = o.getDouble("lon"),
                lat = o.getDouble("lat"),
                species = o.optString("species", ""),
                country = o.str("country"),
                source = o.str("source"),
                precisionM = if (o.isNull("precision_m")) null else o.optInt("precision_m"),
                habitat = o.num("habitat"),
                pTrigger = o.num("p_trigger"),
                pSeasonTrigger = o.num("p_season_trigger"),
            )
        }
        return out
    }

    /** Finds within +- [days] of [center], renumbered 0.. (for the map layer index). */
    fun near(all: List<Find>, center: LocalDate, days: Int = 7): List<Find> {
        val c = center.toEpochDay()
        return all.filter { Math.abs(it.epochDay - c) <= days }
            .sortedBy { it.epochDay }
            .mapIndexed { i, f -> f.copy(idx = i) }
    }

    private fun JSONObject.str(name: String): String? =
        if (!has(name) || isNull(name)) null else optString(name).trim().takeIf { it.isNotEmpty() }

    private fun JSONObject.num(name: String): Double? =
        if (!has(name) || isNull(name)) null else optDouble(name, Double.NaN).takeIf { !it.isNaN() }
}

/** "Top X %" of the season days for a held-out percentile p (p = 0.83 -> 17), at least 1. */
fun topPercent(p: Double): Int = Math.round((1.0 - p) * 100.0).toInt().coerceAtLeast(1)
