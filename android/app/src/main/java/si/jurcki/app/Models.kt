package si.jurcki.app

import android.graphics.Bitmap
import android.graphics.Color
import androidx.core.graphics.ColorUtils
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

/** Default location of the published backend output (the repo is private -> token needed). */
const val DEFAULT_BASE_URL = "https://raw.githubusercontent.com/DavidCufer/Jurcki/data/output/"

enum class BaseMap { STREETS, TOPO }

data class AppSettings(
    val baseUrl: String = DEFAULT_BASE_URL,
    val token: String = "",
    val opacity: Float = 0.7f,
    val baseMap: BaseMap = BaseMap.STREETS,
)

/** `bounds` of meta.json: [minLon, minLat, maxLon, maxLat] in EPSG:4326. */
data class Bounds(val minLon: Double, val minLat: Double, val maxLon: Double, val maxLat: Double)

/** One legend stop: score value (0..1) -> ARGB colour. */
data class LegendStop(val value: Double, val color: Int)

data class Meta(
    val generated: String,
    val days: List<String>,
    val bounds: Bounds,
    val legend: List<LegendStop>,
)

data class Hotspot(
    val idx: Int,
    val lat: Double,
    val lon: Double,
    val score: Double,
    val habitat: Double?,
    val trigger: Double?,
    val elev: Double?,
)

data class Report(
    val idx: Int,
    val lat: Double,
    val lon: Double,
    val date: String?,
    val source: String?,
    val precisionM: Double?,
    val text: String?,
)

/** Everything needed to draw one forecast day. */
data class DayData(
    val overlay: Bitmap?,
    val hotspots: List<Hotspot>,
    val fromCache: Boolean,
    val error: String?,
)

sealed interface Selection {
    data class OfHotspot(val hotspot: Hotspot) : Selection
    data class OfReport(val report: Report) : Selection
}

val DEFAULT_LEGEND: List<LegendStop> = listOf(
    LegendStop(0.0, Color.parseColor("#FFF7BC")),
    LegendStop(0.35, Color.parseColor("#FEC44F")),
    LegendStop(0.65, Color.parseColor("#EC7014")),
    LegendStop(1.0, Color.parseColor("#7F0000")),
)

/** Linear interpolation of the legend, used for hotspot circles and the bottom sheet. */
fun List<LegendStop>.colorAt(score: Double): Int {
    if (isEmpty()) return Color.GRAY
    if (score <= first().value) return first().color
    if (score >= last().value) return last().color
    for (i in 1 until size) {
        val a = this[i - 1]
        val b = this[i]
        if (score <= b.value) {
            val t = ((score - a.value) / (b.value - a.value)).toFloat().coerceIn(0f, 1f)
            return ColorUtils.blendARGB(a.color, b.color, t)
        }
    }
    return last().color
}

fun colorToHex(color: Int): String = String.format(Locale.US, "#%06X", color and 0xFFFFFF)

// ---------------------------------------------------------------------------------------------
// Parsing (org.json is part of Android, no extra dependency)
// ---------------------------------------------------------------------------------------------

private fun JSONObject.str(name: String): String? =
    if (isNull(name)) null else optString(name).trim().takeIf { it.isNotEmpty() }

private fun JSONObject.num(name: String): Double? =
    if (isNull(name)) null else optDouble(name, Double.NaN).takeIf { !it.isNaN() }

fun parseMeta(json: String): Meta {
    val o = JSONObject(json)

    val daysArr = o.getJSONArray("days")
    val days = (0 until daysArr.length()).map { daysArr.getString(it) }.distinct().sorted()
    require(days.isNotEmpty()) { "days must not be empty" }

    val b = o.getJSONArray("bounds")
    require(b.length() == 4) { "bounds must have 4 numbers" }
    val bounds = Bounds(b.getDouble(0), b.getDouble(1), b.getDouble(2), b.getDouble(3))

    val legend = mutableListOf<LegendStop>()
    val legArr: JSONArray? = o.optJSONArray("legend")
    if (legArr != null) {
        for (i in 0 until legArr.length()) {
            val e = legArr.optJSONArray(i) ?: continue
            if (e.length() < 2) continue
            try {
                legend += LegendStop(e.getDouble(0), Color.parseColor(e.getString(1)))
            } catch (e: Exception) {
                // skip malformed stop
            }
        }
    }
    val cleaned = legend.sortedBy { it.value }.distinctBy { it.value }

    return Meta(
        generated = o.optString("generated", ""),
        days = days,
        bounds = bounds,
        legend = if (cleaned.size >= 2) cleaned else DEFAULT_LEGEND,
    )
}

/** Calls [block] with (lon, lat, properties) for every Point feature. */
private inline fun forEachPoint(json: String, block: (Double, Double, JSONObject) -> Unit) {
    val features = JSONObject(json).optJSONArray("features") ?: return
    for (i in 0 until features.length()) {
        val f = features.optJSONObject(i) ?: continue
        val g = f.optJSONObject("geometry") ?: continue
        if (g.optString("type") != "Point") continue
        val c = g.optJSONArray("coordinates") ?: continue
        if (c.length() < 2) continue
        val lon = c.optDouble(0, Double.NaN)
        val lat = c.optDouble(1, Double.NaN)
        if (lon.isNaN() || lat.isNaN()) continue
        block(lon, lat, f.optJSONObject("properties") ?: JSONObject())
    }
}

fun parseHotspots(json: String): List<Hotspot> {
    val out = ArrayList<Hotspot>()
    forEachPoint(json) { lon, lat, p ->
        out += Hotspot(
            idx = out.size,
            lat = lat,
            lon = lon,
            score = p.num("score") ?: 0.0,
            habitat = p.num("habitat"),
            trigger = p.num("trigger"),
            elev = p.num("elev"),
        )
    }
    // Draw low scores first so the best cells end up on top.
    return out.sortedBy { it.score }.mapIndexed { i, h -> h.copy(idx = i) }
}

fun parseReports(json: String): List<Report> {
    val out = ArrayList<Report>()
    forEachPoint(json) { lon, lat, p ->
        out += Report(
            idx = out.size,
            lat = lat,
            lon = lon,
            date = p.str("date"),
            source = p.str("source"),
            precisionM = p.num("precision_m"),
            text = p.str("text"),
        )
    }
    return out
}

/** GeoJSON for the hotspot circle layer (colour precomputed from the legend). */
fun hotspotsToGeoJson(list: List<Hotspot>, legend: List<LegendStop>): String {
    val features = JSONArray()
    for (h in list) {
        features.put(
            JSONObject()
                .put("type", "Feature")
                .put("geometry", JSONObject().put("type", "Point").put("coordinates", JSONArray().put(h.lon).put(h.lat)))
                .put(
                    "properties",
                    JSONObject()
                        .put("idx", h.idx)
                        .put("score", h.score)
                        .put("color", colorToHex(legend.colorAt(h.score))),
                ),
        )
    }
    return JSONObject().put("type", "FeatureCollection").put("features", features).toString()
}

fun reportsToGeoJson(list: List<Report>): String {
    val features = JSONArray()
    for (r in list) {
        features.put(
            JSONObject()
                .put("type", "Feature")
                .put("geometry", JSONObject().put("type", "Point").put("coordinates", JSONArray().put(r.lon).put(r.lat)))
                .put("properties", JSONObject().put("idx", r.idx)),
        )
    }
    return JSONObject().put("type", "FeatureCollection").put("features", features).toString()
}
