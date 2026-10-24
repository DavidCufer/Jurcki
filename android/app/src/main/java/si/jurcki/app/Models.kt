package si.jurcki.app

import org.json.JSONArray
import org.json.JSONObject
import si.jurcki.app.engine.ColorMap
import si.jurcki.app.engine.Find
import si.jurcki.app.engine.Hotspot

enum class BaseMap { STREETS, TOPO }

/** Persisted settings (DataStore). No server address or token: the app needs no configuration. */
data class AppSettings(
    val opacity: Float = 0.7f,
    val baseMap: BaseMap = BaseMap.STREETS,
)

/** Corner coordinates of the score image: EPSG:4326 outer pixel edges. */
data class Bounds(val minLon: Double, val minLat: Double, val maxLon: Double, val maxLat: Double)

enum class Mode { FORECAST, HISTORY }

/** One day chip; [forecast] = the day is after today (model data, not yet observed). */
data class DayChip(val date: String, val forecast: Boolean)

enum class ProgressKind { LOADING_MODEL, WEATHER, WAIT_RATE_LIMIT, COMPUTING }

/** What the engine is doing right now (shown as a text next to the progress bar). */
data class Progress(val kind: ProgressKind, val done: Int = 0, val total: Int = 0, val waitSeconds: Int = 0)

enum class ErrorKind { NO_NETWORK, RATE_LIMIT, API, BAD_RESPONSE, PACKAGE, OTHER }

data class AppError(val kind: ErrorKind, val detail: String?)

/** Where the weather of the displayed days comes from. */
enum class DataSource { FORECAST, ERA5, FORECAST_FOR_HISTORY }

/** Provenance line of the displayed map. */
data class DataNote(
    val source: DataSource,
    val fetchedAtMillis: Long,
    /** Cached copy shown because the download failed. */
    val stale: Boolean,
    val fromCache: Boolean,
)

sealed interface Selection {
    data class OfHotspot(val hotspot: Hotspot) : Selection
    data class OfFind(val find: Find) : Selection
}

/** GeoJSON for the hotspot circle layer (colour from the model's colour table). */
fun hotspotsToGeoJson(list: List<Hotspot>, colorMap: ColorMap): String {
    val features = JSONArray()
    // Draw low scores first so the best cells end up on top.
    for (h in list.sortedBy { it.score }) {
        features.put(
            JSONObject()
                .put("type", "Feature")
                .put("geometry", JSONObject().put("type", "Point").put("coordinates", JSONArray().put(h.lon).put(h.lat)))
                .put(
                    "properties",
                    JSONObject()
                        .put("idx", h.idx)
                        .put("score", h.score)
                        .put("color", colorMap.hex(h.score)),
                ),
        )
    }
    return JSONObject().put("type", "FeatureCollection").put("features", features).toString()
}

/** GeoJSON points of finds; `idx` = [Find.idx] of the list that is passed. */
fun findsToGeoJson(list: List<Find>): String {
    val features = JSONArray()
    for (f in list) {
        features.put(
            JSONObject()
                .put("type", "Feature")
                .put("geometry", JSONObject().put("type", "Point").put("coordinates", JSONArray().put(f.lon).put(f.lat)))
                .put("properties", JSONObject().put("idx", f.idx)),
        )
    }
    return JSONObject().put("type", "FeatureCollection").put("features", features).toString()
}
