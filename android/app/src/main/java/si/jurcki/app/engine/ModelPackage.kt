package si.jurcki.app.engine

import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.util.zip.GZIPInputStream

/** Raster geometry of the phone raster (`model.json` `raster`), see docs/APP_ENGINE.md section 2. */
data class RasterGeo(
    val nx: Int,
    val ny: Int,
    val px: Double,
    val x0: Double,
    val yTop: Double,
    val rEarth: Double,
    val minLon: Double,
    val minLat: Double,
    val maxLon: Double,
    val maxLat: Double,
) {
    companion object {
        fun fromJson(o: JSONObject): RasterGeo {
            val b = o.getJSONArray("bounds")
            return RasterGeo(
                nx = o.getInt("nx"),
                ny = o.getInt("ny"),
                px = o.getDouble("px"),
                x0 = o.getDouble("x0"),
                yTop = o.getDouble("y_top"),
                rEarth = o.getDouble("r_earth"),
                minLon = b.getDouble(0),
                minLat = b.getDouble(1),
                maxLon = b.getDouble(2),
                maxLat = b.getDouble(3),
            )
        }
    }
}

/** Trigger parameters (`model.json` `trigger.params`). */
data class TriggerParams(
    val tOpt: Double,
    val tSd: Double,
    val tWindow: Int,
    val pWindow: Int,
    val pHalf: Double,
    val pulseMin: Double,
    val pulseLagLo: Int,
    val pulseLagHi: Int,
    val heatTmax: Double,
    val heatWindow: Int,
    val frostTmin: Double,
    val frostWindow: Int,
) {
    companion object {
        fun fromJson(o: JSONObject): TriggerParams {
            val lag = o.getJSONArray("pulse_lag")
            return TriggerParams(
                tOpt = o.getDouble("t_opt"),
                tSd = o.getDouble("t_sd"),
                tWindow = o.getInt("t_window"),
                pWindow = o.getInt("p_window"),
                pHalf = o.getDouble("p_half"),
                pulseMin = o.getDouble("pulse_min"),
                pulseLagLo = lag.getInt(0),
                pulseLagHi = lag.getInt(1),
                heatTmax = o.getDouble("heat_tmax"),
                heatWindow = o.getInt("heat_window"),
                frostTmin = o.getDouble("frost_tmin"),
                frostWindow = o.getInt("frost_window"),
            )
        }
    }
}

/** The 0.25 degree weather lattice (`model.json` `weather.lattice`), point k = iLat * nLon + iLon. */
class Lattice(
    val lat0: Double,
    val lon0: Double,
    val step: Double,
    val nLat: Int,
    val nLon: Int,
    val pointLat: DoubleArray,
    val pointLon: DoubleArray,
    val pointElev: DoubleArray,
) {
    val size: Int get() = nLat * nLon

    companion object {
        fun fromJson(o: JSONObject): Lattice {
            val nLat = o.getInt("n_lat")
            val nLon = o.getInt("n_lon")
            val pts = o.getJSONArray("points")
            require(pts.length() == nLat * nLon) { "lattice has ${pts.length()} points, expected ${nLat * nLon}" }
            val lat = DoubleArray(pts.length())
            val lon = DoubleArray(pts.length())
            val elev = DoubleArray(pts.length())
            for (k in 0 until pts.length()) {
                val p = pts.getJSONArray(k)
                lat[k] = p.getDouble(0)
                lon[k] = p.getDouble(1)
                elev[k] = p.getDouble(2)
            }
            return Lattice(o.getDouble("lat0"), o.getDouble("lon0"), o.getDouble("step"), nLat, nLon, lat, lon, elev)
        }
    }
}

/** One legend stop: score (0..1) -> opaque ARGB colour. */
data class LegendStop(val value: Double, val argb: Int)

/** Colour table (`model.json` `colormap`): 64 RGBA levels, used directly. */
class ColorMap(
    val minScore: Double,
    val alphaFullAt: Double,
    val alphaMax: Double,
    val levels: Int,
    /** ARGB int per level (straight alpha). */
    val lutArgb: IntArray,
    val legend: List<LegendStop>,
) {
    /** level = clamp(rint(score * (levels - 1)), 0, levels - 1); rint = round half to even. */
    fun level(score: Double): Int {
        if (score.isNaN()) return 0
        val l = Math.rint(score * (levels - 1))
        return if (l < 0.0) 0 else if (l > (levels - 1).toDouble()) levels - 1 else l.toInt()
    }

    /** ARGB of a score, 0 (fully transparent) for NaN. */
    fun argb(score: Double): Int = if (score.isNaN()) 0 else lutArgb[level(score)]

    /** "#RRGGBB" of the LUT colour of a score (alpha ignored), for map circles. */
    fun hex(score: Double): String = String.format(java.util.Locale.US, "#%06X", argb(score) and 0xFFFFFF)

    companion object {
        fun fromJson(o: JSONObject): ColorMap {
            val lut = o.getJSONArray("lut")
            val levels = o.getInt("levels")
            require(lut.length() == levels) { "colormap lut has ${lut.length()} rows, expected $levels" }
            val argb = IntArray(levels)
            for (i in 0 until levels) {
                val e = lut.getJSONArray(i)
                argb[i] = (e.getInt(3) shl 24) or (e.getInt(0) shl 16) or (e.getInt(1) shl 8) or e.getInt(2)
            }
            val legendArr = o.getJSONArray("legend")
            val legend = ArrayList<LegendStop>()
            for (i in 0 until legendArr.length()) {
                val e = legendArr.getJSONArray(i)
                legend += LegendStop(e.getDouble(0), parseHexRgb(e.getString(1)))
            }
            return ColorMap(
                minScore = o.getDouble("min_score"),
                alphaFullAt = o.getDouble("alpha_full_at"),
                alphaMax = o.getDouble("alpha_max"),
                levels = levels,
                lutArgb = argb,
                legend = legend,
            )
        }

        /** "#RRGGBB" -> opaque ARGB int. */
        fun parseHexRgb(hex: String): Int {
            val h = hex.trim().removePrefix("#")
            return (0xFF shl 24) or h.toInt(16)
        }
    }
}

data class HotspotConfig(val minScore: Double, val minSepPx: Double, val maxN: Int)

/**
 * Everything the phone engine needs: `model.json` plus the three gzip rasters of `assets/model/`.
 * No Android types, so the same class is used by the JVM unit tests.
 */
class ModelPackage(
    val createdAt: String,
    val geo: RasterGeo,
    val params: TriggerParams,
    val lattice: Lattice,
    val reference: Double,
    /** Index 1..12 = season factor of the month (index 0 unused). */
    val seasonFactor: DoubleArray,
    val lapseRate: Double,
    val colorMap: ColorMap,
    val hotspotConfig: HotspotConfig,
    val forecastPastDays: Int,
    val forecastDays: Int,
    val batchSize: Int,
    val historyDaysNeeded: Int,
    /** Days shown before / after today in forecast mode (2 and 6). */
    val shownPast: Int,
    val shownAhead: Int,
    /** habitat 0..1, row-major, row 0 = north; nodata -> 0. */
    val habitat: FloatArray,
    /** Metres; nodata -> 0. */
    val elev: FloatArray,
    /** 1 = Slovenia land, 0 = outside. */
    val mask: ByteArray,
    /** Raw `model.json` for anything not modelled above (e.g. request templates in tests). */
    val json: JSONObject,
) {
    companion object {
        const val HABITAT_NODATA = 255
        const val HABITAT_SCALE = 250.0f
        const val ELEV_NODATA = -32768

        /** Reads the four files via [open] (name relative to `assets/model/`). */
        fun load(open: (String) -> InputStream): ModelPackage {
            fun bytes(name: String): ByteArray = open(name).use { it.readBytes() }
            return parse(
                String(bytes("model.json"), Charsets.UTF_8),
                bytes("habitat.u8.gz"),
                bytes("elev.i16.gz"),
                bytes("mask.u8.gz"),
            )
        }

        fun parse(modelJson: String, habitatGz: ByteArray, elevGz: ByteArray, maskGz: ByteArray): ModelPackage {
            val root = JSONObject(modelJson)
            val geo = RasterGeo.fromJson(root.getJSONObject("raster"))
            val n = geo.nx * geo.ny

            val hb = gunzip(habitatGz)
            val eb = gunzip(elevGz)
            val mb = gunzip(maskGz)
            require(hb.size == n) { "habitat raster has ${hb.size} bytes, expected $n" }
            require(eb.size == 2 * n) { "elevation raster has ${eb.size} bytes, expected ${2 * n}" }
            require(mb.size == n) { "mask raster has ${mb.size} bytes, expected $n" }

            val habitat = FloatArray(n)
            for (i in 0 until n) {
                val v = hb[i].toInt() and 0xFF
                habitat[i] = if (v == HABITAT_NODATA) 0f else v / HABITAT_SCALE
            }
            val elev = FloatArray(n)
            for (i in 0 until n) {
                val lo = eb[2 * i].toInt() and 0xFF
                val hi = eb[2 * i + 1].toInt() // signed
                val v = (hi shl 8) or lo
                elev[i] = if (v == ELEV_NODATA) 0f else v.toFloat()
            }

            val score = root.getJSONObject("score")
            val season = DoubleArray(13)
            val table = score.getJSONObject("season_factor")
            for (m in 1..12) season[m] = table.getDouble(m.toString())

            val weather = root.getJSONObject("weather")
            val fr = weather.getJSONObject("forecast_recent")
            val outDays = fr.getJSONObject("output_days")
            val hs = root.getJSONObject("hotspots")

            return ModelPackage(
                createdAt = root.optString("created", ""),
                geo = geo,
                params = TriggerParams.fromJson(root.getJSONObject("trigger").getJSONObject("params")),
                lattice = Lattice.fromJson(weather.getJSONObject("lattice")),
                reference = score.getDouble("reference"),
                seasonFactor = season,
                lapseRate = root.getDouble("lapse_rate_k_per_m"),
                colorMap = ColorMap.fromJson(root.getJSONObject("colormap")),
                hotspotConfig = HotspotConfig(hs.getDouble("min_score"), hs.getDouble("min_sep_px"), hs.getInt("max_n")),
                forecastPastDays = fr.getInt("past_days"),
                forecastDays = fr.getInt("forecast_days"),
                batchSize = weather.getInt("batch_size"),
                historyDaysNeeded = root.getJSONObject("features").getInt("history_days_needed"),
                shownPast = outDays.getInt("past"),
                shownAhead = outDays.getInt("ahead"),
                habitat = habitat,
                elev = elev,
                mask = mb,
                json = root,
            )
        }

        private fun gunzip(gz: ByteArray): ByteArray =
            GZIPInputStream(ByteArrayInputStream(gz)).use { it.readBytes() }
    }
}
