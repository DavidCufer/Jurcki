package si.jurcki.app.engine

import kotlin.math.floor

/** Position of a lon or lat on one lattice axis: lower corner index and fraction towards the next one. */
class AxisPos(val i0: Int, val d: Double)

/** Coarse features + orography of one weather download: everything needed to score any of its days. */
class WeatherField(
    val dates: List<String>,
    val features: CoarseFeatures,
    val orog: DoubleArray,
) {
    fun indexOf(date: String): Int = dates.indexOf(date)
}

/** Score and trigger of one day for every pixel (NaN outside the mask). */
class DayScores(val score: FloatArray, val trigger: FloatArray, val season: Double)

/** Everything the UI needs for one displayed day. */
class DayMap(
    val date: String,
    val season: Double,
    val argb: IntArray,
    val hotspots: List<Hotspot>,
    /** Mean score over the land pixels. */
    val meanScore: Double,
)

/** The five trigger inputs at one pixel (temperatures already lapse-corrected). */
class PixelFeatures(
    val t20: Double,
    val p26: Double,
    val pulse: Double,
    val tmax10: Double,
    val tmin7: Double,
)

/** Per-day coarse values in lattice order (one DoubleArray of `nLat * nLon` per feature). */
internal class DayGrid(
    val t20: DoubleArray,
    val p26: DoubleArray,
    val pulse: DoubleArray,
    val tmax10: DoubleArray,
    val tmin7: DoubleArray,
)

/**
 * The on-phone model: downscaling of the coarse features, trigger, score, rendering and hotspots
 * (docs/APP_ENGINE.md sections 5-9). Pure Kotlin, no Android types.
 */
class Engine(val pkg: ModelPackage) {

    val geometry = Geometry(pkg.geo)
    private val lat = pkg.lattice
    private val nx = pkg.geo.nx
    private val ny = pkg.geo.ny

    private val rowPos: Array<AxisPos> = Array(ny) { axis(geometry.lats[it], lat.lat0, lat.step, lat.nLat) }
    private val colPos: Array<AxisPos> = Array(nx) { axis(geometry.lons[it], lat.lon0, lat.step, lat.nLon) }

    /** Coarse features of a weather download. */
    fun field(w: DailyWeather): WeatherField = WeatherField(w.dates, coarseFeatures(w, pkg.params), w.orog)

    /** Season factor of the month of an ISO date (`yyyy-MM-dd`, local day). */
    fun seasonFactor(date: String): Double = pkg.seasonFactor[date.substring(5, 7).toInt()]

    /** score = clamp(habitat * season * trigger / reference, 0, 1). */
    fun scoreOf(habitat: Double, season: Double, trigger: Double): Double =
        clamp(habitat * season * trigger / pkg.reference, 0.0, 1.0)

    /** Downscaled features at an arbitrary place (used by the parity tests; the raster loop shares the maths). */
    fun pixelFeatures(field: WeatherField, dayIdx: Int, lon: Double, latitude: Double, elev: Double): PixelFeatures {
        val g = dayGrid(field, dayIdx)
        val rp = axis(latitude, lat.lat0, lat.step, lat.nLat)
        val cp = axis(lon, lat.lon0, lat.step, lat.nLon)
        val out = DoubleArray(5)
        features(g, field.orog, rp.i0, rp.d, cp.i0, cp.d, elev, out)
        return PixelFeatures(out[0], out[1], out[2], out[3], out[4])
    }

    /** Trigger of the downscaled features. */
    fun trigger(f: PixelFeatures): Double = triggerOf(f.t20, f.p26, f.pulse, f.tmax10, f.tmin7, pkg.params)

    /** Score and trigger of every pixel for one day of [field]. */
    fun computeDay(field: WeatherField, dayIdx: Int): DayScores {
        val g = dayGrid(field, dayIdx)
        val season = seasonFactor(field.dates[dayIdx])
        val n = nx * ny
        val score = FloatArray(n)
        val trig = FloatArray(n)
        score.fill(Float.NaN)
        trig.fill(Float.NaN)
        val out = DoubleArray(5)
        val p = pkg.params
        for (r in 0 until ny) {
            val rp = rowPos[r]
            for (c in 0 until nx) {
                val idx = r * nx + c
                if (pkg.mask[idx].toInt() == 0) continue
                val cp = colPos[c]
                features(g, field.orog, rp.i0, rp.d, cp.i0, cp.d, pkg.elev[idx].toDouble(), out)
                val tr = triggerOf(out[0], out[1], out[2], out[3], out[4], p)
                trig[idx] = tr.toFloat()
                score[idx] = scoreOf(pkg.habitat[idx].toDouble(), season, tr).toFloat()
            }
        }
        return DayScores(score, trig, season)
    }

    /** Score, colours and hotspots of one day. */
    fun dayMap(field: WeatherField, dayIdx: Int): DayMap {
        val s = computeDay(field, dayIdx)
        var sum = 0.0
        var cnt = 0
        for (i in s.score.indices) {
            val v = s.score[i]
            if (!v.isNaN()) {
                sum += v
                cnt++
            }
        }
        return DayMap(
            date = field.dates[dayIdx],
            season = s.season,
            argb = Render.toArgb(s.score, pkg.colorMap),
            hotspots = HotspotPicker.pick(pkg, geometry, s.score, s.trigger),
            meanScore = if (cnt > 0) sum / cnt else 0.0,
        )
    }

    // ---- internals -------------------------------------------------------------------------

    private fun dayGrid(field: WeatherField, dayIdx: Int): DayGrid {
        val f = field.features
        val n = lat.size
        return DayGrid(
            t20 = DoubleArray(n) { f.t20[it][dayIdx] },
            p26 = DoubleArray(n) { f.p26[it][dayIdx] },
            pulse = DoubleArray(n) { f.pulse[it][dayIdx] },
            tmax10 = DoubleArray(n) { f.tmax10[it][dayIdx] },
            tmin7 = DoubleArray(n) { f.tmin7[it][dayIdx] },
        )
    }

    /** Bilinear interpolation of the five features + lapse shift of the temperatures; writes to [out]. */
    private fun features(
        g: DayGrid,
        orog: DoubleArray,
        i0: Int,
        di: Double,
        j0: Int,
        dj: Double,
        elev: Double,
        out: DoubleArray,
    ) {
        val w00 = (1 - di) * (1 - dj)
        val w01 = (1 - di) * dj
        val w10 = di * (1 - dj)
        val w11 = di * dj
        val k00 = i0 * lat.nLon + j0
        val k01 = k00 + 1
        val k10 = k00 + lat.nLon
        val k11 = k10 + 1
        val dz = pkg.lapseRate * (elev - (orog[k00] * w00 + orog[k01] * w01 + orog[k10] * w10 + orog[k11] * w11))
        out[0] = g.t20[k00] * w00 + g.t20[k01] * w01 + g.t20[k10] * w10 + g.t20[k11] * w11 + dz
        out[1] = g.p26[k00] * w00 + g.p26[k01] * w01 + g.p26[k10] * w10 + g.p26[k11] * w11
        out[2] = g.pulse[k00] * w00 + g.pulse[k01] * w01 + g.pulse[k10] * w10 + g.pulse[k11] * w11
        out[3] = g.tmax10[k00] * w00 + g.tmax10[k01] * w01 + g.tmax10[k10] * w10 + g.tmax10[k11] * w11 + dz
        out[4] = g.tmin7[k00] * w00 + g.tmin7[k01] * w01 + g.tmin7[k10] * w10 + g.tmin7[k11] * w11 + dz
    }

    companion object {
        /** Clamped lattice axis position: i0 = min(floor(f), n - 2), d = f - i0 with f clamped to 0..n-1. */
        fun axis(v: Double, v0: Double, step: Double, n: Int): AxisPos {
            val f = clamp((v - v0) / step, 0.0, (n - 1).toDouble())
            val i0 = minOf(floor(f).toInt(), n - 2)
            return AxisPos(i0, f - i0)
        }
    }
}
