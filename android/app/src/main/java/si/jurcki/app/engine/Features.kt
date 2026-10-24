package si.jurcki.app.engine

import kotlin.math.exp

/**
 * The five coarse features per lattice point and day, arrays `[point][day]`
 * (docs/APP_ENGINE.md section 5). Computed on the unshifted Open-Meteo series.
 */
class CoarseFeatures(
    val t20: Array<DoubleArray>,
    val p26: Array<DoubleArray>,
    val pulse: Array<DoubleArray>,
    val tmax10: Array<DoubleArray>,
    val tmin7: Array<DoubleArray>,
)

/** Features of all lattice points; windows are trailing, include the current day and are truncated at the start. */
fun coarseFeatures(w: DailyWeather, p: TriggerParams): CoarseFeatures {
    val n = w.tmean.size
    return CoarseFeatures(
        t20 = Array(n) { rollMean(w.tmean[it], p.tWindow) },
        p26 = Array(n) { rollSum(w.precip[it], p.pWindow) },
        pulse = Array(n) { pulseSeries(w.precip[it], p) },
        tmax10 = Array(n) { rollMax(w.tmax[it], p.heatWindow) },
        tmin7 = Array(n) { rollMin(w.tmin[it], p.frostWindow) },
    )
}

fun rollSum(x: DoubleArray, w: Int): DoubleArray = DoubleArray(x.size) { t ->
    var s = 0.0
    for (j in maxOf(0, t - w + 1)..t) s += x[j]
    s
}

fun rollMean(x: DoubleArray, w: Int): DoubleArray = DoubleArray(x.size) { t ->
    val a = maxOf(0, t - w + 1)
    var s = 0.0
    for (j in a..t) s += x[j]
    s / (t - a + 1)
}

fun rollMax(x: DoubleArray, w: Int): DoubleArray = DoubleArray(x.size) { t ->
    var m = Double.NEGATIVE_INFINITY
    for (j in maxOf(0, t - w + 1)..t) if (x[j] > m) m = x[j]
    m
}

fun rollMin(x: DoubleArray, w: Int): DoubleArray = DoubleArray(x.size) { t ->
    var m = Double.POSITIVE_INFINITY
    for (j in maxOf(0, t - w + 1)..t) if (x[j] < m) m = x[j]
    m
}

/**
 * pulse[t] = 1 if a soaking rain (3-day sum >= pulse_min) ended between `lo` and `hi` days ago:
 * `recent[t] = max(soak[t-(hi-lo) .. t])`, `pulse[t] = recent[t - lo]` for t >= lo, else 0.
 */
fun pulseSeries(precip: DoubleArray, p: TriggerParams): DoubleArray {
    val rain3 = rollSum(precip, 3)
    val soak = DoubleArray(precip.size) { if (rain3[it] >= p.pulseMin) 1.0 else 0.0 }
    val recent = rollMax(soak, p.pulseLagHi - p.pulseLagLo + 1)
    return DoubleArray(precip.size) { t -> if (t >= p.pulseLagLo) recent[t - p.pulseLagLo] else 0.0 }
}

/** Fruiting trigger 0..1 from the five features (docs/APP_ENGINE.md section 7). */
fun triggerOf(t20: Double, p26: Double, pulse: Double, tmax10: Double, tmin7: Double, p: TriggerParams): Double {
    val z = (t20 - p.tOpt) / p.tSd
    val temp = exp(-0.5 * z * z)
    val rain = 0.6 * (p26 / (p26 + p.pHalf)) + 0.4 * pulse
    val heat = clamp(1.0 - (tmax10 - p.heatTmax) / 5.0, 0.2, 1.0)
    val frost = clamp((tmin7 - p.frostTmin) / 3.0 + 1.0, 0.0, 1.0)
    return clamp(temp * rain * heat * frost, 0.0, 1.0)
}

internal fun clamp(v: Double, lo: Double, hi: Double): Double = if (v < lo) lo else if (v > hi) hi else v
