package si.jurcki.app.engine

import kotlin.math.ceil

/** One de-clustered hot spot: pixel position, pixel-centre lon/lat and the values at that pixel. */
data class Hotspot(
    val idx: Int,
    val row: Int,
    val col: Int,
    val lon: Double,
    val lat: Double,
    val score: Double,
    val habitat: Double,
    val trigger: Double,
    val elev: Double,
)

/** Hotspot selection (docs/APP_ENGINE.md section 9). */
object HotspotPicker {

    /**
     * Pixel indices (row * nx + col) of the picks, best first.
     *
     * rank = score + 0.001 * habitat * trigger in Float32; candidates are pixels with mask == 1 and
     * rank >= [minScore]; order: rank descending, ties by row then column; a pick suppresses every
     * candidate with pixel distance strictly < [minSepPx]; at most [maxN] picks.
     */
    fun pickIndices(
        score: FloatArray,
        habitat: FloatArray,
        trigger: FloatArray,
        mask: ByteArray,
        nx: Int,
        ny: Int,
        minScore: Double,
        minSepPx: Double,
        maxN: Int,
    ): IntArray {
        val n = nx * ny
        val thr = minScore.toFloat()
        val rank = FloatArray(n)
        var count = 0
        for (i in 0 until n) {
            if (mask[i].toInt() == 0) {
                rank[i] = Float.NaN
                continue
            }
            val s = score[i]
            val h = habitat[i]
            val t = trigger[i]
            val r = s + 0.001f * h * t
            rank[i] = r
            if (r >= thr) count++ // NaN compares false
        }
        if (count == 0) return IntArray(0)

        // Sort key: high 32 bits = inverted float bits (rank descending; ranks are positive so the
        // bit patterns are monotonic), low 32 bits = pixel index (ascending = row-major tie break).
        val keys = LongArray(count)
        var k = 0
        for (i in 0 until n) {
            val r = rank[i]
            if (r >= thr) {
                val bits = java.lang.Float.floatToIntBits(r)
                keys[k++] = ((Int.MAX_VALUE - bits).toLong() shl 32) or i.toLong()
            }
        }
        keys.sort()

        val suppressed = BooleanArray(n)
        val lim = minSepPx * minSepPx
        val reach = ceil(minSepPx).toInt()
        val picks = ArrayList<Int>()
        for (key in keys) {
            val i = (key and 0xFFFFFFFFL).toInt()
            if (suppressed[i]) continue
            picks += i
            if (picks.size >= maxN) break
            val r = i / nx
            val c = i % nx
            for (dr in -reach..reach) {
                val rr = r + dr
                if (rr < 0 || rr >= ny) continue
                for (dc in -reach..reach) {
                    val cc = c + dc
                    if (cc < 0 || cc >= nx) continue
                    if ((dr * dr + dc * dc).toDouble() < lim) suppressed[rr * nx + cc] = true
                }
            }
        }
        return picks.toIntArray()
    }

    /** Picks for one day of the package raster, with pixel-centre coordinates. */
    fun pick(pkg: ModelPackage, geometry: Geometry, score: FloatArray, trigger: FloatArray): List<Hotspot> {
        val nx = pkg.geo.nx
        val cfg = pkg.hotspotConfig
        val idx = pickIndices(
            score, pkg.habitat, trigger, pkg.mask, nx, pkg.geo.ny,
            cfg.minScore, cfg.minSepPx, cfg.maxN,
        )
        return idx.mapIndexed { n, i ->
            val r = i / nx
            val c = i % nx
            Hotspot(
                idx = n,
                row = r,
                col = c,
                lon = geometry.lons[c],
                lat = geometry.lats[r],
                score = score[i].toDouble(),
                habitat = pkg.habitat[i].toDouble(),
                trigger = trigger[i].toDouble(),
                elev = pkg.elev[i].toDouble(),
            )
        }
    }
}
