package si.jurcki.app.engine

import kotlin.math.PI
import kotlin.math.atan
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.tan

/** lon/lat <-> pixel of the Web-Mercator aligned phone raster (docs/APP_ENGINE.md section 2). */
class Geometry(val g: RasterGeo) {

    fun mercX(lon: Double): Double = g.rEarth * Math.toRadians(lon)

    fun mercY(lat: Double): Double = g.rEarth * ln(tan(PI / 4 + Math.toRadians(lat) / 2))

    /** Fractional column; pixel c covers [c, c + 1), its centre is at c + 0.5. */
    fun col(lon: Double): Double = (mercX(lon) - g.x0) / g.px

    /** Fractional row; row 0 is the northern edge. */
    fun row(lat: Double): Double = (g.yTop - mercY(lat)) / g.px

    fun lonOf(c: Int): Double = Math.toDegrees((g.x0 + (c + 0.5) * g.px) / g.rEarth)

    fun latOf(r: Int): Double = Math.toDegrees(2 * atan(exp((g.yTop - (r + 0.5) * g.px) / g.rEarth)) - PI / 2)

    /** Pixel index (row * nx + col) containing lon/lat, or -1 outside the raster. */
    fun indexOf(lon: Double, lat: Double): Int {
        val r = floor(row(lat)).toInt()
        val c = floor(col(lon)).toInt()
        if (r < 0 || r >= g.ny || c < 0 || c >= g.nx) return -1
        return r * g.nx + c
    }

    /** Pixel-centre longitudes, one per column. */
    val lons: DoubleArray = DoubleArray(g.nx) { lonOf(it) }

    /** Pixel-centre latitudes, one per row. */
    val lats: DoubleArray = DoubleArray(g.ny) { latOf(it) }
}
