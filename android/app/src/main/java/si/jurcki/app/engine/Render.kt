package si.jurcki.app.engine

/** Score -> ARGB through the 64-level colour table (docs/APP_ENGINE.md section 8). No Android types. */
object Render {

    /**
     * One ARGB int per pixel (row-major, row 0 = north). NaN score (outside the mask) -> 0 =
     * fully transparent. The array can be turned into a Bitmap with `argbToBitmap`.
     */
    fun toArgb(score: FloatArray, colorMap: ColorMap): IntArray {
        val out = IntArray(score.size)
        for (i in score.indices) {
            val s = score[i]
            out[i] = if (s.isNaN()) 0 else colorMap.lutArgb[colorMap.level(s.toDouble())]
        }
        return out
    }
}
