package si.jurcki.app.engine

import android.graphics.Bitmap

/** The only Android-dependent piece of the engine package: ARGB ints (straight alpha) -> Bitmap. */
fun argbToBitmap(argb: IntArray, width: Int, height: Int): Bitmap {
    val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    bmp.setPixels(argb, 0, width, 0, 0, width, height)
    return bmp
}
