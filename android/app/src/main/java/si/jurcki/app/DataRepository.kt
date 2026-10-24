package si.jurcki.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/** HTTP status != 2xx. 404 is expected for optional files (e.g. no hotspots for a day). */
class HttpStatusException(val code: Int) : IOException("HTTP $code")

/** Result of loading meta.json: [value] is null only if neither network nor cache worked. */
data class MetaResult(val value: Meta?, val fromCache: Boolean, val error: Throwable?)

/**
 * Downloads the backend output (meta.json, score_*.png, hotspots_*.geojson, reports.geojson) with
 * OkHttp and keeps a copy on disk (one sub-folder per base URL). Every load falls back to the
 * last cached copy when the network fails.
 */
class DataRepository(context: Context) {

    private val root = File(context.applicationContext.filesDir, "data-cache")

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .callTimeout(90, TimeUnit.SECONDS)
        .build()

    private fun dirFor(baseUrl: String): File {
        val hash = MessageDigest.getInstance("SHA-256")
            .digest(normalizeBaseUrl(baseUrl).toByteArray())
            .joinToString("") { "%02x".format(it) }
            .take(12)
        return File(root, hash).also { it.mkdirs() }
    }

    // ---- public API -------------------------------------------------------------------------

    /** Cached meta.json for [s], without touching the network. */
    suspend fun cachedMeta(s: AppSettings): Meta? = withContext(Dispatchers.IO) {
        readMeta(File(dirFor(s.baseUrl), META))
    }

    /** Downloads meta.json; on failure falls back to the cached copy. */
    suspend fun refreshMeta(s: AppSettings): MetaResult = withContext(Dispatchers.IO) {
        val dir = dirFor(s.baseUrl)
        val metaFile = File(dir, META)
        val oldGenerated = readMeta(metaFile)?.generated
        try {
            val tmp = File(dir, "$META.new")
            download(url(s, META), s.token, tmp)
            val meta = parseMeta(tmp.readText())
            if (!tmp.renameTo(metaFile)) {
                tmp.copyTo(metaFile, overwrite = true)
                tmp.delete()
            }
            if (oldGenerated != null && oldGenerated != meta.generated) purgeDays(dir)
            MetaResult(meta, fromCache = false, error = null)
        } catch (e: Exception) {
            MetaResult(readMeta(metaFile), fromCache = true, error = e)
        }
    }

    /**
     * Loads the score PNG and the hotspots of [day]. With [force] the files are re-downloaded,
     * otherwise a cached copy is used when present. A failed download falls back to the cache.
     */
    suspend fun loadDay(s: AppSettings, day: String, force: Boolean): DayData = withContext(Dispatchers.IO) {
        val dir = dirFor(s.baseUrl)
        val png = File(dir, "score_$day.png")
        val geo = File(dir, "hotspots_$day.geojson")
        var error: Throwable? = null
        var fromCache = false

        for ((name, file) in listOf("score_$day.png" to png, "hotspots_$day.geojson" to geo)) {
            if (!force && file.exists()) {
                fromCache = true
                continue
            }
            try {
                download(url(s, name), s.token, file)
            } catch (e: Exception) {
                if (file.exists()) fromCache = true
                // A missing hotspots file is not an error worth showing.
                if (!(name.endsWith(".geojson") && e is HttpStatusException && e.code == 404)) error = e
            }
        }

        val bitmap = decode(png)
        val hotspots = runCatching { if (geo.exists()) parseHotspots(geo.readText()) else emptyList() }
            .getOrDefault(emptyList())
        DayData(bitmap, hotspots, fromCache, error?.let { describe(it) })
    }

    /** Best-effort download of a day's files so that it works offline later. */
    suspend fun prefetchDay(s: AppSettings, day: String) = withContext(Dispatchers.IO) {
        val dir = dirFor(s.baseUrl)
        for (name in listOf("score_$day.png", "hotspots_$day.geojson")) {
            val file = File(dir, name)
            if (file.exists()) continue
            try {
                download(url(s, name), s.token, file)
            } catch (e: Exception) {
                // ignore
            }
        }
    }

    suspend fun loadReports(s: AppSettings, force: Boolean): List<Report> = withContext(Dispatchers.IO) {
        val file = File(dirFor(s.baseUrl), REPORTS)
        if (force || !file.exists()) {
            try {
                download(url(s, REPORTS), s.token, file)
            } catch (e: Exception) {
                // keep the cached copy, if any
            }
        }
        runCatching { if (file.exists()) parseReports(file.readText()) else emptyList() }
            .getOrDefault(emptyList())
    }

    // ---- internals --------------------------------------------------------------------------

    private fun readMeta(f: File): Meta? =
        if (f.exists()) runCatching { parseMeta(f.readText()) }.getOrNull() else null

    private fun purgeDays(dir: File) {
        dir.listFiles()?.forEach {
            if (it.name.startsWith("score_") || it.name.startsWith("hotspots_")) it.delete()
        }
    }

    private fun decode(f: File): Bitmap? {
        if (!f.exists()) return null
        val bmp = BitmapFactory.decodeFile(f.absolutePath)
        if (bmp == null) f.delete() // corrupt download
        return bmp
    }

    private fun url(s: AppSettings, name: String) = normalizeBaseUrl(s.baseUrl) + name

    /** Blocking download to [dest] (atomic: temp file + rename). Call on Dispatchers.IO. */
    private fun download(url: String, token: String, dest: File) {
        val request = try {
            Request.Builder().url(url).apply {
                if (token.isNotBlank()) header("Authorization", "Bearer ${token.trim()}")
            }.build()
        } catch (e: IllegalArgumentException) {
            throw IOException("Invalid URL: $url", e)
        }
        client.newCall(request).execute().use { resp ->
            if (!resp.isSuccessful) throw HttpStatusException(resp.code)
            val body = resp.body ?: throw IOException("Empty response")
            dest.parentFile?.mkdirs()
            val tmp = File(dest.parentFile, dest.name + ".tmp")
            body.byteStream().use { input -> tmp.outputStream().use { out -> input.copyTo(out) } }
            if (!tmp.renameTo(dest)) {
                tmp.copyTo(dest, overwrite = true)
                tmp.delete()
            }
        }
    }

    private fun describe(e: Throwable): String =
        if (e is HttpStatusException) "HTTP ${e.code}" else (e.message ?: e.javaClass.simpleName)

    companion object {
        private const val META = "meta.json"
        private const val REPORTS = "reports.geojson"

        /** Ensures exactly one trailing slash and strips a pasted "meta.json" suffix. */
        fun normalizeBaseUrl(raw: String): String {
            var u = raw.trim()
            if (u.endsWith("/$META")) u = u.removeSuffix(META)
            return u.trimEnd('/') + "/"
        }
    }
}
