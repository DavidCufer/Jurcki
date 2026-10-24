package si.jurcki.app.engine

import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.File
import java.io.IOException
import java.net.SocketTimeoutException
import java.security.MessageDigest
import java.time.Instant
import java.time.ZoneId
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Progress of a download: [done] of [total] requests finished; [waitSeconds] > 0 while waiting after HTTP 429. */
data class WeatherProgress(val done: Int, val total: Int, val waitSeconds: Int = 0)

/** Raw response bodies of one set of requests. */
class FetchResult(
    val bodies: List<String>,
    /** True if nothing had to be downloaded. */
    val fromCache: Boolean,
    /** True if at least one body is an out-of-date cache copy because the download failed. */
    val stale: Boolean,
    /** Time (epoch millis) the oldest body was downloaded. */
    val fetchedAtMillis: Long,
)

internal class RawResponse(val code: Int, val body: String)

/**
 * Open-Meteo GETs with OkHttp and a per-request disk cache (file name = hash of the URL).
 * Cache policy per call: `maxAgeMs == null` -> cache forever (archive); otherwise a copy is fresh while
 * younger than `maxAgeMs` and from the same Europe/Ljubljana calendar day. If a download fails and an
 * old copy exists, the old copy is used (and reported as stale).
 */
class WeatherClient(
    private val http: OkHttpClient,
    private val dir: File,
    private val now: () -> Long = { System.currentTimeMillis() },
) {
    private val zone = ZoneId.of(OpenMeteo.TIMEZONE)

    suspend fun fetchAll(
        requests: List<WeatherRequest>,
        maxAgeMs: Long?,
        force: Boolean,
        onProgress: (WeatherProgress) -> Unit,
    ): FetchResult {
        val bodies = ArrayList<String>(requests.size)
        var downloaded = false
        var stale = false
        var oldest = Long.MAX_VALUE
        for ((i, req) in requests.withIndex()) {
            onProgress(WeatherProgress(i, requests.size))
            val file = cacheFile(req)
            val cached = readCache(file)
            if (cached != null && !force && isFresh(file, maxAgeMs)) {
                bodies += cached
                oldest = minOf(oldest, file.lastModified())
                continue
            }
            try {
                val body = download(req) { wait -> onProgress(WeatherProgress(i, requests.size, wait)) }
                writeCache(file, body, req.url == OpenMeteo.ARCHIVE_URL)
                bodies += body
                downloaded = true
                oldest = minOf(oldest, now())
            } catch (e: WeatherException) {
                if (cached == null) throw e
                bodies += cached
                stale = true
                oldest = minOf(oldest, file.lastModified())
            }
        }
        onProgress(WeatherProgress(requests.size, requests.size))
        return FetchResult(bodies, fromCache = !downloaded, stale = stale, fetchedAtMillis = oldest)
    }

    /** Deletes every cached response. */
    suspend fun clearCache() = withContext(Dispatchers.IO) {
        dir.listFiles()?.forEach { if (it.name.startsWith("wx-")) it.delete() }
    }

    // ---- download ----------------------------------------------------------------------------

    private suspend fun download(req: WeatherRequest, onWait: (Int) -> Unit): String {
        val request = Request.Builder()
            .url(req.fullUrl())
            .header("User-Agent", "Jurcki-Android/0.2")
            .build()
        var serverErrors = 0
        var timeouts = 0
        var rateWaits = 0
        while (true) {
            val raw: RawResponse = try {
                http.newCall(request).await()
            } catch (e: SocketTimeoutException) {
                if (timeouts++ < 1) {
                    delay(1500)
                    continue
                }
                throw WeatherException(WeatherException.Kind.NO_NETWORK, "Timeout: ${e.message}", e)
            } catch (e: IOException) {
                throw WeatherException(WeatherException.Kind.NO_NETWORK, e.message ?: e.javaClass.simpleName, e)
            }
            when {
                raw.code in 200..299 -> {
                    OpenMeteo.parseLocations(raw.body) // validates: throws on {"error":true} or bad JSON
                    return raw.body
                }
                raw.code == 429 -> {
                    if (rateWaits++ >= 2) {
                        throw WeatherException(WeatherException.Kind.RATE_LIMIT, "HTTP 429 (Open-Meteo rate limit)")
                    }
                    for (left in 60 downTo 1) {
                        onWait(left)
                        delay(1000)
                    }
                }
                raw.code >= 500 -> {
                    if (serverErrors++ >= 3) {
                        throw WeatherException(WeatherException.Kind.API, "HTTP ${raw.code}: ${reasonOf(raw.body)}")
                    }
                    delay(1000L shl (serverErrors - 1)) // 1 s, 2 s, 4 s
                }
                else -> throw WeatherException(WeatherException.Kind.API, "HTTP ${raw.code}: ${reasonOf(raw.body)}")
            }
        }
    }

    private fun reasonOf(body: String): String =
        try {
            org.json.JSONObject(body).optString("reason", body.take(200))
        } catch (e: Exception) {
            body.take(200)
        }

    private suspend fun Call.await(): RawResponse = suspendCancellableCoroutine { cont: CancellableContinuation<RawResponse> ->
        val call = this
        cont.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                cont.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                val result = try {
                    RawResponse(response.code, response.body?.string() ?: "")
                } catch (e: IOException) {
                    response.close()
                    cont.resumeWithException(e)
                    return
                }
                response.close()
                cont.resume(result)
            }
        })
    }

    // ---- cache -------------------------------------------------------------------------------

    private fun cacheFile(req: WeatherRequest): File {
        val hash = MessageDigest.getInstance("SHA-256")
            .digest(req.fullUrl().toByteArray())
            .joinToString("") { "%02x".format(it) }
            .take(24)
        val kind = if (req.url == OpenMeteo.ARCHIVE_URL) "a" else "f"
        return File(dir, "wx-$kind-$hash.json")
    }

    private fun isFresh(file: File, maxAgeMs: Long?): Boolean {
        if (maxAgeMs == null) return true
        val t = file.lastModified()
        val n = now()
        if (n - t >= maxAgeMs) return false
        val d1 = Instant.ofEpochMilli(t).atZone(zone).toLocalDate()
        val d2 = Instant.ofEpochMilli(n).atZone(zone).toLocalDate()
        return d1 == d2
    }

    private fun readCache(file: File): String? =
        try {
            if (file.isFile) file.readText() else null
        } catch (e: IOException) {
            null
        }

    private fun writeCache(file: File, body: String, archive: Boolean) {
        try {
            dir.mkdirs()
            val tmp = File(dir, file.name + ".tmp")
            tmp.writeText(body)
            if (!tmp.renameTo(file)) {
                file.writeText(body)
                tmp.delete()
            }
            if (archive) pruneArchive()
        } catch (e: IOException) {
            // cache is best effort
        }
    }

    private fun pruneArchive() {
        val files = dir.listFiles { f -> f.name.startsWith("wx-a-") } ?: return
        if (files.size <= MAX_ARCHIVE_FILES) return
        files.sortedBy { it.lastModified() }.take(files.size - MAX_ARCHIVE_FILES).forEach { it.delete() }
    }

    companion object {
        const val FORECAST_MAX_AGE_MS = 3L * 60 * 60 * 1000
        private const val MAX_ARCHIVE_FILES = 60

        fun defaultHttpClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(45, TimeUnit.SECONDS)
            .callTimeout(90, TimeUnit.SECONDS)
            .build()
    }
}
