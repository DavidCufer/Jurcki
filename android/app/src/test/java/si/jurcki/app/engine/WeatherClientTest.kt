package si.jurcki.app.engine

import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.time.LocalDate

/** Client and cache behaviour with a fake HTTP layer (no network). */
class WeatherClientTest {

    private val pkg get() = TestData.pkg
    private val locations: JSONArray by lazy { TestData.json("parity/weather_full.json").getJSONArray("responses") }

    /** Answers a weather request with the slice of the full vector that it asked for. */
    private fun sliceFor(req: Request): String {
        val n = req.url.queryParameter("latitude")!!.split(",").size
        val first = req.url.queryParameter("latitude")!!.split(",").first().toDouble()
        val firstLon = req.url.queryParameter("longitude")!!.split(",").first().toDouble()
        val lat = pkg.lattice
        var start = -1
        for (k in 0 until lat.size) if (lat.pointLat[k] == first && lat.pointLon[k] == firstLon) start = k
        val out = JSONArray()
        for (k in start until start + n) out.put(locations.getJSONObject(k))
        return out.toString()
    }

    private fun response(req: Request, code: Int, body: String): Response =
        Response.Builder().request(req).protocol(Protocol.HTTP_1_1).code(code).message("test")
            .body(body.toResponseBody("application/json".toMediaType())).build()

    private class Fake(val handler: (Request) -> Response) : Interceptor {
        var calls = 0
        override fun intercept(chain: Interceptor.Chain): Response {
            calls++
            return handler(chain.request())
        }
    }

    private fun client(fake: Fake): OkHttpClient = OkHttpClient.Builder().addInterceptor(fake).build()

    private fun tmp(): File = Files.createTempDirectory("wx-test").toFile()

    private val forecastReqs get() = OpenMeteo.forecastRequests(pkg.lattice, pkg.forecastPastDays, pkg.forecastDays, pkg.batchSize)

    @Test
    fun downloads_caches_and_refreshes() = runBlocking {
        val fake = Fake { req -> response(req, 200, sliceFor(req)) }
        val now = System.currentTimeMillis()
        val c = WeatherClient(client(fake), tmp()) { now }

        val first = c.fetchAll(forecastReqs, WeatherClient.FORECAST_MAX_AGE_MS, false) { }
        assertEquals(3, fake.calls)
        assertEquals(3, first.bodies.size)
        assertFalse(first.fromCache)
        assertFalse(first.stale)

        // assemble the three bodies like the app does
        val all = first.bodies.flatMap { OpenMeteo.parseLocations(it) }
        assertEquals(120, all.size)
        val w = OpenMeteo.assembleDaily(all, pkg.lattice.nLat, pkg.lattice.nLon, pkg.lattice.pointElev)
        assertEquals(69, w.dates.size)

        // second call right away: served from the cache
        val cached = c.fetchAll(forecastReqs, WeatherClient.FORECAST_MAX_AGE_MS, false) { }
        assertEquals("no new requests", 3, fake.calls)
        assertTrue(cached.fromCache)

        // forced refresh downloads again
        c.fetchAll(forecastReqs, WeatherClient.FORECAST_MAX_AGE_MS, true) { }
        assertEquals(6, fake.calls)
        Unit
    }

    @Test
    fun old_cache_is_refreshed_and_stale_copy_used_when_offline() = runBlocking {
        val dir = tmp()
        val ok = Fake { req -> response(req, 200, sliceFor(req)) }
        val realNow = System.currentTimeMillis()
        WeatherClient(client(ok), dir) { realNow }.fetchAll(forecastReqs, WeatherClient.FORECAST_MAX_AGE_MS, false) { }

        // 4 hours later the cache is too old: offline -> the old copy is used and reported as stale
        val offline = Fake { throw IOException("no route") }
        val later = realNow + 4L * 60 * 60 * 1000
        val res = WeatherClient(client(offline), dir) { later }.fetchAll(forecastReqs, WeatherClient.FORECAST_MAX_AGE_MS, false) { }
        assertTrue(res.stale)
        assertEquals(3, res.bodies.size)
        assertTrue(offline.calls >= 3)

        // no cache + offline -> NO_NETWORK
        try {
            WeatherClient(client(offline), tmp()) { later }.fetchAll(forecastReqs, WeatherClient.FORECAST_MAX_AGE_MS, false) { }
            fail("expected NO_NETWORK")
        } catch (e: WeatherException) {
            assertEquals(WeatherException.Kind.NO_NETWORK, e.kind)
        }
        Unit
    }

    @Test
    fun api_error_is_reported_with_reason() = runBlocking {
        val fake = Fake { req -> response(req, 400, """{"error":true,"reason":"Date is out of range"}""") }
        try {
            WeatherClient(client(fake), tmp()).fetchAll(forecastReqs, null, false) { }
            fail("expected API error")
        } catch (e: WeatherException) {
            assertEquals(WeatherException.Kind.API, e.kind)
            assertTrue(e.message!!, e.message!!.contains("out of range"))
        }
        assertEquals("a 400 is not retried", 1, fake.calls)
        Unit
    }

    @Test
    fun http_200_with_error_body_is_an_error_and_not_cached() = runBlocking {
        val dir = tmp()
        val fake = Fake { req -> response(req, 200, """{"error":true,"reason":"Invalid parameter"}""") }
        try {
            WeatherClient(client(fake), dir).fetchAll(forecastReqs, null, false) { }
            fail("expected API error")
        } catch (e: WeatherException) {
            assertEquals(WeatherException.Kind.API, e.kind)
        }
        assertEquals(0, dir.listFiles()?.count { it.name.startsWith("wx-") } ?: 0)
        Unit
    }

    @Test
    fun archive_responses_are_cached_forever_and_service_picks_the_api() = runBlocking {
        val seenUrls = ArrayList<String>()
        val fake = Fake { req ->
            seenUrls += req.url.toString()
            response(req, 200, sliceFor(req))
        }
        var now = System.currentTimeMillis()
        val service = WeatherService(WeatherClient(client(fake), tmp()) { now }, pkg)
        val today = LocalDate.of(2025, 9, 29)

        // a window far in the past -> archive API with 60 days before the first day
        val first = LocalDate.of(2025, 7, 1)
        val last = LocalDate.of(2025, 7, 7)
        val b1 = service.history(first, last, today) { }
        assertEquals(WeatherSource.ERA5, b1.source)
        assertEquals(3, seenUrls.size)
        assertTrue(seenUrls.all { it.startsWith("https://archive-api.open-meteo.com/v1/archive?") })
        val (start, end) = HistoryPlan.archiveRange(first, last, today, pkg.historyDaysNeeded)
        assertTrue(!start.isAfter(first.minusDays(60)))
        assertTrue(!end.isBefore(last))
        assertTrue(seenUrls[0].contains("start_date=$start"))
        assertTrue(seenUrls[0].contains("end_date=$end"))
        assertTrue(seenUrls[0].contains("models=era5"))

        now += 400L * 24 * 60 * 60 * 1000 // a year later: still cached
        val b2 = service.history(first, last, today) { }
        assertEquals(3, seenUrls.size)
        assertTrue(b2.fromCache)

        // the next day shares the blocks (same requests) unless it crosses a block boundary
        val nextRange = HistoryPlan.archiveRange(first.plusDays(1), last.plusDays(1), today, pkg.historyDaysNeeded)
        if (nextRange == (start to end)) {
            service.history(first.plusDays(1), last.plusDays(1), today) { }
            assertEquals("same blocks -> no new download", 3, seenUrls.size)
        }

        // the last days -> forecast API (archive is too far behind)
        seenUrls.clear()
        val b3 = service.history(today.minusDays(4), today, today) { }
        assertEquals(WeatherSource.FORECAST, b3.source)
        assertTrue(seenUrls.all { it.startsWith("https://api.open-meteo.com/v1/forecast?") })
        Unit
    }
}
