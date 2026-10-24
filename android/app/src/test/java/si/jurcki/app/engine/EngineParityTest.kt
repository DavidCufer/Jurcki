package si.jurcki.app.engine

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import si.jurcki.app.engine.TestData.assertClose
import si.jurcki.app.engine.TestData.assertMatrixClose
import si.jurcki.app.engine.TestData.doubles
import si.jurcki.app.engine.TestData.matrix
import kotlin.math.floor

/**
 * Checks the Kotlin engine against the vectors produced by the Python lab
 * (the json files in `android/app/src/test/resources/parity/`, docs/APP_ENGINE.md section 11).
 */
class EngineParityTest {

    private val tolSeries = 1e-5
    private val tolFeaturesCoarse = 1e-4
    private val tolFeaturesPixel = 1e-3
    private val tolTrigger = 1e-4
    private val tolScore = 2e-4

    // ---- geometry ------------------------------------------------------------------------------

    @Test
    fun geometry_lonlat_to_pixel_and_back() {
        val v = TestData.json("parity/geometry.json")
        val geo = RasterGeo.fromJson(v.getJSONObject("raster"))
        val tol = v.getJSONObject("tolerance")
        val tolPx = tol.getDouble("pixel_rowcol")
        val tolDeg = tol.getDouble("lonlat_deg")

        // The package carries the same raster definition.
        val pg = TestData.pkg.geo
        assertEquals(geo.nx, pg.nx)
        assertEquals(geo.ny, pg.ny)
        assertClose("x0", geo.x0, pg.x0, 1e-6)
        assertClose("yTop", geo.yTop, pg.yTop, 1e-6)

        val g = Geometry(geo)
        val pts = v.getJSONArray("lonlat_to_pixel")
        assertTrue(pts.length() >= 10)
        for (i in 0 until pts.length()) {
            val p = pts.getJSONObject(i)
            val name = p.getString("name")
            val lon = p.getDouble("lon")
            val lat = p.getDouble("lat")
            val row = g.row(lat)
            val col = g.col(lon)
            assertClose("$name row", p.getDouble("row"), row, tolPx)
            assertClose("$name col", p.getDouble("col"), col, tolPx)
            if (p.isNull("pixel")) {
                // outside the raster (or exactly on its far edge)
                assertTrue(
                    "$name is outside",
                    row < 0 || col < 0 || row >= geo.ny - tolPx || col >= geo.nx - tolPx,
                )
            } else {
                val pixel = p.getJSONArray("pixel")
                assertEquals("$name pixel row", pixel.getInt(0), floor(row).toInt())
                assertEquals("$name pixel col", pixel.getInt(1), floor(col).toInt())
                assertEquals("$name index", pixel.getInt(0) * geo.nx + pixel.getInt(1), g.indexOf(lon, lat))
            }
        }

        val centres = v.getJSONArray("pixel_centres")
        assertTrue(centres.length() >= 4)
        for (i in 0 until centres.length()) {
            val c = centres.getJSONObject(i)
            val r = c.getInt("row")
            val cc = c.getInt("col")
            assertClose("lon($cc)", c.getDouble("lon"), g.lonOf(cc), tolDeg)
            assertClose("lat($r)", c.getDouble("lat"), g.latOf(r), tolDeg)
            // and back: the centre of a pixel maps to (r + 0.5, c + 0.5)
            assertClose("row back", r + 0.5, g.row(g.latOf(r)), 1e-6)
            assertClose("col back", cc + 0.5, g.col(g.lonOf(cc)), 1e-6)
        }
    }

    // ---- daily assembly, coarse features, trigger (small block) ----------------------------------

    @Test
    fun weather_small_daily_features_trigger() {
        val v = TestData.json("parity/weather_small.json")
        val lat = v.getJSONObject("lattice")
        val nLat = lat.getInt("n_lat")
        val nLon = lat.getInt("n_lon")
        val pts = v.getJSONArray("points")
        val elev = DoubleArray(pts.length()) { pts.getJSONArray(it).getDouble(2) }
        val params = TriggerParams.fromJson(v.getJSONObject("params"))
        val ex = v.getJSONObject("expected")

        // Through the string path, like the app: the body of a request is a JSON array.
        val locations = OpenMeteo.parseLocations(v.getJSONArray("responses").toString())
        val dw = OpenMeteo.assembleDaily(locations, nLat, nLon, elev)

        assertEquals(TestData.strings(ex.getJSONArray("dates")), dw.dates)
        for (k in 0 until nLat * nLon) assertClose("orog[$k]", ex.getJSONArray("orography").getDouble(k), dw.orog[k], 1e-9)

        val daily = ex.getJSONObject("daily")
        assertMatrixClose("tmean", matrix(daily.getJSONArray("tmean")), dw.tmean, tolSeries)
        assertMatrixClose("tmax", matrix(daily.getJSONArray("tmax")), dw.tmax, tolSeries)
        assertMatrixClose("tmin", matrix(daily.getJSONArray("tmin")), dw.tmin, tolSeries)
        assertMatrixClose("precip", matrix(daily.getJSONArray("precip")), dw.precip, tolSeries)

        val f = coarseFeatures(dw, params)
        val ef = ex.getJSONObject("features")
        assertMatrixClose("t20", matrix(ef.getJSONArray("t20")), f.t20, tolFeaturesCoarse)
        assertMatrixClose("p26", matrix(ef.getJSONArray("p26")), f.p26, tolFeaturesCoarse)
        assertMatrixClose("pulse", matrix(ef.getJSONArray("pulse")), f.pulse, tolFeaturesCoarse)
        assertMatrixClose("tmax10", matrix(ef.getJSONArray("tmax10")), f.tmax10, tolFeaturesCoarse)
        assertMatrixClose("tmin7", matrix(ef.getJSONArray("tmin7")), f.tmin7, tolFeaturesCoarse)

        val et = matrix(ex.getJSONArray("trigger"))
        for (k in et.indices) {
            for (t in et[k].indices) {
                val got = triggerOf(f.t20[k][t], f.p26[k][t], f.pulse[k][t], f.tmax10[k][t], f.tmin7[k][t], params)
                assertClose("trigger[$k][$t]", et[k][t], got, tolTrigger)
            }
        }
    }

    // ---- requests --------------------------------------------------------------------------------

    @Test
    fun forecast_requests_match_vector_and_model_json() {
        val pkg = TestData.pkg
        val built = OpenMeteo.forecastRequests(pkg.lattice, pkg.forecastPastDays, pkg.forecastDays, pkg.batchSize)
        assertEquals(3, built.size)

        val full = TestData.json("parity/weather_full.json").getJSONArray("requests")
        val model = pkg.json.getJSONObject("weather").getJSONObject("forecast_recent").getJSONArray("requests")
        for (src in listOf(full, model)) {
            assertEquals(built.size, src.length())
            for (i in built.indices) {
                val e = src.getJSONObject(i)
                assertEquals(e.getString("url"), built[i].url)
                val p = e.getJSONObject("params")
                assertEquals("params of request $i", paramMap(p), built[i].params)
            }
        }
        assertEquals(50, built[0].params.getValue("latitude").split(",").size)
        assertEquals(20, built[2].params.getValue("latitude").split(",").size)
        assertTrue(built[0].fullUrl().startsWith("https://api.open-meteo.com/v1/forecast?"))
        assertTrue(!built[0].fullUrl().contains("hourly"))
    }

    @Test
    fun archive_requests_match_model_json_example() {
        val pkg = TestData.pkg
        val example = pkg.json.getJSONObject("weather").getJSONObject("history").getJSONArray("requests_example")
        val first = example.getJSONObject(0).getJSONObject("params")
        val start = java.time.LocalDate.parse(first.getString("start_date"))
        val end = java.time.LocalDate.parse(first.getString("end_date"))
        val built = OpenMeteo.archiveRequests(pkg.lattice, start, end, pkg.batchSize)
        assertEquals(example.length(), built.size)
        for (i in built.indices) {
            val e = example.getJSONObject(i)
            assertEquals(e.getString("url"), built[i].url)
            assertEquals(paramMap(e.getJSONObject("params")), built[i].params)
        }
        assertEquals("era5", built[0].params["models"])
    }

    private fun paramMap(p: JSONObject): Map<String, String> {
        val out = HashMap<String, String>()
        val it = p.keys()
        while (it.hasNext()) {
            val k = it.next()
            out[k] = p.get(k).toString()
        }
        return out
    }

    // ---- full lattice: downscaling, trigger, score, end to end -----------------------------------

    @Test
    fun cells_downscaled_features_trigger_score() {
        val v = TestData.json("parity/cells.json")
        val engine = TestData.engine
        val pkg = engine.pkg
        val weather = TestData.json("parity/" + v.getString("weather_file"))
        val params = TriggerParams.fromJson(v.getJSONObject("params"))
        assertEquals("package parameters = vector parameters", pkg.params, params)
        assertClose("reference", v.getDouble("reference"), pkg.reference, 1e-12)

        val dw = OpenMeteo.assembleDaily(
            OpenMeteo.parseLocations(weather.getJSONArray("responses").toString()),
            pkg.lattice.nLat, pkg.lattice.nLon, pkg.lattice.pointElev,
        )
        val field = engine.field(dw)
        val cells = v.getJSONArray("cells")
        val days = TestData.strings(v.getJSONArray("days"))
        assertEquals(2, days.size)

        for (day in days) {
            val dayIdx = field.indexOf(day)
            assertTrue("$day in the weather window", dayIdx >= 0)
            val ex = v.getJSONObject("expected").getJSONObject(day)
            assertClose("season $day", ex.getDouble("season_factor"), engine.seasonFactor(day), 1e-12)
            val ef = ex.getJSONObject("features")
            for (i in 0 until cells.length()) {
                val c = cells.getJSONObject(i)
                val name = c.getString("name")
                val pf = engine.pixelFeatures(field, dayIdx, c.getDouble("lon"), c.getDouble("lat"), c.getDouble("elev"))
                assertClose("$day $name t20", ef.getJSONArray("t20").getDouble(i), pf.t20, tolFeaturesPixel)
                assertClose("$day $name p26", ef.getJSONArray("p26").getDouble(i), pf.p26, tolFeaturesPixel)
                assertClose("$day $name pulse", ef.getJSONArray("pulse").getDouble(i), pf.pulse, tolFeaturesPixel)
                assertClose("$day $name tmax10", ef.getJSONArray("tmax10").getDouble(i), pf.tmax10, tolFeaturesPixel)
                assertClose("$day $name tmin7", ef.getJSONArray("tmin7").getDouble(i), pf.tmin7, tolFeaturesPixel)
                val trig = engine.trigger(pf)
                assertClose("$day $name trigger", ex.getJSONArray("trigger").getDouble(i), trig, tolTrigger)
                val season = engine.seasonFactor(day)
                val hab = c.getDouble("habitat")
                assertClose("$day $name raw", ex.getJSONArray("raw").getDouble(i), hab * season * trig, tolTrigger)
                assertClose("$day $name score", ex.getJSONArray("score").getDouble(i), engine.scoreOf(hab, season, trig), tolScore)
            }
        }
    }

    @Test
    fun cells_raster_loop_matches_vectors_at_package_pixels() {
        val v = TestData.json("parity/cells.json")
        val engine = TestData.engine
        val pkg = engine.pkg
        val nx = pkg.geo.nx
        val weather = TestData.json("parity/" + v.getString("weather_file"))
        val dw = OpenMeteo.assembleDaily(
            OpenMeteo.parseLocations(weather.getJSONArray("responses").toString()),
            pkg.lattice.nLat, pkg.lattice.nLon, pkg.lattice.pointElev,
        )
        val field = engine.field(dw)
        val cells = v.getJSONArray("cells")
        for (day in TestData.strings(v.getJSONArray("days"))) {
            val dayIdx = field.indexOf(day)
            val ex = v.getJSONObject("expected").getJSONObject(day)
            val scores = engine.computeDay(field, dayIdx)
            val dayMap = engine.dayMap(field, dayIdx)
            var checked = 0
            for (i in 0 until cells.length()) {
                val c = cells.getJSONObject(i)
                if (c.isNull("row")) continue
                val idx = c.getInt("row") * nx + c.getInt("col")
                val name = c.getString("name")
                // the raster itself: habitat and elevation decode
                assertClose("$name habitat raster", c.getDouble("habitat"), pkg.habitat[idx].toDouble(), 1e-6)
                assertClose("$name elev raster", c.getDouble("elev"), pkg.elev[idx].toDouble(), 1e-9)
                assertEquals("$name on land", 1, pkg.mask[idx].toInt())
                // the loop
                assertClose("$day $name loop trigger", ex.getJSONArray("trigger").getDouble(i), scores.trigger[idx].toDouble(), tolTrigger)
                assertClose("$day $name loop score", ex.getJSONArray("score").getDouble(i), scores.score[idx].toDouble(), tolScore)
                // the colour of that pixel
                assertEquals("$name argb", pkg.colorMap.argb(scores.score[idx].toDouble()), dayMap.argb[idx])
                checked++
            }
            assertTrue("checked $checked real cells", checked >= 15)
        }
    }

    // ---- score scaling, colours, hotspots ------------------------------------------------------------

    @Test
    fun score_scaling() {
        val v = TestData.json("parity/misc.json")
        val engine = TestData.engine
        assertClose("reference", v.getDouble("reference"), engine.pkg.reference, 1e-12)
        val items = v.getJSONArray("scoring")
        for (i in 0 until items.length()) {
            val o = items.getJSONObject(i)
            val h = o.getDouble("habitat")
            val s = o.getDouble("season")
            val t = o.getDouble("trigger")
            assertClose("raw[$i]", o.getDouble("raw"), h * s * t, 1e-6)
            assertClose("score[$i]", o.getDouble("score"), engine.scoreOf(h, s, t), tolScore)
        }
    }

    @Test
    fun colours_lut_and_rule() {
        val v = TestData.json("parity/misc.json")
        val cm = TestData.pkg.colorMap

        // LUT of the package == LUT of the vector
        val lut = v.getJSONArray("lut")
        assertEquals(64, lut.length())
        for (i in 0 until 64) {
            assertEquals("lut[$i]", rgbaToArgb(lut.getJSONArray(i)), cm.lutArgb[i])
        }

        // score -> colour
        val colours = v.getJSONArray("colours")
        for (i in 0 until colours.length()) {
            val o = colours.getJSONObject(i)
            val rgba = o.getJSONArray("rgba")
            val score = if (o.isNull("score")) Double.NaN else o.getDouble("score")
            val argb = cm.argb(score)
            if (score.isNaN()) {
                assertEquals("NaN is transparent", 0, argb ushr 24)
            } else {
                assertEquals("colour of $score", rgbaToArgb(rgba), argb)
            }
        }

        // the documented rule rebuilds the table exactly (legend stops -> 64 levels)
        val rebuilt = rebuildLut(cm)
        for (i in 0 until 64) assertEquals("rebuilt lut[$i]", cm.lutArgb[i], rebuilt[i])

        // rint = half to even: 0.5 * 63 = 31.5 -> level 32; 2.5/63 -> level 2
        assertEquals(32, cm.level(0.5))
        assertEquals(2, cm.level(2.5 / 63.0))
        assertEquals(0, cm.level(-0.3))
        assertEquals(63, cm.level(1.7))
    }

    private fun rgbaToArgb(a: org.json.JSONArray): Int =
        (a.getInt(3) shl 24) or (a.getInt(0) shl 16) or (a.getInt(1) shl 8) or a.getInt(2)

    /** The colour rule of docs/APP_ENGINE.md section 8, written independently of the LUT in model.json. */
    private fun rebuildLut(cm: ColorMap): IntArray {
        val n = cm.levels
        val step = 1.0 / (n - 1)
        val out = IntArray(n)
        for (i in 0 until n) {
            val s = if (i == n - 1) 1.0 else i * step
            var rgb = 0
            for (shift in intArrayOf(16, 8, 0)) {
                val xs = DoubleArray(cm.legend.size) { cm.legend[it].value }
                val ys = DoubleArray(cm.legend.size) { ((cm.legend[it].argb shr shift) and 0xFF).toDouble() }
                rgb = (rgb shl 8) or Math.rint(interp(s, xs, ys)).toInt()
            }
            val alpha = if (s < cm.minScore) {
                0.0
            } else {
                val f = ((s - cm.minScore) / (cm.alphaFullAt - cm.minScore)).coerceIn(0.0, 1.0)
                f * cm.alphaMax
            }
            val a255 = Math.rint(alpha * 255).toInt()
            out[i] = (a255 shl 24) or rgb
        }
        return out
    }

    /** numpy.interp for increasing xs. */
    private fun interp(x: Double, xs: DoubleArray, ys: DoubleArray): Double {
        if (x <= xs.first()) return ys.first()
        if (x >= xs.last()) return ys.last()
        var j = 0
        while (j < xs.size - 2 && x >= xs[j + 1]) j++
        val slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
        return slope * (x - xs[j]) + ys[j]
    }

    @Test
    fun hotspots_selection() {
        val v = TestData.json("parity/misc.json").getJSONObject("hotspots")
        val ny = v.getInt("ny")
        val nx = v.getInt("nx")
        val score = FloatArray(nx * ny)
        val hab = FloatArray(nx * ny)
        val trig = FloatArray(nx * ny)
        val mask = ByteArray(nx * ny)
        val s = matrix(v.getJSONArray("score"))
        val h = matrix(v.getJSONArray("habitat"))
        val t = matrix(v.getJSONArray("trigger"))
        val m = v.getJSONArray("mask")
        for (r in 0 until ny) {
            for (c in 0 until nx) {
                score[r * nx + c] = s[r][c].toFloat()
                hab[r * nx + c] = h[r][c].toFloat()
                trig[r * nx + c] = t[r][c].toFloat()
                mask[r * nx + c] = m.getJSONArray(r).getInt(c).toByte()
            }
        }
        val picks = HotspotPicker.pickIndices(
            score, hab, trig, mask, nx, ny,
            v.getDouble("min_score"), v.getDouble("min_sep_px"), v.getInt("max_n"),
        )
        val expected = v.getJSONArray("expected_picks")
        assertEquals("number of picks", expected.length(), picks.size)
        for (i in picks.indices) {
            val e = expected.getJSONArray(i)
            assertEquals("pick $i row", e.getInt(0), picks[i] / nx)
            assertEquals("pick $i col", e.getInt(1), picks[i] % nx)
        }
        // the model's configuration is the one of the vector
        val cfg = TestData.pkg.hotspotConfig
        assertClose("min_score", v.getDouble("min_score"), cfg.minScore, 1e-12)
        assertClose("min_sep_px", v.getDouble("min_sep_px"), cfg.minSepPx, 1e-12)
        assertEquals(v.getInt("max_n"), cfg.maxN)
    }

    @Test
    fun hotspots_of_a_full_day_are_land_pixels_in_rank_order() {
        val v = TestData.json("parity/cells.json")
        val engine = TestData.engine
        val pkg = engine.pkg
        val weather = TestData.json("parity/" + v.getString("weather_file"))
        val dw = OpenMeteo.assembleDaily(
            OpenMeteo.parseLocations(weather.getJSONArray("responses").toString()),
            pkg.lattice.nLat, pkg.lattice.nLon, pkg.lattice.pointElev,
        )
        val field = engine.field(dw)
        val map = engine.dayMap(field, field.indexOf("2025-09-30"))
        assertTrue("some hotspots in September", map.hotspots.size > 50)
        assertTrue(map.hotspots.size <= pkg.hotspotConfig.maxN)
        var prev = Double.MAX_VALUE
        for (h in map.hotspots) {
            assertEquals(1, pkg.mask[h.row * pkg.geo.nx + h.col].toInt())
            assertClose("lon", engine.geometry.lonOf(h.col), h.lon, 1e-12)
            assertClose("lat", engine.geometry.latOf(h.row), h.lat, 1e-12)
            val rank = h.score + 0.001 * h.habitat * h.trigger
            assertTrue("rank order", rank <= prev + 1e-6)
            assertTrue("rank >= min", rank >= pkg.hotspotConfig.minScore - 1e-6)
            prev = rank
        }
        // minimum separation
        for (i in map.hotspots.indices) {
            for (j in i + 1 until map.hotspots.size) {
                val a = map.hotspots[i]
                val b = map.hotspots[j]
                val d = Math.hypot((a.row - b.row).toDouble(), (a.col - b.col).toDouble())
                assertTrue("hotspots $i and $j are ${d} px apart", d >= pkg.hotspotConfig.minSepPx)
            }
        }
    }
}
