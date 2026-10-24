package si.jurcki.app.engine

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import si.jurcki.app.StatsData
import si.jurcki.app.engine.TestData.assertClose
import java.time.LocalDate

/** Checks that do not need parity vectors: edge cases, package decoding, helpers. */
class EngineUnitTest {

    // ---- gap filling ----------------------------------------------------------------------------

    @Test
    fun fill_gaps_interior_leading_trailing() {
        val nan = Double.NaN
        val a = doubleArrayOf(nan, nan, 2.0, nan, nan, nan, 6.0, 7.0, nan, nan)
        assertTrue(OpenMeteo.fillGaps(a))
        val expected = doubleArrayOf(2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0)
        for (i in a.indices) assertClose("a[$i]", expected[i], a[i], 1e-12)
    }

    @Test
    fun fill_gaps_without_data_is_left_alone() {
        val a = doubleArrayOf(Double.NaN, Double.NaN)
        assertFalse(OpenMeteo.fillGaps(a))
        assertTrue(a[0].isNaN())
    }

    @Test
    fun missing_point_copies_the_nearest_valid_point() {
        // 2 x 3 lattice, point 0 has no data; nearest valid: point 1 (distance 1), not point 3 (distance 1, later)
        val series = Array(6) { k -> DoubleArray(3) { k * 10.0 + it } }
        series[0] = DoubleArray(3) { Double.NaN }
        val valid = booleanArrayOf(false, true, true, true, true, true)
        OpenMeteo.fillMissingPoints(series, valid, 2, 3)
        assertEquals(10.0, series[0][0], 0.0)
        assertEquals(12.0, series[0][2], 0.0)
    }

    @Test
    fun daily_assembly_uses_union_of_days_and_nulls() {
        // two points, point 1 starts one day later and has a null; elevation falls back to the request value
        fun loc(days: List<String>, mean: String, elevation: String?): JSONObject =
            JSONObject(
                """{${if (elevation != null) "\"elevation\":$elevation," else ""}"daily":{"time":${days.map { "\"$it\"" }},
                "temperature_2m_mean":$mean,"temperature_2m_max":$mean,"temperature_2m_min":$mean,"precipitation_sum":$mean}}""",
            )
        val l0 = loc(listOf("2025-09-01", "2025-09-02", "2025-09-03"), "[1.0,2.0,3.0]", "100.0")
        val l1 = loc(listOf("2025-09-02", "2025-09-03"), "[null,5.0]", null)
        val w = OpenMeteo.assembleDaily(listOf(l0, l1), 1, 2, doubleArrayOf(111.0, 222.0))
        assertEquals(listOf("2025-09-01", "2025-09-02", "2025-09-03"), w.dates)
        assertEquals(100.0, w.orog[0], 0.0)
        assertEquals(222.0, w.orog[1], 0.0)
        // point 1: day 1 missing, day 2 null -> carried back from the first valid value (5.0)
        for (t in 0..2) assertEquals(5.0, w.tmean[1][t], 1e-12)
    }

    @Test
    fun parse_locations_single_object_and_error() {
        val one = OpenMeteo.parseLocations("""{"daily":{"time":["2025-09-01"]}}""")
        assertEquals(1, one.size)
        try {
            OpenMeteo.parseLocations("""{"error":true,"reason":"Too many requests"}""")
            fail("error body must throw")
        } catch (e: WeatherException) {
            assertEquals(WeatherException.Kind.API, e.kind)
            assertTrue(e.message!!.contains("Too many"))
        }
        try {
            OpenMeteo.parseLocations("not json")
            fail("garbage must throw")
        } catch (e: WeatherException) {
            assertEquals(WeatherException.Kind.BAD_RESPONSE, e.kind)
        }
    }

    // ---- features ----------------------------------------------------------------------------------

    @Test
    fun trailing_windows_are_truncated_at_the_start() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        assertEquals(listOf(1.0, 1.5, 2.0, 3.0), rollMean(x, 3).toList())
        assertEquals(listOf(1.0, 3.0, 6.0, 9.0), rollSum(x, 3).toList())
        assertEquals(listOf(1.0, 2.0, 3.0, 4.0), rollMax(x, 3).toList())
        assertEquals(listOf(1.0, 1.0, 1.0, 2.0), rollMin(x, 3).toList())
    }

    @Test
    fun pulse_is_lagged_flag_of_a_soaking_rain() {
        val p = TestData.pkg.params // lag 14..19, 3-day sum >= 10.46
        val precip = DoubleArray(60)
        precip[20] = 12.0 // soaking rain ends on day 20 (3-day sum 12 from day 20 on until day 22)
        val pulse = pulseSeries(precip, p)
        // soak flags on days 20, 21, 22; recent = max over the previous 6 days; pulse[t] = recent[t - 14]
        assertEquals(0.0, pulse[33], 0.0)
        assertEquals(1.0, pulse[34], 0.0)
        assertEquals(1.0, pulse[41], 0.0)
        assertEquals(0.0, pulse[42], 0.0)
        assertEquals(0.0, pulse[10], 0.0)
    }

    @Test
    fun trigger_is_clamped_and_responds_to_heat_and_frost() {
        val p = TestData.pkg.params
        val base = triggerOf(p.tOpt, 200.0, 1.0, 20.0, 10.0, p)
        assertTrue(base in 0.0..1.0)
        assertTrue("heat lowers the trigger", triggerOf(p.tOpt, 200.0, 1.0, 40.0, 10.0, p) < base)
        assertTrue("frost lowers the trigger", triggerOf(p.tOpt, 200.0, 1.0, 20.0, -5.0, p) < base)
        assertEquals(0.0, triggerOf(p.tOpt, 0.0, 0.0, 20.0, 10.0, p), 0.0)
    }

    // ---- package ---------------------------------------------------------------------------------------

    @Test
    fun package_rasters_are_consistent() {
        val pkg = TestData.pkg
        val n = pkg.geo.nx * pkg.geo.ny
        assertEquals(1009, pkg.geo.nx)
        assertEquals(661, pkg.geo.ny)
        assertEquals(n, pkg.habitat.size)
        assertEquals(n, pkg.elev.size)
        assertEquals(n, pkg.mask.size)
        var land = 0
        var habitatOutside = 0
        for (i in 0 until n) {
            if (pkg.mask[i].toInt() == 1) {
                land++
                assertTrue(pkg.habitat[i] in 0f..1f)
            } else if (pkg.habitat[i] != 0f) {
                habitatOutside++
            }
        }
        assertTrue("land pixels: $land", land in 250_000..450_000)
        assertEquals("nodata habitat outside the mask is 0", 0, habitatOutside)
        // elevation of known places (cells.json): Triglav area is high, Koper is low
        val cells = TestData.json("parity/cells.json").getJSONArray("cells")
        for (i in 0 until cells.length()) {
            val c = cells.getJSONObject(i)
            if (c.isNull("row")) continue
            val idx = c.getInt("row") * pkg.geo.nx + c.getInt("col")
            if (c.getString("name") == "Triglav") assertTrue(pkg.elev[idx] > 1500f)
            if (c.getString("name") == "Koper") assertTrue(pkg.elev[idx] < 200f)
        }
        assertEquals(120, pkg.lattice.size)
        assertEquals(62, pkg.forecastPastDays)
        assertEquals(7, pkg.forecastDays)
        assertEquals(2, pkg.shownPast)
        assertEquals(6, pkg.shownAhead)
        assertEquals(0.0, pkg.seasonFactor[1], 0.0)
        assertEquals(1.0, pkg.seasonFactor[9], 0.0)
        assertEquals(64, pkg.colorMap.levels)
    }

    @Test
    fun geometry_bounds_match_the_outer_pixel_edges() {
        val pkg = TestData.pkg
        val g = Geometry(pkg.geo)
        assertClose("west", pkg.geo.minLon, g.lonOf(0) - (g.lonOf(1) - g.lonOf(0)) / 2, 1e-5)
        assertClose("north", pkg.geo.maxLat, g.latOf(0) + (g.latOf(0) - g.latOf(1)) / 2, 1e-4)
        assertEquals(-1, g.indexOf(10.0, 46.0))
        assertTrue(g.indexOf(14.5, 46.05) >= 0)
    }

    // ---- rendering, hotspots ---------------------------------------------------------------------------------

    @Test
    fun render_makes_nan_and_zero_transparent() {
        val cm = TestData.pkg.colorMap
        val argb = Render.toArgb(floatArrayOf(Float.NaN, 0f, 0.5f, 1f), cm)
        assertEquals(0, argb[0])
        assertEquals(0, argb[1] ushr 24)
        assertTrue(argb[2] ushr 24 > 200)
        assertEquals(cm.lutArgb[63], argb[3])
    }

    @Test
    fun hotspot_separation_is_strictly_less_than_six_pixels() {
        val nx = 40
        val ny = 10
        val score = FloatArray(nx * ny)
        val hab = FloatArray(nx * ny) { 1f }
        val trig = FloatArray(nx * ny) { 1f }
        val mask = ByteArray(nx * ny) { 1 }
        fun put(r: Int, c: Int, v: Float) {
            score[r * nx + c] = v
        }
        put(5, 5, 0.9f)
        put(5, 11, 0.8f) // exactly 6 px from the first: kept
        put(5, 16, 0.7f) // 5 px from the second: dropped
        put(5, 30, 0.6f)
        val picks = HotspotPicker.pickIndices(score, hab, trig, mask, nx, ny, 0.15, 6.0, 300)
        assertEquals(listOf(5 * nx + 5, 5 * nx + 11, 5 * nx + 30), picks.toList())
        val limited = HotspotPicker.pickIndices(score, hab, trig, mask, nx, ny, 0.15, 6.0, 2)
        assertEquals(2, limited.size)
    }

    @Test
    fun hotspot_ties_break_by_row_then_column() {
        val nx = 30
        val ny = 30
        val score = FloatArray(nx * ny)
        val hab = FloatArray(nx * ny) { 1f }
        val trig = FloatArray(nx * ny) { 1f }
        val mask = ByteArray(nx * ny) { 1 }
        score[20 * nx + 3] = 1f
        score[4 * nx + 25] = 1f
        score[4 * nx + 3] = 1f
        val picks = HotspotPicker.pickIndices(score, hab, trig, mask, nx, ny, 0.15, 6.0, 300)
        assertEquals(listOf(4 * nx + 3, 4 * nx + 25, 20 * nx + 3), picks.toList())
    }

    // ---- history planning -------------------------------------------------------------------------------------

    @Test
    fun history_plan_windows_and_sources() {
        val today = LocalDate.of(2026, 10, 24)
        val (f1, l1) = HistoryPlan.window(LocalDate.of(2025, 9, 10), today)
        assertEquals(LocalDate.of(2025, 9, 7), f1)
        assertEquals(LocalDate.of(2025, 9, 13), l1)
        assertTrue(HistoryPlan.useArchive(l1, today))
        val (f2, l2) = HistoryPlan.window(today.minusDays(1), today)
        assertEquals(today.minusDays(4), f2)
        assertEquals(today, l2) // never after today
        assertFalse(HistoryPlan.useArchive(l2, today))
        assertFalse(HistoryPlan.useArchive(today.minusDays(6), today))
        assertTrue(HistoryPlan.useArchive(today.minusDays(7), today))

        // archive ranges: whole 14-day blocks, 60 days lead-in, never later than today - 7
        val (rs, re) = HistoryPlan.archiveRange(f1, l1, today, 60)
        assertTrue(!rs.isAfter(f1.minusDays(60)))
        assertTrue(!re.isBefore(l1))
        assertEquals(HistoryPlan.archiveRange(f1, l1, today, 60), HistoryPlan.archiveRange(f1.plusDays(0), l1, today, 60))
        val (_, reRecent) = HistoryPlan.archiveRange(today.minusDays(20), today.minusDays(14), today, 60)
        assertTrue(!reRecent.isAfter(today.minusDays(7)))

        val all = (0..10).map { today.minusDays(5).plusDays(it.toLong()).toString() }
        val shown = HistoryPlan.forecastDates(all, today, 2, 6)
        assertEquals(today.minusDays(2).toString(), shown.first())
        assertEquals(today.plusDays(5).toString(), shown.last())
    }

    // ---- finds and stats -----------------------------------------------------------------------------------------

    @Test
    fun finds_parse_and_near() {
        val finds = FindsParser.parse(TestData.assetText("finds.json"))
        assertEquals(248, finds.size)
        val f = finds.first { it.date == "2010-08-12" }
        assertEquals(15.04, f.lon, 1e-9)
        assertNotNull(f.pSeasonTrigger)
        assertEquals(0.833, f.pSeasonTrigger!!, 1e-9)
        assertEquals(null, f.precisionM)
        val near = FindsParser.near(finds, LocalDate.of(2010, 8, 12), 7)
        assertTrue(near.any { it.date == "2010-08-12" })
        assertTrue(near.all { Math.abs(it.epochDay - LocalDate.of(2010, 8, 12).toEpochDay()) <= 7 })
        assertEquals(near.indices.toList(), near.map { it.idx })
        // finds outside the backtest have no percentiles
        assertTrue(finds.any { it.pSeasonTrigger == null })
        assertEquals(17, topPercent(0.833))
        assertEquals(1, topPercent(1.0))
        assertEquals(50, topPercent(0.5))
    }

    @Test
    fun stats_json_parses() {
        val s = StatsData.parse(TestData.assetText("stats.json"))
        val hab = s.habitatUnit("si_all")!!
        assertEquals(0.4645, hab.prior!!.auc, 1e-9)
        assertEquals(0.6202, hab.model!!.auc, 1e-9)
        assertEquals(229, hab.model!!.n)
        assertEquals(10, s.habitatBlockKm)
        val tuned = s.triggerRow("tuned_loyo:comb_tol")!!
        assertEquals(0.7095, tuned.tolerant!!.auc, 1e-9)
        assertEquals(0.6615, s.triggerRow("season_only:comb_tol")!!.tolerant!!.auc, 1e-9)
        assertEquals(17, s.perYear.size)
        assertEquals(2010, s.perYear.first().year)
        assertEquals(11, s.params.size)
        assertEquals("t_opt", s.params.first().name)
        assertEquals("17.61", s.params.first().tuned)
        assertEquals("14-19", s.params.first { it.name == "pulse_lag" }.tuned)
        assertEquals(267, s.weekend!!.n)
        assertEquals(248, s.findsInBounds)
        assertEquals("AT", s.byCountry.first().label)
        assertEquals(6, s.glossary.size)
        assertEquals(3, s.gainTunedVsSeason.size)
        assertTrue(s.triggerTitle.text(true).startsWith("When"))
        assertTrue(s.triggerTitle.text(false).startsWith("Kdaj"))
    }
}
