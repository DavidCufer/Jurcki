package si.jurcki.app

import org.json.JSONArray
import org.json.JSONObject

/** Text in both languages (stats.json carries `sl` and `en`). */
data class Loc(val sl: String, val en: String) {
    fun text(english: Boolean): String = if (english && en.isNotEmpty()) en else sl
}

/** AUC and hit rates of one evaluation row (`n` = number of finds). */
data class AucRow(val n: Int, val auc: Double, val top10: Double, val top20: Double)

data class HabitatUnit(val key: String, val label: Loc, val prior: AucRow?, val model: AucRow?)

data class TriggerRow(val key: String, val label: Loc, val tolerant: AucRow?, val exact: AucRow?)

data class YearRow(
    val year: Int,
    val n: Int,
    val aucTuned: Double,
    val aucLiterature: Double,
    val aucSeason: Double,
    val top20Tuned: Double,
    val top20Literature: Double,
    val top20Season: Double,
)

data class ParamRow(
    val name: String,
    val unit: String,
    val literature: String,
    val tuned: String,
    val text: Loc,
)

data class WeekendEffect(val n: Int, val offDayShare: Double, val baselineShare: Double, val text: Loc)

data class CountItem(val label: String, val count: Int)

/**
 * The parsed `stats.json` ("Model / success" screen). Every accessor tolerates missing parts
 * (null / empty), so an older or newer stats.json never crashes the screen.
 */
class StatsData private constructor(
    val generated: String,
    val habitatTitle: Loc,
    val habitatDescription: Loc,
    val habitatBlockKm: Int,
    val habitatUnits: List<HabitatUnit>,
    val importance: List<Pair<String, Double>>,
    val triggerTitle: Loc,
    val triggerDescription: Loc,
    val triggerRows: List<TriggerRow>,
    val triggerNote: Loc?,
    val triggerFinds: Int,
    val gainTunedVsLiterature: List<Double>,
    val gainTunedVsSeason: List<Double>,
    val perYear: List<YearRow>,
    val perYearNote: Loc?,
    val params: List<ParamRow>,
    val weekend: WeekendEffect?,
    val findsInBounds: Int,
    val duplicatesRemoved: Int,
    val inBacktest: Int,
    val byCountry: List<CountItem>,
    val bySource: List<CountItem>,
    val bySpecies: List<CountItem>,
    val byYear: List<CountItem>,
    val dataNotes: List<Loc>,
    val glossary: List<Pair<String, Loc>>,
) {
    fun triggerRow(key: String): TriggerRow? = triggerRows.firstOrNull { it.key == key }

    /** Habitat row of a unit key (`si_all`, `si_dated`, ...). */
    fun habitatUnit(key: String): HabitatUnit? = habitatUnits.firstOrNull { it.key == key }

    companion object {
        fun parse(json: String): StatsData {
            val root = JSONObject(json)
            val hab = root.optJSONObject("habitat") ?: JSONObject()
            val trg = root.optJSONObject("trigger") ?: JSONObject()
            val data = root.optJSONObject("data") ?: JSONObject()

            // --- habitat
            val blockKm = hab.optDouble("block_km_main", 10.0).toInt()
            val cv = hab.optJSONObject("cv")?.optJSONObject("${blockKm}km")
            val finalVariant = hab.optString("final_variant", "full")
            val units = hab.optJSONObject("units")
            val habitatUnits = ArrayList<HabitatUnit>()
            for (key in listOf("si_all", "si_dated", "si_sfi", "foreign_dated")) {
                val prior = cv?.optJSONObject("prior")?.optJSONObject(key)?.let { aucRow(it) }
                val model = cv?.optJSONObject(finalVariant)?.optJSONObject(key)?.let { aucRow(it) }
                if (prior == null && model == null) continue
                habitatUnits += HabitatUnit(key, loc(units?.optJSONObject(key)) ?: Loc(key, key), prior, model)
            }
            val importance = ArrayList<Pair<String, Double>>()
            hab.optJSONObject("importance")?.let { imp ->
                val keys = imp.keys()
                while (keys.hasNext()) {
                    val k = keys.next()
                    importance += k to imp.optDouble(k, 0.0)
                }
            }
            importance.sortByDescending { it.second }

            // --- trigger
            val rows = ArrayList<TriggerRow>()
            val rowsArr = trg.optJSONArray("rows")
            if (rowsArr != null) {
                for (i in 0 until rowsArr.length()) {
                    val r = rowsArr.optJSONObject(i) ?: continue
                    rows += TriggerRow(
                        key = r.optString("key"),
                        label = loc(r) ?: Loc(r.optString("key"), r.optString("key")),
                        tolerant = r.optJSONObject("tolerant_3d")?.let { aucRow(it) },
                        exact = r.optJSONObject("exact_day")?.let { aucRow(it) },
                    )
                }
            }
            val mlNote = trg.optString("ml_note", "")
            val mlNoteEn = trg.optString("ml_note_en", "")
            val gain = trg.optJSONObject("gain_ci_auc")?.optJSONObject("values")

            // --- per year
            val years = ArrayList<YearRow>()
            val pyArr = root.optJSONArray("per_year")
            if (pyArr != null) {
                for (i in 0 until pyArr.length()) {
                    val y = pyArr.optJSONObject(i) ?: continue
                    years += YearRow(
                        year = y.optInt("year"),
                        n = y.optInt("n"),
                        aucTuned = y.optDouble("auc_tuned_loyo", Double.NaN),
                        aucLiterature = y.optDouble("auc_literature", Double.NaN),
                        aucSeason = y.optDouble("auc_season", Double.NaN),
                        top20Tuned = y.optDouble("hit_top20_tuned_loyo", Double.NaN),
                        top20Literature = y.optDouble("hit_top20_literature", Double.NaN),
                        top20Season = y.optDouble("hit_top20_season", Double.NaN),
                    )
                }
            }
            years.sortBy { it.year }

            // --- parameters
            val params = ArrayList<ParamRow>()
            val pArr = root.optJSONArray("params")
            if (pArr != null) {
                for (i in 0 until pArr.length()) {
                    val p = pArr.optJSONObject(i) ?: continue
                    params += ParamRow(
                        name = p.optString("name"),
                        unit = p.optString("unit"),
                        literature = valueText(p.opt("literature")),
                        tuned = valueText(p.opt("tuned")),
                        text = loc(p) ?: Loc("", ""),
                    )
                }
            }

            // --- weekend effect
            val we = root.optJSONObject("weekend_effect")?.let {
                WeekendEffect(
                    n = it.optInt("n"),
                    offDayShare = it.optDouble("off_day_share", Double.NaN),
                    baselineShare = it.optDouble("baseline_share", Double.NaN),
                    text = loc(it) ?: Loc("", ""),
                )
            }

            // --- data counts
            val notes = ArrayList<Loc>()
            loc(data.optJSONObject("habitat_training_note"))?.let { notes += it }
            loc(data.optJSONObject("backtest_note"))?.let { notes += it }

            // --- glossary (fixed order first)
            val glossary = ArrayList<Pair<String, Loc>>()
            val gl = root.optJSONObject("glossary")
            if (gl != null) {
                val order = listOf("auc", "top10", "top20", "tol", "loyo", "blocks")
                for (k in order) loc(gl.optJSONObject(k))?.let { glossary += k to it }
                val keys = gl.keys()
                while (keys.hasNext()) {
                    val k = keys.next()
                    if (k !in order) loc(gl.optJSONObject(k))?.let { glossary += k to it }
                }
            }

            return StatsData(
                generated = root.optString("generated", ""),
                habitatTitle = pair(hab.optJSONArray("title")) ?: Loc("Kje?", "Where?"),
                habitatDescription = pair(hab.optJSONArray("description")) ?: Loc("", ""),
                habitatBlockKm = blockKm,
                habitatUnits = habitatUnits,
                importance = importance,
                triggerTitle = pair(trg.optJSONArray("title")) ?: Loc("Kdaj?", "When?"),
                triggerDescription = pair(trg.optJSONArray("description")) ?: Loc("", ""),
                triggerRows = rows,
                triggerNote = if (mlNote.isNotEmpty()) Loc(mlNote, mlNoteEn) else null,
                triggerFinds = trg.optInt("n_finds"),
                gainTunedVsLiterature = doubles(gain?.optJSONArray("tuned_minus_literature")),
                gainTunedVsSeason = doubles(gain?.optJSONArray("tuned_minus_season")),
                perYear = years,
                perYearNote = loc(root.optJSONObject("per_year_note")),
                params = params,
                weekend = we,
                findsInBounds = data.optInt("finds_2010plus_in_bounds"),
                duplicatesRemoved = data.optInt("duplicates_removed"),
                inBacktest = data.optInt("in_backtest"),
                byCountry = counts(data.optJSONObject("by_country"), byValue = true),
                bySource = counts(data.optJSONObject("by_source"), byValue = true),
                bySpecies = counts(data.optJSONObject("by_species"), byValue = true),
                byYear = counts(data.optJSONObject("by_year"), byValue = false),
                dataNotes = notes,
                glossary = glossary,
            )
        }

        private fun aucRow(o: JSONObject): AucRow = AucRow(
            n = o.optInt("n"),
            auc = o.optDouble("auc", Double.NaN),
            top10 = o.optDouble("hit_top10", Double.NaN),
            top20 = o.optDouble("hit_top20", Double.NaN),
        )

        /** `{"sl": .., "en": ..}` -> Loc (null if the object is missing or has neither). */
        private fun loc(o: JSONObject?): Loc? {
            if (o == null) return null
            val sl = o.optString("sl", "")
            val en = o.optString("en", "")
            if (sl.isEmpty() && en.isEmpty()) return null
            return Loc(sl.ifEmpty { en }, en)
        }

        /** `["sl text", "en text"]` -> Loc. */
        private fun pair(a: JSONArray?): Loc? {
            if (a == null || a.length() == 0) return null
            val sl = a.optString(0, "")
            val en = if (a.length() > 1) a.optString(1, "") else ""
            return Loc(sl, en)
        }

        private fun doubles(a: JSONArray?): List<Double> {
            if (a == null) return emptyList()
            return (0 until a.length()).map { a.optDouble(it, Double.NaN) }
        }

        private fun counts(o: JSONObject?, byValue: Boolean): List<CountItem> {
            if (o == null) return emptyList()
            val out = ArrayList<CountItem>()
            val keys = o.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                out += CountItem(k, o.optInt(k))
            }
            return if (byValue) out.sortedByDescending { it.count } else out.sortedBy { it.label }
        }

        private fun valueText(v: Any?): String = when (v) {
            null -> "-"
            is JSONArray -> (0 until v.length()).joinToString("-") { valueText(v.opt(it)) }
            is Number -> {
                val d = v.toDouble()
                if (d == Math.floor(d) && Math.abs(d) < 1e9) d.toLong().toString()
                else String.format(java.util.Locale.US, "%.2f", d)
            }
            else -> v.toString()
        }
    }
}
