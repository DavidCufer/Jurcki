@file:OptIn(ExperimentalMaterial3Api::class)

package si.jurcki.app

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlin.math.roundToInt

// Bar colours: tuned model, literature parameters, season only.
private val COLOR_TUNED = Color(0xFF2E7D32)
private val COLOR_LITERATURE = Color(0xFF1565C0)
private val COLOR_SEASON = Color(0xFF8D6E63)
private val COLOR_PRIOR = Color(0xFF8D6E63)

/** "Model / success" screen, built from `assets/model/stats.json`. */
@Composable
fun StatsScreen(stats: StatsData?, onBack: () -> Unit) {
    val english = remember { Locale.getDefault().language == "en" }
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(stringResource(R.string.stats_title)) },
                    navigationIcon = {
                        IconButton(onClick = onBack) {
                            Icon(
                                Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = stringResource(R.string.action_back),
                            )
                        }
                    },
                )
            },
        ) { padding ->
            if (stats == null) {
                Text(
                    stringResource(R.string.stats_unavailable),
                    modifier = Modifier.padding(padding).padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium,
                )
            } else {
                LazyColumn(
                    modifier = Modifier
                        .padding(padding)
                        .fillMaxSize()
                        .navigationBarsPadding(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    item { HeadlineWhere(stats) }
                    item { HeadlineWhen(stats, english) }
                    item { HeadlineHits(stats) }
                    item { AucExplainCard() }
                    item { HabitatCard(stats, english) }
                    item { TriggerCard(stats, english) }
                    item { PerYearCard(stats) }
                    item { ParamsCard(stats, english) }
                    item { WeekendCard(stats, english) }
                    item { DataCard(stats, english) }
                    item { GlossaryCard(stats, english) }
                    item {
                        Text(
                            stringResource(R.string.stats_generated, stats.generated),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Headline cards
// ---------------------------------------------------------------------------------------------

@Composable
private fun HeadlineWhere(stats: StatsData) {
    val u = stats.habitatUnit("si_all") ?: stats.habitatUnits.firstOrNull()
    val prior = u?.prior
    val model = u?.model
    StatCard(stringResource(R.string.stats_card_where)) {
        if (prior == null || model == null) {
            Text(stringResource(R.string.stats_unavailable))
            return@StatCard
        }
        Text(
            "AUC ${fmt2(prior.auc)}  \u2192  ${fmt2(model.auc)}",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        AucBar(stringResource(R.string.stats_prior), prior.auc, COLOR_PRIOR)
        AucBar(stringResource(R.string.stats_model), model.auc, COLOR_TUNED)
        Caption(stringResource(R.string.stats_where_caption, stats.habitatBlockKm, model.n))
    }
}

@Composable
private fun HeadlineWhen(stats: StatsData, english: Boolean) {
    val seasonRow = stats.triggerRow("season_only:comb_tol")
    val litRow = stats.triggerRow("literature:comb_tol")
    val tunedRow = stats.triggerRow("tuned_loyo:comb_tol")
    val season = seasonRow?.tolerant
    val lit = litRow?.tolerant
    val tuned = tunedRow?.tolerant
    StatCard(stringResource(R.string.stats_card_when)) {
        if (seasonRow == null || litRow == null || tunedRow == null || season == null || lit == null || tuned == null) {
            Text(stringResource(R.string.stats_unavailable))
            return@StatCard
        }
        Text(
            "AUC ${fmt2(season.auc)}  \u2192  ${fmt2(tuned.auc)}",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        AucBar(seasonRow.label.text(english), season.auc, COLOR_SEASON)
        AucBar(litRow.label.text(english), lit.auc, COLOR_LITERATURE)
        AucBar(tunedRow.label.text(english), tuned.auc, COLOR_TUNED)
        val g = stats.gainTunedVsSeason
        if (g.size >= 3) {
            Caption(stringResource(R.string.stats_gain, signed2(g[0]), signed2(g[1]), signed2(g[2])))
        }
        Caption(stringResource(R.string.stats_when_caption, stats.triggerFinds))
    }
}

@Composable
private fun HeadlineHits(stats: StatsData) {
    val u = stats.habitatUnit("si_all")
    val prior = u?.prior
    val model = u?.model
    val season = stats.triggerRow("season_only:comb_tol")?.tolerant
    val lit = stats.triggerRow("literature:comb_tol")?.tolerant
    val tuned = stats.triggerRow("tuned_loyo:comb_tol")?.tolerant
    StatCard(stringResource(R.string.stats_card_hits)) {
        Caption(stringResource(R.string.stats_hits_explain))
        if (prior != null && model != null) {
            Text(stringResource(R.string.stats_card_where), style = MaterialTheme.typography.labelLarge)
            PercentBar(stringResource(R.string.stats_prior), prior.top20, COLOR_PRIOR)
            PercentBar(stringResource(R.string.stats_model), model.top20, COLOR_TUNED)
        }
        if (season != null && lit != null && tuned != null) {
            Spacer(Modifier.height(4.dp))
            Text(stringResource(R.string.stats_card_when), style = MaterialTheme.typography.labelLarge)
            PercentBar(stringResource(R.string.stats_row_season), season.top20, COLOR_SEASON)
            PercentBar(stringResource(R.string.stats_row_literature), lit.top20, COLOR_LITERATURE)
            PercentBar(stringResource(R.string.stats_row_tuned), tuned.top20, COLOR_TUNED)
        }
        Caption(stringResource(R.string.stats_hits_chance))
    }
}

@Composable
private fun AucExplainCard() {
    StatCard(stringResource(R.string.stats_auc_title)) {
        Text(stringResource(R.string.stats_auc_text), style = MaterialTheme.typography.bodyMedium)
        AucBar(stringResource(R.string.stats_auc_05), 0.5, MaterialTheme.colorScheme.outline)
        AucBar(stringResource(R.string.stats_auc_07), 0.7, COLOR_LITERATURE)
        AucBar(stringResource(R.string.stats_auc_10), 1.0, COLOR_TUNED)
        Caption(stringResource(R.string.stats_auc_note))
    }
}

// ---------------------------------------------------------------------------------------------
// Detail cards
// ---------------------------------------------------------------------------------------------

@Composable
private fun HabitatCard(stats: StatsData, english: Boolean) {
    StatCard(stats.habitatTitle.text(english)) {
        Text(stats.habitatDescription.text(english), style = MaterialTheme.typography.bodyMedium)
        Caption(stringResource(R.string.stats_habitat_table, stats.habitatBlockKm))
        for (u in stats.habitatUnits) {
            Text(u.label.text(english), style = MaterialTheme.typography.labelLarge)
            u.prior?.let { AucBar(stringResource(R.string.stats_prior) + "  (n=${it.n})", it.auc, COLOR_PRIOR) }
            u.model?.let { AucBar(stringResource(R.string.stats_model), it.auc, COLOR_TUNED) }
        }
        if (stats.importance.isNotEmpty()) {
            Spacer(Modifier.height(4.dp))
            Text(stringResource(R.string.stats_importance_title), style = MaterialTheme.typography.labelLarge)
            val max = stats.importance.maxOf { it.second }.takeIf { it > 0.0 } ?: 1.0
            for ((name, v) in stats.importance) {
                RelativeBar(name, v, max, COLOR_TUNED)
            }
        }
    }
}

@Composable
private fun TriggerCard(stats: StatsData, english: Boolean) {
    StatCard(stats.triggerTitle.text(english)) {
        Text(stats.triggerDescription.text(english), style = MaterialTheme.typography.bodyMedium)
        Caption(stringResource(R.string.stats_trigger_table))
        // Header
        Row(Modifier.fillMaxWidth()) {
            Text("", Modifier.weight(2.4f))
            HeaderCell(stringResource(R.string.stats_col_auc), Modifier.weight(0.8f))
            HeaderCell(stringResource(R.string.stats_col_top10), Modifier.weight(0.9f))
            HeaderCell(stringResource(R.string.stats_col_top20), Modifier.weight(0.9f))
        }
        for (r in stats.triggerRows) {
            val t = r.tolerant ?: continue
            HorizontalDivider(Modifier.padding(vertical = 2.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(r.label.text(english), Modifier.weight(2.4f), style = MaterialTheme.typography.bodySmall)
                NumCell(fmt2(t.auc), Modifier.weight(0.8f))
                NumCell(pct(t.top10), Modifier.weight(0.9f))
                NumCell(pct(t.top20), Modifier.weight(0.9f))
            }
        }
        stats.triggerNote?.let { Caption(it.text(english)) }
    }
}

@Composable
private fun PerYearCard(stats: StatsData) {
    val english = remember { Locale.getDefault().language == "en" }
    StatCard(stringResource(R.string.stats_per_year_title)) {
        stats.perYearNote?.let { Caption(it.text(english)) }
        // Legend
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            LegendDot(COLOR_TUNED, stringResource(R.string.stats_row_tuned_short))
            LegendDot(COLOR_LITERATURE, stringResource(R.string.stats_row_literature_short))
            LegendDot(COLOR_SEASON, stringResource(R.string.stats_row_season_short))
        }
        // Numbers
        Row(Modifier.fillMaxWidth()) {
            HeaderCell(stringResource(R.string.stats_col_year), Modifier.weight(1f))
            HeaderCell("n", Modifier.weight(0.6f))
            HeaderCell(stringResource(R.string.stats_row_tuned_short), Modifier.weight(1f))
            HeaderCell(stringResource(R.string.stats_row_literature_short), Modifier.weight(1f))
            HeaderCell(stringResource(R.string.stats_row_season_short), Modifier.weight(1f))
            HeaderCell(stringResource(R.string.stats_col_top20), Modifier.weight(1f))
        }
        for (y in stats.perYear) {
            Row(Modifier.fillMaxWidth()) {
                NumCell(y.year.toString(), Modifier.weight(1f))
                NumCell(y.n.toString(), Modifier.weight(0.6f))
                NumCell(fmt2(y.aucTuned), Modifier.weight(1f))
                NumCell(fmt2(y.aucLiterature), Modifier.weight(1f))
                NumCell(fmt2(y.aucSeason), Modifier.weight(1f))
                NumCell(pct(y.top20Tuned), Modifier.weight(1f))
            }
        }
        // Bars: AUC per year (axis 0..1, thin line = 0.5 = guessing)
        Spacer(Modifier.height(6.dp))
        Text(stringResource(R.string.stats_per_year_bars), style = MaterialTheme.typography.labelLarge)
        for (y in stats.perYear) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "${y.year} (${y.n})",
                    Modifier.width(76.dp),
                    style = MaterialTheme.typography.labelSmall,
                )
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    ValueBar(y.aucTuned, COLOR_TUNED, 6)
                    ValueBar(y.aucLiterature, COLOR_LITERATURE, 6)
                    ValueBar(y.aucSeason, COLOR_SEASON, 6)
                }
            }
            Spacer(Modifier.height(2.dp))
        }
    }
}

@Composable
private fun ParamsCard(stats: StatsData, english: Boolean) {
    StatCard(stringResource(R.string.stats_params_title)) {
        Row(Modifier.fillMaxWidth()) {
            Text("", Modifier.weight(1.4f))
            HeaderCell(stringResource(R.string.stats_param_lit), Modifier.weight(1f))
            HeaderCell(stringResource(R.string.stats_param_used), Modifier.weight(1f))
        }
        for (p in stats.params) {
            HorizontalDivider(Modifier.padding(vertical = 2.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    p.name + if (p.unit.isNotEmpty()) "\n(${p.unit})" else "",
                    Modifier.weight(1.4f),
                    style = MaterialTheme.typography.labelMedium,
                )
                NumCell(p.literature, Modifier.weight(1f))
                NumCell(p.tuned, Modifier.weight(1f), bold = true)
            }
            val text = p.text.text(english)
            if (text.isNotEmpty()) {
                Text(
                    text,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun WeekendCard(stats: StatsData, english: Boolean) {
    val w = stats.weekend ?: return
    StatCard(stringResource(R.string.stats_weekend_title)) {
        PercentBar(stringResource(R.string.stats_weekend_finds), w.offDayShare, COLOR_LITERATURE)
        PercentBar(stringResource(R.string.stats_weekend_season), w.baselineShare, COLOR_SEASON)
        Text(w.text.text(english), style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun DataCard(stats: StatsData, english: Boolean) {
    StatCard(stringResource(R.string.stats_data_title)) {
        Text(
            stringResource(R.string.stats_data_totals, stats.findsInBounds, stats.duplicatesRemoved, stats.inBacktest),
            style = MaterialTheme.typography.bodyMedium,
        )
        CountSection(stringResource(R.string.stats_by_country), stats.byCountry)
        CountSection(stringResource(R.string.stats_by_species), stats.bySpecies)
        CountSection(stringResource(R.string.stats_by_source), stats.bySource)
        CountSection(stringResource(R.string.stats_by_year), stats.byYear)
        for (n in stats.dataNotes) {
            Caption(n.text(english))
        }
    }
}

@Composable
private fun GlossaryCard(stats: StatsData, english: Boolean) {
    if (stats.glossary.isEmpty()) return
    StatCard(stringResource(R.string.stats_glossary_title)) {
        for ((key, loc) in stats.glossary) {
            Text(termName(key), style = MaterialTheme.typography.labelLarge)
            Text(loc.text(english), style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun termName(key: String): String = when (key) {
    "auc" -> "AUC"
    "top10" -> stringResource(R.string.stats_col_top10)
    "top20" -> stringResource(R.string.stats_col_top20)
    "tol" -> stringResource(R.string.stats_term_tol)
    "loyo" -> "LOYO"
    "blocks" -> stringResource(R.string.stats_term_blocks)
    else -> key
}

// ---------------------------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------------------------

@Composable
private fun StatCard(title: String, content: @Composable () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
private fun Caption(text: String) {
    Text(text, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun HeaderCell(text: String, modifier: Modifier) {
    Text(
        text,
        modifier,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        textAlign = TextAlign.End,
    )
}

@Composable
private fun NumCell(text: String, modifier: Modifier, bold: Boolean = false) {
    Text(
        text,
        modifier,
        style = MaterialTheme.typography.bodySmall,
        fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal,
        textAlign = TextAlign.End,
    )
}

@Composable
private fun LegendDot(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.width(12.dp).height(8.dp).background(color, RoundedCornerShape(2.dp)))
        Spacer(Modifier.width(4.dp))
        Text(label, style = MaterialTheme.typography.labelSmall)
    }
}

/** Label + value + bar on the AUC axis 0..1 (thin vertical line = 0.5 = guessing). */
@Composable
private fun AucBar(label: String, value: Double, color: Color) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
            Text(fmt2(value), style = MaterialTheme.typography.labelLarge)
        }
        ValueBar(value, color, 10)
    }
}

/** Label + percentage + bar on the axis 0..100 % (vertical line = 20 % = chance for "best 20 %"). */
@Composable
private fun PercentBar(label: String, value: Double, color: Color) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
            Text(pct(value), style = MaterialTheme.typography.labelLarge)
        }
        ValueBar(value, color, 10, markerAt = 0.2)
    }
}

/** Bar relative to a maximum (feature importance), no marker. */
@Composable
private fun RelativeBar(label: String, value: Double, max: Double, color: Color) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.width(120.dp), style = MaterialTheme.typography.labelSmall)
        Box(Modifier.weight(1f)) {
            ValueBar(value / max, color, 8, markerAt = null)
        }
        Spacer(Modifier.width(6.dp))
        Text(fmt2(value), Modifier.width(36.dp), style = MaterialTheme.typography.labelSmall, textAlign = TextAlign.End)
    }
}

/** Horizontal bar of `value` on a 0..1 axis; [markerAt] draws a thin line (0.5 by default). */
@Composable
private fun ValueBar(value: Double, color: Color, heightDp: Int, markerAt: Double? = 0.5) {
    val track = MaterialTheme.colorScheme.surfaceVariant
    val marker = MaterialTheme.colorScheme.onSurface
    val frac = if (value.isNaN()) 0f else value.toFloat().coerceIn(0f, 1f)
    Box(
        Modifier
            .fillMaxWidth()
            .height(heightDp.dp)
            .clip(RoundedCornerShape((heightDp / 2).dp))
            .background(track),
    ) {
        if (frac > 0f) {
            Box(
                Modifier
                    .fillMaxWidth(frac)
                    .fillMaxHeight()
                    .background(color),
            )
        }
        if (markerAt != null && markerAt > 0.0 && markerAt < 1.0) {
            Row(Modifier.fillMaxSize()) {
                Spacer(Modifier.weight(markerAt.toFloat()))
                Box(
                    Modifier
                        .width(1.dp)
                        .fillMaxHeight()
                        .background(marker.copy(alpha = 0.6f)),
                )
                Spacer(Modifier.weight(1f - markerAt.toFloat()))
            }
        }
    }
}

@Composable
private fun CountSection(title: String, items: List<CountItem>) {
    if (items.isEmpty()) return
    val max = items.maxOf { it.count }.coerceAtLeast(1)
    Text(title, style = MaterialTheme.typography.labelLarge)
    for (c in items) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(c.label, Modifier.width(120.dp), style = MaterialTheme.typography.labelSmall, maxLines = 1)
            Box(Modifier.weight(1f)) {
                ValueBar(c.count.toDouble() / max, MaterialTheme.colorScheme.primary, 8, markerAt = null)
            }
            Spacer(Modifier.width(6.dp))
            Text(
                c.count.toString(),
                Modifier.width(36.dp),
                style = MaterialTheme.typography.labelSmall,
                textAlign = TextAlign.End,
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Number formatting (default locale: decimal comma in Slovenian)
// ---------------------------------------------------------------------------------------------

private fun fmt2(v: Double): String = if (v.isNaN()) "-" else String.format(Locale.getDefault(), "%.2f", v)

private fun signed2(v: Double): String =
    if (v.isNaN()) "-" else String.format(Locale.getDefault(), "%+.2f", v)

private fun pct(v: Double): String = if (v.isNaN()) "-" else "${(v * 100).roundToInt()} %"
