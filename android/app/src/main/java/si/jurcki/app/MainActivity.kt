package si.jurcki.app

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            JurckiTheme {
                JurckiApp()
            }
        }
    }
}

private const val SCREEN_MAP = "map"
private const val SCREEN_STATS = "stats"
private const val SCREEN_ABOUT = "about"

@Composable
private fun JurckiApp(vm: MainViewModel = viewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    var screen by rememberSaveable { mutableStateOf(SCREEN_MAP) }

    // Coming back to the app after a few hours: refresh the forecast.
    LifecycleEventEffect(Lifecycle.Event.ON_START) { vm.onForeground() }

    Box(Modifier.fillMaxSize()) {
        // The map stays composed underneath (MapView must not be re-attached); the other screens cover it.
        MapScreen(
            state = state,
            vm = vm,
            onOpenSettings = { screen = SCREEN_ABOUT },
            onOpenStats = { screen = SCREEN_STATS },
        )
        if (screen == SCREEN_STATS) {
            BackHandler { screen = SCREEN_MAP }
            StatsScreen(stats = state.stats, onBack = { screen = SCREEN_MAP })
        } else if (screen == SCREEN_ABOUT) {
            BackHandler { screen = SCREEN_MAP }
            AboutScreen(state = state, onClearCache = { vm.clearWeatherCache() }, onBack = { screen = SCREEN_MAP })
        }
    }
}

@Composable
private fun JurckiTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val scheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> darkColorScheme()
        else -> lightColorScheme()
    }
    MaterialTheme(colorScheme = scheme, content = content)
}
