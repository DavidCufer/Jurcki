package si.jurcki.app

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import java.io.IOException

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

/** Persists [AppSettings] (data URL, optional token, overlay opacity, base map) in DataStore. */
class SettingsStore(context: Context) {
    private val store = context.applicationContext.dataStore

    val flow: Flow<AppSettings> = store.data
        .catch { e -> if (e is IOException) emit(emptyPreferences()) else throw e }
        .map { p ->
            AppSettings(
                baseUrl = p[KEY_URL]?.takeIf { it.isNotBlank() } ?: DEFAULT_BASE_URL,
                token = p[KEY_TOKEN] ?: "",
                opacity = p[KEY_OPACITY] ?: 0.7f,
                baseMap = runCatching { BaseMap.valueOf(p[KEY_BASE_MAP] ?: "") }.getOrDefault(BaseMap.STREETS),
            )
        }

    suspend fun setConnection(baseUrl: String, token: String) {
        store.edit {
            it[KEY_URL] = baseUrl
            it[KEY_TOKEN] = token
        }
    }

    suspend fun setOpacity(value: Float) {
        store.edit { it[KEY_OPACITY] = value }
    }

    suspend fun setBaseMap(value: BaseMap) {
        store.edit { it[KEY_BASE_MAP] = value.name }
    }

    private companion object {
        val KEY_URL = stringPreferencesKey("base_url")
        val KEY_TOKEN = stringPreferencesKey("token")
        val KEY_OPACITY = floatPreferencesKey("opacity")
        val KEY_BASE_MAP = stringPreferencesKey("base_map")
    }
}
