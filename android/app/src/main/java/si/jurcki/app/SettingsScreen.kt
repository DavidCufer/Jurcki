@file:OptIn(ExperimentalMaterial3Api::class)

package si.jurcki.app

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp

/**
 * Data source settings: base URL of the published backend output and an optional GitHub
 * personal access token (sent as `Authorization: Bearer <token>`) for the private repository.
 */
@Composable
fun SettingsScreen(
    state: UiState,
    onSave: (baseUrl: String, token: String) -> Unit,
    onBack: () -> Unit,
) {
    var url by rememberSaveable { mutableStateOf(state.settings.baseUrl) }
    var token by rememberSaveable { mutableStateOf(state.settings.token) }
    var showToken by rememberSaveable { mutableStateOf(false) }

    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(stringResource(R.string.settings_title)) },
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
            Column(
                Modifier
                    .padding(padding)
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .imePadding()
                    .navigationBarsPadding()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(stringResource(R.string.settings_data_section), style = MaterialTheme.typography.titleMedium)

                OutlinedTextField(
                    value = url,
                    onValueChange = { url = it },
                    label = { Text(stringResource(R.string.settings_base_url)) },
                    supportingText = { Text(stringResource(R.string.settings_base_url_help)) },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                    modifier = Modifier.fillMaxWidth(),
                )

                OutlinedTextField(
                    value = token,
                    onValueChange = { token = it },
                    label = { Text(stringResource(R.string.settings_token)) },
                    supportingText = { Text(stringResource(R.string.settings_token_help)) },
                    singleLine = true,
                    visualTransformation = if (showToken) VisualTransformation.None else PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    trailingIcon = {
                        TextButton(onClick = { showToken = !showToken }) {
                            Text(stringResource(if (showToken) R.string.settings_token_hide else R.string.settings_token_show))
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Button(onClick = {
                        onSave(url, token)
                        onBack()
                    }) {
                        Text(stringResource(R.string.settings_save))
                    }
                    OutlinedButton(onClick = { url = DEFAULT_BASE_URL }) {
                        Text(stringResource(R.string.settings_reset_url))
                    }
                }

                Spacer(Modifier.height(8.dp))
                Text(stringResource(R.string.settings_cache_section), style = MaterialTheme.typography.titleMedium)
                val meta = state.meta
                if (meta == null) {
                    Text(stringResource(R.string.settings_cache_none), style = MaterialTheme.typography.bodyMedium)
                } else {
                    Text(
                        stringResource(R.string.settings_cache_generated, meta.generated),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Text(
                        stringResource(R.string.settings_cache_days, meta.days.joinToString(", ")),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }

                Spacer(Modifier.height(8.dp))
                Text(stringResource(R.string.settings_maps_section), style = MaterialTheme.typography.titleMedium)
                Text(
                    stringResource(R.string.settings_attribution),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
