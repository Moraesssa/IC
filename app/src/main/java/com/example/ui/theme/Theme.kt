package com.example.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val DarkColorScheme =
  darkColorScheme(
    primary = EditorialBlue,
    secondary = EditorialSoftBlue,
    tertiary = EditorialDarkNavy,
    background = DarkBackground,
    surface = DarkSurface,
    onPrimary = EditorialCream,
    onSecondary = EditorialDarkNavy,
    onTertiary = EditorialCream,
    onBackground = EditorialCream,
    onSurface = EditorialCream,
    error = ErrorRed
  )

private val LightColorScheme =
  lightColorScheme(
    primary = EditorialBlue,
    secondary = EditorialMuted,
    tertiary = EditorialDarkNavy,
    background = EditorialCream,
    surface = Color.White,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onTertiary = Color.White,
    onBackground = EditorialCharcoal,
    onSurface = EditorialCharcoal,
    error = ErrorRed
  )

@Composable
fun MyApplicationTheme(
  darkTheme: Boolean = isSystemInDarkTheme(),
  // Use our curated colors instead of system dynamic ones for strict design identity
  dynamicColor: Boolean = false,
  content: @Composable () -> Unit,
) {
  val colorScheme =
    when {
      dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
        val context = LocalContext.current
        if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
      }

      darkTheme -> DarkColorScheme
      else -> LightColorScheme
    }

  MaterialTheme(colorScheme = colorScheme, typography = Typography, content = content)
}
