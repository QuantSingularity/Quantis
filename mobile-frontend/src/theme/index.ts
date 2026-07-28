import {
  MD3DarkTheme,
  MD3LightTheme,
  adaptNavigationTheme,
} from "react-native-paper";
import {
  DarkTheme as NavigationDarkTheme,
  DefaultTheme as NavigationDefaultTheme,
} from "@react-navigation/native";
import { colorTokens, ThemeMode } from "./tokens";

export const buildPaperTheme = (mode: ThemeMode) => {
  const palette = colorTokens[mode];
  const base = mode === "dark" ? MD3DarkTheme : MD3LightTheme;

  return {
    ...base,
    dark: mode === "dark",
    roundness: 12,
    colors: {
      ...base.colors,
      primary: palette.primary,
      secondary: palette.secondary,
      background: palette.background,
      surface: palette.surface,
      surfaceVariant: palette.surfaceElevated,
      error: palette.error,
      onBackground: palette.textPrimary,
      onSurface: palette.textPrimary,
      outline: palette.border,
    },
  };
};

export const buildNavigationTheme = (mode: ThemeMode) => {
  const palette = colorTokens[mode];
  const { LightTheme, DarkTheme } = adaptNavigationTheme({
    reactNavigationLight: NavigationDefaultTheme,
    reactNavigationDark: NavigationDarkTheme,
  });
  const base = mode === "dark" ? DarkTheme : LightTheme;

  return {
    ...base,
    colors: {
      ...base.colors,
      primary: palette.primary,
      background: palette.background,
      card: palette.surface,
      text: palette.textPrimary,
      border: palette.border,
      notification: palette.error,
    },
  };
};
