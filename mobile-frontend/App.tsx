import React from "react";
import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { PaperProvider } from "react-native-paper";
import { AuthProvider } from "./src/context/AuthContext";
import {
  ThemeModeProvider,
  useThemeMode,
} from "./src/context/ThemeModeContext";
import RootNavigator from "./src/navigation/RootNavigator";
import { buildPaperTheme } from "./src/theme";

const ThemedApp: React.FC = () => {
  const { mode } = useThemeMode();
  const paperTheme = buildPaperTheme(mode);

  return (
    <PaperProvider theme={paperTheme}>
      <StatusBar style={mode === "dark" ? "light" : "dark"} />
      <AuthProvider>
        <RootNavigator />
      </AuthProvider>
    </PaperProvider>
  );
};

export default function App() {
  return (
    <SafeAreaProvider>
      <ThemeModeProvider>
        <ThemedApp />
      </ThemeModeProvider>
    </SafeAreaProvider>
  );
}
