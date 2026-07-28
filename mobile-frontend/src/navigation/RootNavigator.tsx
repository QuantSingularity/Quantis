import { NavigationContainer } from "@react-navigation/native";
import React from "react";
import LoadingScreen from "../components/LoadingScreen";
import { useAuth } from "../context/AuthContext";
import { buildNavigationTheme } from "../theme";
import { useThemeMode } from "../context/ThemeModeContext";
import AuthNavigator from "./AuthNavigator";
import MainNavigator from "./MainNavigator";

const RootNavigator: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();
  const { mode } = useThemeMode();

  if (isLoading) return <LoadingScreen label="Checking your session…" />;

  return (
    <NavigationContainer theme={buildNavigationTheme(mode)}>
      {isAuthenticated ? <MainNavigator /> : <AuthNavigator />}
    </NavigationContainer>
  );
};

export default RootNavigator;
