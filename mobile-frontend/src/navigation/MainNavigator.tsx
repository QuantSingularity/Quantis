import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import MaterialCommunityIcons from "@expo/vector-icons/MaterialCommunityIcons";
import React from "react";
import { useTheme } from "react-native-paper";
import DashboardNavigator from "./DashboardNavigator";
import DatasetsNavigator from "./DatasetsNavigator";
import ModelsNavigator from "./ModelsNavigator";
import MoreNavigator from "./MoreNavigator";
import PredictionsNavigator from "./PredictionsNavigator";
import type { MainTabParamList } from "./types";

const Tab = createBottomTabNavigator<MainTabParamList>();

const ICONS: Record<
  keyof MainTabParamList,
  keyof typeof MaterialCommunityIcons.glyphMap
> = {
  DashboardTab: "view-dashboard-outline",
  DatasetsTab: "database-outline",
  ModelsTab: "chart-timeline-variant",
  PredictionsTab: "chart-line",
  MoreTab: "menu",
};

const LABELS: Record<keyof MainTabParamList, string> = {
  DashboardTab: "Dashboard",
  DatasetsTab: "Datasets",
  ModelsTab: "Models",
  PredictionsTab: "Predict",
  MoreTab: "More",
};

const MainNavigator: React.FC = () => {
  const theme = useTheme();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.onSurfaceVariant,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopColor: theme.colors.outlineVariant,
        },
        tabBarLabel: LABELS[route.name as keyof MainTabParamList],
        tabBarIcon: ({ color, size }) => (
          <MaterialCommunityIcons
            name={ICONS[route.name as keyof MainTabParamList]}
            color={color}
            size={size}
          />
        ),
      })}
    >
      <Tab.Screen name="DashboardTab" component={DashboardNavigator} />
      <Tab.Screen name="DatasetsTab" component={DatasetsNavigator} />
      <Tab.Screen name="ModelsTab" component={ModelsNavigator} />
      <Tab.Screen name="PredictionsTab" component={PredictionsNavigator} />
      <Tab.Screen name="MoreTab" component={MoreNavigator} />
    </Tab.Navigator>
  );
};

export default MainNavigator;
