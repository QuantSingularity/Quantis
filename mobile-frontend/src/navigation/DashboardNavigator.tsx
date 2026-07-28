import { createNativeStackNavigator } from "@react-navigation/native-stack";
import React from "react";
import DashboardScreen from "../screens/DashboardScreen";
import type { DashboardStackParamList } from "./types";

const Stack = createNativeStackNavigator<DashboardStackParamList>();

const DashboardNavigator: React.FC = () => (
  <Stack.Navigator screenOptions={{ headerTitle: "Dashboard" }}>
    <Stack.Screen
      name="DashboardHome"
      component={DashboardScreen}
      options={{ headerShown: false }}
    />
  </Stack.Navigator>
);

export default DashboardNavigator;
