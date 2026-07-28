import { createNativeStackNavigator } from "@react-navigation/native-stack";
import React from "react";
import AdminSystemScreen from "../screens/more/AdminSystemScreen";
import AdminUsersScreen from "../screens/more/AdminUsersScreen";
import ApiKeysScreen from "../screens/more/ApiKeysScreen";
import ChangePasswordScreen from "../screens/more/ChangePasswordScreen";
import FinancialScreen from "../screens/more/FinancialScreen";
import MfaSetupScreen from "../screens/more/MfaSetupScreen";
import MoreMenuScreen from "../screens/more/MoreMenuScreen";
import NotificationsScreen from "../screens/more/NotificationsScreen";
import ProfileScreen from "../screens/more/ProfileScreen";
import type { MoreStackParamList } from "./types";

const Stack = createNativeStackNavigator<MoreStackParamList>();

const MoreNavigator: React.FC = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="MoreMenu"
      component={MoreMenuScreen}
      options={{ headerTitle: "More" }}
    />
    <Stack.Screen
      name="Profile"
      component={ProfileScreen}
      options={{ headerTitle: "Profile" }}
    />
    <Stack.Screen
      name="ChangePassword"
      component={ChangePasswordScreen}
      options={{ headerTitle: "Change password" }}
    />
    <Stack.Screen
      name="MfaSetup"
      component={MfaSetupScreen}
      options={{ headerTitle: "Two-factor authentication" }}
    />
    <Stack.Screen
      name="ApiKeys"
      component={ApiKeysScreen}
      options={{ headerTitle: "API keys" }}
    />
    <Stack.Screen
      name="Financial"
      component={FinancialScreen}
      options={{ headerTitle: "Financial" }}
    />
    <Stack.Screen
      name="Notifications"
      component={NotificationsScreen}
      options={{ headerTitle: "Notifications" }}
    />
    <Stack.Screen
      name="AdminUsers"
      component={AdminUsersScreen}
      options={{ headerTitle: "Users" }}
    />
    <Stack.Screen
      name="AdminSystem"
      component={AdminSystemScreen}
      options={{ headerTitle: "System health" }}
    />
  </Stack.Navigator>
);

export default MoreNavigator;
