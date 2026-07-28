import { createNativeStackNavigator } from "@react-navigation/native-stack";
import React from "react";
import ModelDetailScreen from "../screens/ModelDetailScreen";
import ModelsListScreen from "../screens/ModelsListScreen";
import type { ModelsStackParamList } from "./types";

const Stack = createNativeStackNavigator<ModelsStackParamList>();

const ModelsNavigator: React.FC = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="ModelsList"
      component={ModelsListScreen}
      options={{ headerTitle: "Models" }}
    />
    <Stack.Screen
      name="ModelDetail"
      component={ModelDetailScreen}
      options={{ headerTitle: "Model" }}
    />
  </Stack.Navigator>
);

export default ModelsNavigator;
