import { createNativeStackNavigator } from "@react-navigation/native-stack";
import React from "react";
import PredictionsScreen from "../screens/PredictionsScreen";
import type { PredictionsStackParamList } from "./types";

const Stack = createNativeStackNavigator<PredictionsStackParamList>();

const PredictionsNavigator: React.FC = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="PredictionsHome"
      component={PredictionsScreen}
      options={{ headerTitle: "Predictions" }}
    />
  </Stack.Navigator>
);

export default PredictionsNavigator;
