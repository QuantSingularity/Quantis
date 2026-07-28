import { createNativeStackNavigator } from "@react-navigation/native-stack";
import React from "react";
import DatasetDetailScreen from "../screens/DatasetDetailScreen";
import DatasetsListScreen from "../screens/DatasetsListScreen";
import type { DatasetsStackParamList } from "./types";

const Stack = createNativeStackNavigator<DatasetsStackParamList>();

const DatasetsNavigator: React.FC = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="DatasetsList"
      component={DatasetsListScreen}
      options={{ headerTitle: "Datasets" }}
    />
    <Stack.Screen
      name="DatasetDetail"
      component={DatasetDetailScreen}
      options={{ headerTitle: "Dataset" }}
    />
  </Stack.Navigator>
);

export default DatasetsNavigator;
