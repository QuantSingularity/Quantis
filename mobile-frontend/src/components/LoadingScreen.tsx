import React from "react";
import { StyleSheet, View } from "react-native";
import { ActivityIndicator, Text, useTheme } from "react-native-paper";

interface Props {
  label?: string;
}

const LoadingScreen: React.FC<Props> = ({ label = "Loading…" }) => {
  const theme = useTheme();
  return (
    <View
      style={[styles.container, { backgroundColor: theme.colors.background }]}
    >
      <ActivityIndicator size="large" color={theme.colors.primary} />
      <Text
        variant="bodyMedium"
        style={{ marginTop: 12, color: theme.colors.onSurfaceVariant }}
      >
        {label}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: "center", justifyContent: "center" },
});

export default LoadingScreen;
