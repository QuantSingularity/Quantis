import React from "react";
import { RefreshControl, ScrollView, StyleSheet, View } from "react-native";
import { useTheme } from "react-native-paper";

interface Props {
  children: React.ReactNode;
  scroll?: boolean;
  refreshing?: boolean;
  onRefresh?: () => void;
}

/** Consistent screen padding + background + optional pull-to-refresh. */
const ScreenContainer: React.FC<Props> = ({
  children,
  scroll = true,
  refreshing,
  onRefresh,
}) => {
  const theme = useTheme();

  if (!scroll) {
    return (
      <View
        style={[styles.container, { backgroundColor: theme.colors.background }]}
      >
        {children}
      </View>
    );
  }

  return (
    <ScrollView
      style={{ backgroundColor: theme.colors.background }}
      contentContainerStyle={styles.container}
      refreshControl={
        onRefresh ? (
          <RefreshControl
            refreshing={Boolean(refreshing)}
            onRefresh={onRefresh}
            tintColor={theme.colors.primary}
          />
        ) : undefined
      }
    >
      {children}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { padding: 16, flexGrow: 1 },
});

export default ScreenContainer;
