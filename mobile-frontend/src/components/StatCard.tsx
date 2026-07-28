import React from "react";
import { StyleSheet, View } from "react-native";
import { Card, Text, useTheme } from "react-native-paper";

interface Props {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
}

const StatCard: React.FC<Props> = ({ label, value, icon }) => {
  const theme = useTheme();
  return (
    <Card style={styles.card} mode="outlined">
      <Card.Content>
        <View style={styles.header}>
          <Text
            variant="labelMedium"
            style={{ color: theme.colors.onSurfaceVariant }}
          >
            {label}
          </Text>
          {icon}
        </View>
        <Text
          variant="headlineMedium"
          style={{ fontWeight: "700", marginTop: 6 }}
        >
          {value}
        </Text>
      </Card.Content>
    </Card>
  );
};

const styles = StyleSheet.create({
  card: { flex: 1, minWidth: "45%" },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
});

export default StatCard;
