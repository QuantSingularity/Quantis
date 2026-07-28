import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import { ProgressBar, Text, useTheme } from "react-native-paper";
import { getErrorMessage, monitoringAPI } from "../../api";
import LoadingScreen from "../../components/LoadingScreen";
import ScreenContainer from "../../components/ScreenContainer";
import StatusChip from "../../components/StatusChip";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "AdminSystem">;

interface HealthData {
  status?: string;
  database_status?: string;
  api_status?: string;
  cpu_usage?: number;
  memory_usage?: { percent?: number };
  disk_usage?: { percent?: number };
}

const UsageBar: React.FC<{ label: string; value?: number }> = ({
  label,
  value = 0,
}) => {
  const theme = useTheme();
  return (
    <View style={styles.usageRow}>
      <View style={styles.usageLabelRow}>
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.onSurfaceVariant }}
        >
          {label}
        </Text>
        <Text variant="bodyMedium" style={{ fontWeight: "700" }}>
          {value}%
        </Text>
      </View>
      <ProgressBar
        progress={Math.min(value, 100) / 100}
        color={
          value > 85
            ? theme.colors.error
            : value > 65
              ? "#F59E0B"
              : theme.colors.primary
        }
        style={styles.progressBar}
      />
    </View>
  );
};

const AdminSystemScreen: React.FC<Props> = () => {
  const theme = useTheme();
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    monitoringAPI
      .health()
      .then(({ data }) => setHealth(data as HealthData))
      .catch((err) => setError(getErrorMessage(err)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingScreen label="Loading system health…" />;

  return (
    <ScreenContainer>
      <Text
        variant="headlineSmall"
        style={{ fontWeight: "700", marginBottom: 16 }}
      >
        System health
      </Text>

      {error && (
        <Text
          variant="bodySmall"
          style={{ color: theme.colors.error, marginBottom: 12 }}
        >
          {error}
        </Text>
      )}

      <View style={styles.statusRow}>
        <Text variant="bodyMedium">Overall status</Text>
        <StatusChip status={health?.status || "unknown"} />
      </View>
      <View style={styles.statusRow}>
        <Text variant="bodyMedium">Database</Text>
        <StatusChip status={health?.database_status || "unknown"} />
      </View>
      <View style={styles.statusRow}>
        <Text variant="bodyMedium">API</Text>
        <StatusChip status={health?.api_status || "unknown"} />
      </View>

      <Text
        variant="titleMedium"
        style={{ fontWeight: "700", marginTop: 24, marginBottom: 12 }}
      >
        Resource usage
      </Text>
      <UsageBar label="CPU" value={health?.cpu_usage} />
      <UsageBar label="Memory" value={health?.memory_usage?.percent} />
      <UsageBar label="Disk" value={health?.disk_usage?.percent} />
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  statusRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  usageRow: { marginBottom: 16 },
  usageLabelRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 6,
  },
  progressBar: { height: 8, borderRadius: 4 },
});

export default AdminSystemScreen;
