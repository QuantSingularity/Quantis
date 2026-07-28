import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import { Card, DataTable, Text, useTheme } from "react-native-paper";
import { datasetsAPI, getErrorMessage } from "../api";
import { Dataset } from "../api/types";
import LoadingScreen from "../components/LoadingScreen";
import ScreenContainer from "../components/ScreenContainer";
import StatusChip from "../components/StatusChip";
import type { DatasetsStackParamList } from "../navigation/types";

type Props = NativeStackScreenProps<DatasetsStackParamList, "DatasetDetail">;

const DatasetDetailScreen: React.FC<Props> = ({ route }) => {
  const { datasetId } = route.params;
  const theme = useTheme();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const [datasetRes, statsRes] = await Promise.allSettled([
        datasetsAPI.get(datasetId),
        datasetsAPI.stats(datasetId),
      ]);
      if (cancelled) return;
      if (datasetRes.status === "fulfilled") setDataset(datasetRes.value.data);
      else setError(getErrorMessage(datasetRes.reason));
      if (statsRes.status === "fulfilled")
        setStats(statsRes.value.data as Record<string, unknown>);
      setLoading(false);
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  if (loading) return <LoadingScreen label="Loading dataset…" />;
  if (error && !dataset) {
    return (
      <ScreenContainer>
        <Text style={{ color: theme.colors.error }}>{error}</Text>
      </ScreenContainer>
    );
  }

  return (
    <ScreenContainer>
      <View style={styles.header}>
        <Text variant="headlineSmall" style={{ fontWeight: "700", flex: 1 }}>
          {dataset?.name}
        </Text>
        <StatusChip status={dataset?.status} />
      </View>
      {dataset?.description && (
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.onSurfaceVariant, marginTop: 6 }}
        >
          {dataset.description}
        </Text>
      )}

      <Card mode="outlined" style={styles.card}>
        <Card.Content>
          <DataTable>
            <DataTable.Row>
              <DataTable.Cell>Rows</DataTable.Cell>
              <DataTable.Cell numeric>
                {String(stats?.row_count ?? dataset?.row_count ?? "—")}
              </DataTable.Cell>
            </DataTable.Row>
            <DataTable.Row>
              <DataTable.Cell>Columns</DataTable.Cell>
              <DataTable.Cell numeric>
                {String(stats?.column_count ?? "—")}
              </DataTable.Cell>
            </DataTable.Row>
            <DataTable.Row>
              <DataTable.Cell>Missing values</DataTable.Cell>
              <DataTable.Cell numeric>
                {String(stats?.missing_values ?? "—")}
              </DataTable.Cell>
            </DataTable.Row>
            <DataTable.Row>
              <DataTable.Cell>Frequency</DataTable.Cell>
              <DataTable.Cell numeric>
                {dataset?.frequency ?? "—"}
              </DataTable.Cell>
            </DataTable.Row>
          </DataTable>
        </Card.Content>
      </Card>
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  header: { flexDirection: "row", alignItems: "center", gap: 12 },
  card: { marginTop: 20 },
});

export default DatasetDetailScreen;
