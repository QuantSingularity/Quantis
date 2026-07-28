import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import { useFocusEffect } from "@react-navigation/native";
import React, { useCallback, useState } from "react";
import { StyleSheet, View } from "react-native";
import { Card, FAB, Text, useTheme } from "react-native-paper";
import { datasetsAPI, getErrorMessage } from "../api";
import { Dataset } from "../api/types";
import EmptyState from "../components/EmptyState";
import LoadingScreen from "../components/LoadingScreen";
import ScreenContainer from "../components/ScreenContainer";
import StatusChip from "../components/StatusChip";
import type { DatasetsStackParamList } from "../navigation/types";

type Props = NativeStackScreenProps<DatasetsStackParamList, "DatasetsList">;

const DatasetsListScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const { data } = await datasetsAPI.list();
      setDatasets(
        (data as unknown as { items?: Dataset[] })?.items ??
          (data as Dataset[]) ??
          [],
      );
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      load().finally(() => setLoading(false));
    }, [load]),
  );

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  if (loading) return <LoadingScreen label="Loading datasets…" />;

  return (
    <View style={styles.flex}>
      <ScreenContainer refreshing={refreshing} onRefresh={onRefresh}>
        {error && (
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.error, marginBottom: 12 }}
          >
            {error}
          </Text>
        )}

        {datasets.length === 0 ? (
          <EmptyState
            title="No datasets yet"
            description="Upload a dataset from the web app to get started, or check back after your team adds one."
          />
        ) : (
          datasets.map((dataset) => (
            <Card
              key={dataset.id}
              mode="outlined"
              style={styles.card}
              onPress={() =>
                navigation.navigate("DatasetDetail", { datasetId: dataset.id })
              }
            >
              <Card.Content>
                <View style={styles.cardHeader}>
                  <Text
                    variant="titleMedium"
                    style={{ fontWeight: "700", flex: 1 }}
                  >
                    {dataset.name}
                  </Text>
                  <StatusChip status={dataset.status} />
                </View>
                <Text
                  variant="bodySmall"
                  style={{ color: theme.colors.onSurfaceVariant, marginTop: 4 }}
                >
                  {dataset.row_count ?? "—"} rows · {dataset.frequency ?? "—"}
                </Text>
              </Card.Content>
            </Card>
          ))
        )}
      </ScreenContainer>
    </View>
  );
};

const styles = StyleSheet.create({
  flex: { flex: 1 },
  card: { marginBottom: 12 },
  cardHeader: { flexDirection: "row", alignItems: "center", gap: 8 },
});

export default DatasetsListScreen;
