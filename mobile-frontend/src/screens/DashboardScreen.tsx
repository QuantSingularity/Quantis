import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useCallback, useState } from "react";
import { useFocusEffect } from "@react-navigation/native";
import { StyleSheet, View } from "react-native";
import {
  Button,
  Card,
  Divider,
  List,
  Text,
  useTheme,
} from "react-native-paper";
import { datasetsAPI, financialAPI, modelsAPI, predictionsAPI } from "../api";
import { Dataset, Model, Prediction } from "../api/types";
import LoadingScreen from "../components/LoadingScreen";
import ScreenContainer from "../components/ScreenContainer";
import StatCard from "../components/StatCard";
import StatusChip from "../components/StatusChip";
import { useAuth } from "../context/AuthContext";
import type { DashboardStackParamList } from "../navigation/types";

type Props = NativeStackScreenProps<DashboardStackParamList, "DashboardHome">;

const DashboardScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [txVolume, setTxVolume] = useState<string>("$0");

  const loadData = useCallback(async () => {
    const results = await Promise.allSettled([
      datasetsAPI.list({ limit: 5 }),
      modelsAPI.list({ limit: 5 }),
      predictionsAPI.history({ limit: 5 }),
      financialAPI.summary(),
    ]);
    const [datasetsRes, modelsRes, predictionsRes, financialRes] = results;
    if (datasetsRes.status === "fulfilled") {
      setDatasets(
        (datasetsRes.value.data as unknown as { items?: Dataset[] })?.items ??
          (datasetsRes.value.data as Dataset[]) ??
          [],
      );
    }
    if (modelsRes.status === "fulfilled") {
      setModels(
        (modelsRes.value.data as unknown as { items?: Model[] })?.items ??
          (modelsRes.value.data as Model[]) ??
          [],
      );
    }
    if (predictionsRes.status === "fulfilled") {
      setPredictions(
        (predictionsRes.value.data as unknown as { items?: Prediction[] })
          ?.items ??
          (predictionsRes.value.data as Prediction[]) ??
          [],
      );
    }
    if (financialRes.status === "fulfilled") {
      const summary = financialRes.value.data as { total_volume?: string };
      if (summary?.total_volume) setTxVolume(summary.total_volume);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      loadData().finally(() => setLoading(false));
    }, [loadData]),
  );

  const onRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  if (loading) return <LoadingScreen label="Loading your workspace…" />;

  const trainedModels = models.filter(
    (m) => m.status === "trained" || m.status === "deployed",
  );

  return (
    <ScreenContainer refreshing={refreshing} onRefresh={onRefresh}>
      <Text variant="headlineSmall" style={{ fontWeight: "700" }}>
        Welcome back, {user?.first_name || user?.username}
      </Text>
      <Text
        variant="bodyMedium"
        style={{
          color: theme.colors.onSurfaceVariant,
          marginTop: 4,
          marginBottom: 20,
        }}
      >
        Here&apos;s what&apos;s happening across your workspace.
      </Text>

      <View style={styles.statsGrid}>
        <StatCard label="Datasets" value={datasets.length} />
        <StatCard label="Trained models" value={trainedModels.length} />
        <StatCard label="Predictions" value={predictions.length} />
        <StatCard label="Tx volume" value={txVolume} />
      </View>

      <Card style={styles.section} mode="outlined">
        <Card.Title
          title="Recent predictions"
          right={() => (
            <Button
              compact
              onPress={() =>
                navigation.getParent()?.navigate("PredictionsTab" as never)
              }
            >
              View all
            </Button>
          )}
        />
        <Card.Content>
          {predictions.length === 0 ? (
            <Text
              variant="bodyMedium"
              style={{ color: theme.colors.onSurfaceVariant }}
            >
              No predictions yet.
            </Text>
          ) : (
            predictions.slice(0, 5).map((p, idx) => (
              <View key={p.id}>
                <List.Item
                  title={`Model #${p.model_id}`}
                  description={
                    p.created_at ? new Date(p.created_at).toLocaleString() : "-"
                  }
                  right={() => <StatusChip status={p.status || "completed"} />}
                  style={styles.listItem}
                />
                {idx < predictions.slice(0, 5).length - 1 && <Divider />}
              </View>
            ))
          )}
        </Card.Content>
      </Card>

      <Card style={styles.section} mode="outlined">
        <Card.Title
          title="Models"
          right={() => (
            <Button
              compact
              onPress={() =>
                navigation.getParent()?.navigate("ModelsTab" as never)
              }
            >
              Manage
            </Button>
          )}
        />
        <Card.Content>
          {models.length === 0 ? (
            <Text
              variant="bodyMedium"
              style={{ color: theme.colors.onSurfaceVariant }}
            >
              No models yet.
            </Text>
          ) : (
            models.slice(0, 5).map((m, idx) => (
              <View key={m.id}>
                <List.Item
                  title={m.name}
                  description={m.model_type}
                  right={() => <StatusChip status={m.status} />}
                  style={styles.listItem}
                />
                {idx < models.slice(0, 5).length - 1 && <Divider />}
              </View>
            ))
          )}
        </Card.Content>
      </Card>
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  statsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 20,
  },
  section: { marginBottom: 16 },
  listItem: { paddingHorizontal: 0 },
});

export default DashboardScreen;
