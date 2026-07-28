import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useCallback, useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import { Button, Card, Chip, Text, useTheme } from "react-native-paper";
import { getErrorMessage, modelsAPI } from "../api";
import { Model } from "../api/types";
import LoadingScreen from "../components/LoadingScreen";
import ScreenContainer from "../components/ScreenContainer";
import StatCard from "../components/StatCard";
import StatusChip from "../components/StatusChip";
import type { ModelsStackParamList } from "../navigation/types";

type Props = NativeStackScreenProps<ModelsStackParamList, "ModelDetail">;

const formatMetric = (value: unknown): string => {
  if (typeof value !== "number") return String(value ?? "—");
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
};

const ModelDetailScreen: React.FC<Props> = ({ route }) => {
  const { modelId } = route.params;
  const theme = useTheme();
  const [model, setModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const { data } = await modelsAPI.get(modelId);
      setModel(data);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }, [modelId]);

  useEffect(() => {
    setLoading(true);
    load().finally(() => setLoading(false));
  }, [load]);

  const handleTrain = async () => {
    setTraining(true);
    try {
      await modelsAPI.train(modelId);
      await load();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setTraining(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading model…" />;
  if (error && !model) {
    return (
      <ScreenContainer>
        <Text style={{ color: theme.colors.error }}>{error}</Text>
      </ScreenContainer>
    );
  }

  const canTrain =
    model && ["created", "failed"].includes(model.status) && !training;
  const metricEntries = Object.entries(model?.metrics || {});

  return (
    <ScreenContainer>
      <View style={styles.header}>
        <Text variant="headlineSmall" style={{ fontWeight: "700", flex: 1 }}>
          {model?.name}
        </Text>
        <StatusChip status={training ? "training" : model?.status} />
      </View>
      <View style={styles.chipRow}>
        <Chip compact textStyle={{ textTransform: "capitalize" }}>
          {String(model?.model_type).replace(/_/g, " ")}
        </Chip>
        <Chip compact mode="outlined">
          v{model?.version}
        </Chip>
      </View>

      {model?.description && (
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.onSurfaceVariant, marginTop: 12 }}
        >
          {model.description}
        </Text>
      )}

      <Button
        mode="contained"
        onPress={handleTrain}
        disabled={!canTrain}
        loading={training}
        style={{ marginTop: 16 }}
      >
        {model?.status === "failed" ? "Retry training" : "Train model"}
      </Button>

      <Text
        variant="titleMedium"
        style={{ fontWeight: "700", marginTop: 24, marginBottom: 12 }}
      >
        Performance metrics
      </Text>
      {metricEntries.length === 0 ? (
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.onSurfaceVariant }}
        >
          No metrics available yet. Train the model to generate them.
        </Text>
      ) : (
        <View style={styles.metricsGrid}>
          {metricEntries.map(([key, value]) => (
            <StatCard
              key={key}
              label={key.replace(/_/g, " ")}
              value={formatMetric(value)}
            />
          ))}
        </View>
      )}

      <Card mode="outlined" style={styles.configCard}>
        <Card.Content>
          <Text
            variant="titleMedium"
            style={{ fontWeight: "700", marginBottom: 8 }}
          >
            Configuration
          </Text>
          <View style={styles.configRow}>
            <Text
              variant="bodyMedium"
              style={{ color: theme.colors.onSurfaceVariant }}
            >
              Dataset ID
            </Text>
            <Text variant="bodyMedium">{model?.dataset_id}</Text>
          </View>
          <View style={styles.configRow}>
            <Text
              variant="bodyMedium"
              style={{ color: theme.colors.onSurfaceVariant }}
            >
              Trained at
            </Text>
            <Text variant="bodyMedium">
              {model?.trained_at
                ? new Date(model.trained_at).toLocaleString()
                : "Not trained yet"}
            </Text>
          </View>
        </Card.Content>
      </Card>
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  header: { flexDirection: "row", alignItems: "center", gap: 12 },
  chipRow: { flexDirection: "row", gap: 8, marginTop: 10 },
  metricsGrid: { flexDirection: "row", flexWrap: "wrap", gap: 12 },
  configCard: { marginTop: 20, marginBottom: 20 },
  configRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 6,
  },
});

export default ModelDetailScreen;
