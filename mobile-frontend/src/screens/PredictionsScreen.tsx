import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import { useFocusEffect } from "@react-navigation/native";
import React, { useCallback, useState } from "react";
import { StyleSheet, View } from "react-native";
import {
  Button,
  Card,
  Menu,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import { getErrorMessage, modelsAPI, predictionsAPI } from "../api";
import { Model, Prediction } from "../api/types";
import EmptyState from "../components/EmptyState";
import LoadingScreen from "../components/LoadingScreen";
import ScreenContainer from "../components/ScreenContainer";
import type { PredictionsStackParamList } from "../navigation/types";

type Props = NativeStackScreenProps<
  PredictionsStackParamList,
  "PredictionsHome"
>;

const PredictionsScreen: React.FC<Props> = () => {
  const theme = useTheme();
  const [models, setModels] = useState<Model[]>([]);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [menuVisible, setMenuVisible] = useState(false);
  const [inputJson, setInputJson] = useState('{\n  "feature_1": 0.5\n}');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<unknown>(null);

  const load = useCallback(async () => {
    const [modelsRes, historyRes] = await Promise.allSettled([
      modelsAPI.list(),
      predictionsAPI.history({ limit: 20 }),
    ]);
    if (modelsRes.status === "fulfilled") {
      const list =
        (modelsRes.value.data as unknown as { items?: Model[] })?.items ??
        (modelsRes.value.data as Model[]) ??
        [];
      setModels(list);
      const trained = list.find(
        (m) => m.status === "trained" || m.status === "deployed",
      );
      if (trained) setSelectedModel((prev) => prev || trained);
    }
    if (historyRes.status === "fulfilled") {
      setHistory(
        (historyRes.value.data as unknown as { items?: Prediction[] })?.items ??
          (historyRes.value.data as Prediction[]) ??
          [],
      );
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      load().finally(() => setLoading(false));
    }, [load]),
  );

  const trainedModels = models.filter(
    (m) => m.status === "trained" || m.status === "deployed",
  );

  const handlePredict = async () => {
    setError(null);
    setResult(null);
    if (!selectedModel) return;
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(inputJson);
    } catch {
      setError('Input must be valid JSON, e.g. {"feature_1": 0.5}');
      return;
    }
    setSubmitting(true);
    try {
      const { data } = await predictionsAPI.predict(selectedModel.id, parsed);
      setResult(data.prediction_result ?? data);
      load();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading predictions…" />;

  return (
    <ScreenContainer>
      <Text
        variant="headlineSmall"
        style={{ fontWeight: "700", marginBottom: 4 }}
      >
        Predictions
      </Text>
      <Text
        variant="bodyMedium"
        style={{ color: theme.colors.onSurfaceVariant, marginBottom: 16 }}
      >
        Run predictions and review recent history.
      </Text>

      {trainedModels.length === 0 ? (
        <EmptyState
          title="No trained models"
          description="Train a model on the web app first."
        />
      ) : (
        <Card mode="outlined" style={styles.formCard}>
          <Card.Content>
            {error && (
              <Text
                variant="bodySmall"
                style={{ color: theme.colors.error, marginBottom: 8 }}
              >
                {error}
              </Text>
            )}
            <Menu
              visible={menuVisible}
              onDismiss={() => setMenuVisible(false)}
              anchor={
                <Button
                  mode="outlined"
                  onPress={() => setMenuVisible(true)}
                  style={styles.modelSelector}
                >
                  {selectedModel ? selectedModel.name : "Select a model"}
                </Button>
              }
            >
              {trainedModels.map((m) => (
                <Menu.Item
                  key={m.id}
                  onPress={() => {
                    setSelectedModel(m);
                    setMenuVisible(false);
                  }}
                  title={m.name}
                />
              ))}
            </Menu>

            <TextInput
              label="Input data (JSON)"
              value={inputJson}
              onChangeText={setInputJson}
              mode="outlined"
              multiline
              numberOfLines={5}
              style={styles.jsonInput}
            />

            <Button
              mode="contained"
              onPress={handlePredict}
              loading={submitting}
              disabled={submitting}
            >
              Run prediction
            </Button>

            {result !== null && (
              <View
                style={[
                  styles.resultBox,
                  { backgroundColor: theme.colors.surfaceVariant },
                ]}
              >
                <Text variant="bodySmall" style={styles.resultText}>
                  {JSON.stringify(result, null, 2)}
                </Text>
              </View>
            )}
          </Card.Content>
        </Card>
      )}

      <Text
        variant="titleMedium"
        style={{ fontWeight: "700", marginTop: 24, marginBottom: 12 }}
      >
        History
      </Text>
      {history.length === 0 ? (
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.onSurfaceVariant }}
        >
          No predictions yet.
        </Text>
      ) : (
        history.map((p) => (
          <Card key={p.id} mode="outlined" style={styles.historyCard}>
            <Card.Content>
              <Text variant="bodyMedium" style={{ fontWeight: "600" }}>
                Model #{p.model_id}
              </Text>
              <Text
                variant="bodySmall"
                style={{ color: theme.colors.onSurfaceVariant }}
              >
                {p.created_at ? new Date(p.created_at).toLocaleString() : "-"}
              </Text>
            </Card.Content>
          </Card>
        ))
      )}
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  formCard: { marginBottom: 8 },
  modelSelector: { marginBottom: 12, justifyContent: "flex-start" },
  jsonInput: { marginBottom: 12, fontFamily: "monospace" },
  resultBox: { marginTop: 12, padding: 12, borderRadius: 8 },
  resultText: { fontFamily: "monospace" },
  historyCard: { marginBottom: 8 },
});

export default PredictionsScreen;
