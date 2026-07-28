import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import { useFocusEffect } from "@react-navigation/native";
import React, { useCallback, useState } from "react";
import { StyleSheet, View } from "react-native";
import { Card, Chip, Text, useTheme } from "react-native-paper";
import { modelsAPI, getErrorMessage } from "../api";
import { Model } from "../api/types";
import EmptyState from "../components/EmptyState";
import LoadingScreen from "../components/LoadingScreen";
import ScreenContainer from "../components/ScreenContainer";
import StatusChip from "../components/StatusChip";
import type { ModelsStackParamList } from "../navigation/types";

type Props = NativeStackScreenProps<ModelsStackParamList, "ModelsList">;

const ModelsListScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const { data } = await modelsAPI.list();
      setModels(
        (data as unknown as { items?: Model[] })?.items ??
          (data as Model[]) ??
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

  if (loading) return <LoadingScreen label="Loading models…" />;

  return (
    <ScreenContainer refreshing={refreshing} onRefresh={onRefresh}>
      {error && (
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.error, marginBottom: 12 }}
        >
          {error}
        </Text>
      )}

      {models.length === 0 ? (
        <EmptyState
          title="No models yet"
          description="Create a model from the web app to start forecasting."
        />
      ) : (
        models.map((model) => (
          <Card
            key={model.id}
            mode="outlined"
            style={styles.card}
            onPress={() =>
              navigation.navigate("ModelDetail", { modelId: model.id })
            }
          >
            <Card.Content>
              <View style={styles.cardHeader}>
                <Text
                  variant="titleMedium"
                  style={{ fontWeight: "700", flex: 1 }}
                >
                  {model.name}
                </Text>
                <StatusChip status={model.status} />
              </View>
              <View style={styles.chipRow}>
                <Chip compact textStyle={styles.typeChip}>
                  {String(model.model_type).replace(/_/g, " ")}
                </Chip>
                <Chip compact mode="outlined">
                  v{model.version}
                </Chip>
              </View>
            </Card.Content>
          </Card>
        ))
      )}
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  card: { marginBottom: 12 },
  cardHeader: { flexDirection: "row", alignItems: "center", gap: 8 },
  chipRow: { flexDirection: "row", gap: 8, marginTop: 8 },
  typeChip: { textTransform: "capitalize" },
});

export default ModelsListScreen;
