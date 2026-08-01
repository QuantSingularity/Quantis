import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import {
  Button,
  Dialog,
  HelperText,
  IconButton,
  List,
  Portal,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import { authAPI, getErrorMessage } from "../../api";
import { ApiKey } from "../../api/types";
import ConfirmDialog from "../../components/ConfirmDialog";
import ScreenContainer from "../../components/ScreenContainer";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "ApiKeys">;

const ApiKeysScreen: React.FC<Props> = () => {
  const theme = useTheme();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [keyName, setKeyName] = useState("");
  const [creating, setCreating] = useState(false);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [revokeTarget, setRevokeTarget] = useState<ApiKey | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const { data } = await authAPI.listApiKeys();
      setKeys(data);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async () => {
    setCreating(true);
    try {
      const { data } = await authAPI.createApiKey(keyName);
      setNewKey(data.key);
      setKeyName("");
      load();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async () => {
    if (!revokeTarget) return;
    try {
      await authAPI.revokeApiKey(revokeTarget.id);
      setKeys((prev) => prev.filter((k) => k.id !== revokeTarget.id));
      setRevokeTarget(null);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  return (
    <ScreenContainer>
      <View style={styles.headerRow}>
        <Text variant="headlineSmall" style={{ fontWeight: "700" }}>
          API keys
        </Text>
        <Button mode="contained" compact onPress={() => setCreateOpen(true)}>
          New key
        </Button>
      </View>

      {error && (
        <HelperText type="error" visible>
          {error}
        </HelperText>
      )}

      {!loading && keys.length === 0 && (
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.onSurfaceVariant, marginTop: 12 }}
        >
          You don&apos;t have any API keys yet.
        </Text>
      )}

      {keys.map((key) => (
        <List.Item
          key={key.id}
          title={key.name}
          description={`${key.key_preview} · Created ${new Date(key.created_at).toLocaleDateString()}`}
          right={(props) => (
            <IconButton
              {...props}
              icon="delete-outline"
              onPress={() => setRevokeTarget(key)}
            />
          )}
        />
      ))}

      <Portal>
        <Dialog visible={createOpen} onDismiss={() => setCreateOpen(false)}>
          <Dialog.Title>
            {newKey ? "API key created" : "New API key"}
          </Dialog.Title>
          <Dialog.Content>
            {newKey ? (
              <View>
                <HelperText type="info" visible>
                  Copy this key now - you won&apos;t be able to see it again.
                </HelperText>
                <TextInput
                  value={newKey}
                  mode="outlined"
                  multiline
                  editable={false}
                />
              </View>
            ) : (
              <TextInput
                label="Key name"
                value={keyName}
                onChangeText={setKeyName}
                mode="outlined"
                placeholder="e.g. Mobile app"
              />
            )}
          </Dialog.Content>
          <Dialog.Actions>
            {newKey ? (
              <Button
                onPress={() => {
                  setNewKey(null);
                  setCreateOpen(false);
                }}
              >
                Done
              </Button>
            ) : (
              <>
                <Button onPress={() => setCreateOpen(false)}>Cancel</Button>
                <Button
                  onPress={handleCreate}
                  loading={creating}
                  disabled={creating || !keyName}
                >
                  Create
                </Button>
              </>
            )}
          </Dialog.Actions>
        </Dialog>
      </Portal>

      <ConfirmDialog
        visible={Boolean(revokeTarget)}
        title="Revoke API key?"
        description={`"${revokeTarget?.name}" will stop working immediately.`}
        confirmLabel="Revoke"
        destructive
        onConfirm={handleRevoke}
        onDismiss={() => setRevokeTarget(null)}
      />
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
});

export default ApiKeysScreen;
