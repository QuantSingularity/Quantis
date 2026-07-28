import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useState } from "react";
import { StyleSheet } from "react-native";
import { Button, HelperText, Text, TextInput } from "react-native-paper";
import { authAPI, getErrorMessage } from "../../api";
import ScreenContainer from "../../components/ScreenContainer";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "ChangePassword">;

const ChangePasswordScreen: React.FC<Props> = ({ navigation }) => {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async () => {
    setError(null);
    if (next !== confirm) {
      setError("New passwords do not match");
      return;
    }
    setSaving(true);
    try {
      await authAPI.changePassword(current, next);
      setSuccess(true);
      setTimeout(() => navigation.goBack(), 1200);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <ScreenContainer>
      <Text variant="headlineSmall" style={styles.title}>
        Change password
      </Text>

      {error && (
        <HelperText type="error" visible>
          {error}
        </HelperText>
      )}
      {success && (
        <HelperText type="info" visible>
          Password updated successfully
        </HelperText>
      )}

      <TextInput
        label="Current password"
        value={current}
        onChangeText={setCurrent}
        secureTextEntry
        mode="outlined"
        style={styles.input}
      />
      <TextInput
        label="New password"
        value={next}
        onChangeText={setNext}
        secureTextEntry
        mode="outlined"
        style={styles.input}
        right={<TextInput.Affix text="" />}
      />
      <HelperText
        type="info"
        visible
        style={{ marginTop: -8, marginBottom: 8 }}
      >
        At least 12 characters, mixed case, digit, and symbol
      </HelperText>
      <TextInput
        label="Confirm new password"
        value={confirm}
        onChangeText={setConfirm}
        secureTextEntry
        mode="outlined"
        style={styles.input}
      />

      <Button
        mode="contained"
        onPress={handleSubmit}
        loading={saving}
        disabled={saving}
      >
        Update password
      </Button>
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  title: { fontWeight: "700", marginBottom: 16 },
  input: { marginBottom: 12 },
});

export default ChangePasswordScreen;
