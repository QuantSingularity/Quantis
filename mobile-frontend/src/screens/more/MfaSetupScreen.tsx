import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useState } from "react";
import { Image, StyleSheet, View } from "react-native";
import {
  Button,
  Chip,
  HelperText,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import { authAPI, getErrorMessage } from "../../api";
import ConfirmDialog from "../../components/ConfirmDialog";
import ScreenContainer from "../../components/ScreenContainer";
import { useAuth } from "../../context/AuthContext";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "MfaSetup">;

interface MfaSetupData {
  qr_code_svg: string;
  secret: string;
}

const MfaSetupScreen: React.FC<Props> = () => {
  const theme = useTheme();
  const { user, refreshUser } = useAuth();
  const [setupData, setSetupData] = useState<MfaSetupData | null>(null);
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [disableOpen, setDisableOpen] = useState(false);
  const [disableCode, setDisableCode] = useState("");

  const startSetup = async () => {
    setError(null);
    try {
      const { data } = await authAPI.setupMfa();
      setSetupData(data);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const confirmEnable = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await authAPI.enableMfa(password, code);
      await refreshUser();
      setSetupData(null);
      setPassword("");
      setCode("");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleDisable = async () => {
    try {
      await authAPI.disableMfa(disableCode);
      await refreshUser();
      setDisableOpen(false);
      setDisableCode("");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  return (
    <ScreenContainer>
      <View style={styles.headerRow}>
        <Text variant="headlineSmall" style={{ fontWeight: "700" }}>
          Two-factor authentication
        </Text>
        <Chip
          compact
          style={{
            backgroundColor: user?.is_mfa_enabled ? "#22C55E22" : undefined,
          }}
        >
          {user?.is_mfa_enabled ? "Enabled" : "Disabled"}
        </Chip>
      </View>

      {error && (
        <HelperText type="error" visible>
          {error}
        </HelperText>
      )}

      {user?.is_mfa_enabled ? (
        <View style={{ marginTop: 16 }}>
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.onSurfaceVariant, marginBottom: 16 }}
          >
            Two-factor authentication is protecting your account.
          </Text>
          <Button
            mode="outlined"
            textColor={theme.colors.error}
            onPress={() => setDisableOpen(true)}
          >
            Disable 2FA
          </Button>
        </View>
      ) : setupData ? (
        <View style={{ marginTop: 16 }}>
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.onSurfaceVariant, marginBottom: 16 }}
          >
            Scan this QR code with your authenticator app, then confirm with a
            6-digit code.
          </Text>
          <Image
            source={{ uri: `data:image/png;base64,${setupData.qr_code_svg}` }}
            style={styles.qrImage}
          />
          <Text
            variant="bodySmall"
            style={{ color: theme.colors.onSurfaceVariant, marginBottom: 16 }}
          >
            Manual entry key: {setupData.secret}
          </Text>
          <TextInput
            label="Account password"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            mode="outlined"
            style={styles.input}
          />
          <TextInput
            label="6-digit code"
            value={code}
            onChangeText={setCode}
            keyboardType="number-pad"
            maxLength={6}
            mode="outlined"
            style={styles.input}
          />
          <Button
            mode="contained"
            onPress={confirmEnable}
            loading={submitting}
            disabled={submitting}
          >
            Confirm & enable
          </Button>
        </View>
      ) : (
        <View style={{ marginTop: 16 }}>
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.onSurfaceVariant, marginBottom: 16 }}
          >
            Add an extra layer of security to your account with an authenticator
            app.
          </Text>
          <Button mode="contained" onPress={startSetup}>
            Set up 2FA
          </Button>
        </View>
      )}

      <ConfirmDialog
        visible={disableOpen}
        title="Disable two-factor authentication?"
        description="This will make your account less secure."
        confirmLabel="Disable"
        destructive
        onConfirm={handleDisable}
        onDismiss={() => setDisableOpen(false)}
      />
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  input: { marginBottom: 12 },
  qrImage: { width: 200, height: 200, marginBottom: 16, borderRadius: 8 },
});

export default MfaSetupScreen;
