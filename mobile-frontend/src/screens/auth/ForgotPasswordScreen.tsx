import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useState } from "react";
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  View,
} from "react-native";
import {
  Button,
  HelperText,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import Logo from "../../components/Logo";
import { authAPI, getErrorMessage } from "../../api";
import type { AuthStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<AuthStackParamList, "ForgotPassword">;

const ForgotPasswordScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [devToken, setDevToken] = useState<string | null>(null);

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const { data } = await authAPI.forgotPassword(email);
      setSent(true);
      if (data.reset_token) setDevToken(data.reset_token);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  if (sent) {
    return (
      <View
        style={[
          styles.successContainer,
          { backgroundColor: theme.colors.background },
        ]}
      >
        <Text
          variant="headlineSmall"
          style={{ fontWeight: "700", textAlign: "center" }}
        >
          Check your email
        </Text>
        <Text
          variant="bodyMedium"
          style={{
            color: theme.colors.onSurfaceVariant,
            marginTop: 8,
            textAlign: "center",
          }}
        >
          If an account exists for {email}, we&apos;ve sent a password reset
          link.
        </Text>
        {devToken && (
          <Text
            variant="bodySmall"
            style={{
              color: theme.colors.onSurfaceVariant,
              marginTop: 16,
              textAlign: "center",
            }}
          >
            Dev mode token: {devToken.slice(0, 24)}…
          </Text>
        )}
        <Button
          mode="contained"
          onPress={() => navigation.navigate("Login")}
          style={{ marginTop: 24 }}
        >
          Back to sign in
        </Button>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={[styles.flex, { backgroundColor: theme.colors.background }]}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.header}>
          <Logo size={40} />
          <Text variant="headlineSmall" style={styles.title}>
            Reset your password
          </Text>
          <Text
            variant="bodyMedium"
            style={{
              color: theme.colors.onSurfaceVariant,
              textAlign: "center",
            }}
          >
            Enter your email and we&apos;ll send you a reset link
          </Text>
        </View>

        {error && (
          <HelperText type="error" visible style={styles.errorText}>
            {error}
          </HelperText>
        )}

        <TextInput
          label="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
          mode="outlined"
          style={styles.input}
        />

        <Button
          mode="contained"
          onPress={handleSubmit}
          loading={submitting}
          disabled={submitting || !email.includes("@")}
          contentStyle={styles.buttonContent}
        >
          Send reset link
        </Button>

        <Button
          compact
          onPress={() => navigation.navigate("Login")}
          style={{ marginTop: 16, alignSelf: "center" }}
        >
          Back to sign in
        </Button>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { padding: 24, flexGrow: 1, justifyContent: "center" },
  header: { alignItems: "center", gap: 6, marginBottom: 20 },
  title: { fontWeight: "700", marginTop: 12, textAlign: "center" },
  input: { marginBottom: 12 },
  errorText: { marginBottom: 8 },
  buttonContent: { paddingVertical: 6 },
  successContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
});

export default ForgotPasswordScreen;
