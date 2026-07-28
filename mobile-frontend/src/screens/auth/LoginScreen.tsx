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
import { useAuth } from "../../context/AuthContext";
import type { AuthStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<AuthStackParamList, "Login">;

const LoginScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();
  const { login, authError, setAuthError } = useAuth();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [mfaCode, setMfaCode] = useState("");
  const [needsMfa, setNeedsMfa] = useState(false);
  const [secureText, setSecureText] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    setAuthError(null);
    setSubmitting(true);
    const result = await login(username, password, mfaCode || undefined);
    setSubmitting(false);
    if (
      !result.success &&
      result.status === 403 &&
      /mfa/i.test(result.error || "")
    ) {
      setNeedsMfa(true);
    }
    // On success, RootNavigator swaps to the Main stack automatically because
    // AuthContext.user becomes truthy.
  };

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
            Welcome back
          </Text>
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.onSurfaceVariant }}
          >
            Sign in to access your Quantis workspace
          </Text>
        </View>

        {authError && (
          <HelperText type="error" visible style={styles.errorText}>
            {authError}
          </HelperText>
        )}

        <TextInput
          label="Username or email"
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
          mode="outlined"
          style={styles.input}
        />
        <TextInput
          label="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry={secureText}
          mode="outlined"
          style={styles.input}
          right={
            <TextInput.Icon
              icon={secureText ? "eye" : "eye-off"}
              onPress={() => setSecureText((s) => !s)}
            />
          }
        />

        {needsMfa && (
          <TextInput
            label="Authenticator code"
            value={mfaCode}
            onChangeText={setMfaCode}
            keyboardType="number-pad"
            maxLength={6}
            mode="outlined"
            style={styles.input}
          />
        )}

        <Button
          onPress={() => navigation.navigate("ForgotPassword")}
          style={styles.forgotButton}
          compact
        >
          Forgot password?
        </Button>

        <Button
          mode="contained"
          onPress={handleSubmit}
          loading={submitting}
          disabled={submitting || !username || !password}
          contentStyle={styles.buttonContent}
        >
          Sign in
        </Button>

        <View style={styles.footer}>
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.onSurfaceVariant }}
          >
            Don&apos;t have an account?{" "}
          </Text>
          <Button compact onPress={() => navigation.navigate("Register")}>
            Create one
          </Button>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { padding: 24, flexGrow: 1, justifyContent: "center" },
  header: { alignItems: "center", gap: 6, marginBottom: 24 },
  title: { fontWeight: "700", marginTop: 12 },
  input: { marginBottom: 12 },
  errorText: { marginBottom: 8 },
  forgotButton: { alignSelf: "flex-end", marginBottom: 8 },
  buttonContent: { paddingVertical: 6 },
  footer: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    marginTop: 16,
  },
});

export default LoginScreen;
