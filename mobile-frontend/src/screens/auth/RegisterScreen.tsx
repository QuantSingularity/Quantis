import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useMemo, useState } from "react";
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
  ProgressBar,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import Logo from "../../components/Logo";
import { useAuth } from "../../context/AuthContext";
import type { AuthStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<AuthStackParamList, "Register">;

const PASSWORD_RULES = [
  { test: (v: string) => v.length >= 12, label: "At least 12 characters" },
  { test: (v: string) => /[A-Z]/.test(v), label: "One uppercase letter" },
  { test: (v: string) => /[a-z]/.test(v), label: "One lowercase letter" },
  { test: (v: string) => /\d/.test(v), label: "One digit" },
  {
    test: (v: string) => /[!@#$%^&*()_+\-=[\]{}|;:,.<>?]/.test(v),
    label: "One special character",
  },
];

const RegisterScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();
  const { register, authError, setAuthError } = useAuth();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  const passedRules = useMemo(
    () => PASSWORD_RULES.filter((rule) => rule.test(password)).length,
    [password],
  );
  const strength = passedRules / PASSWORD_RULES.length;
  const passwordsMatch = password.length > 0 && password === confirmPassword;

  const canSubmit =
    username.length >= 3 &&
    email.includes("@") &&
    passedRules === PASSWORD_RULES.length &&
    passwordsMatch;

  const handleSubmit = async () => {
    setAuthError(null);
    setSubmitting(true);
    const result = await register({
      username,
      email,
      password,
      confirm_password: confirmPassword,
    });
    setSubmitting(false);
    if (result.success) {
      setSuccess(true);
      setTimeout(() => navigation.navigate("Login"), 1200);
    }
  };

  if (success) {
    return (
      <View
        style={[
          styles.successContainer,
          { backgroundColor: theme.colors.background },
        ]}
      >
        <Text variant="headlineSmall" style={{ fontWeight: "700" }}>
          Account created 🎉
        </Text>
        <Text
          variant="bodyMedium"
          style={{ color: theme.colors.onSurfaceVariant, marginTop: 8 }}
        >
          Taking you to sign in…
        </Text>
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
            Create your account
          </Text>
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.onSurfaceVariant }}
          >
            Start forecasting in minutes
          </Text>
        </View>

        {authError && (
          <HelperText type="error" visible style={styles.errorText}>
            {authError}
          </HelperText>
        )}

        <TextInput
          label="Username"
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
          mode="outlined"
          style={styles.input}
        />
        <TextInput
          label="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
          mode="outlined"
          style={styles.input}
        />
        <TextInput
          label="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          mode="outlined"
          style={styles.input}
        />

        {password.length > 0 && (
          <View style={styles.strengthContainer}>
            <ProgressBar
              progress={strength}
              color={strength === 1 ? theme.colors.primary : theme.colors.error}
              style={styles.progressBar}
            />
            {PASSWORD_RULES.map((rule) => (
              <Text
                key={rule.label}
                variant="bodySmall"
                style={{
                  color: rule.test(password)
                    ? theme.colors.primary
                    : theme.colors.onSurfaceVariant,
                }}
              >
                {rule.test(password) ? "✓" : "○"} {rule.label}
              </Text>
            ))}
          </View>
        )}

        <TextInput
          label="Confirm password"
          value={confirmPassword}
          onChangeText={setConfirmPassword}
          secureTextEntry
          mode="outlined"
          style={styles.input}
          error={confirmPassword.length > 0 && !passwordsMatch}
        />
        {confirmPassword.length > 0 && !passwordsMatch && (
          <HelperText type="error" visible>
            Passwords do not match
          </HelperText>
        )}

        <Button
          mode="contained"
          onPress={handleSubmit}
          loading={submitting}
          disabled={submitting || !canSubmit}
          contentStyle={styles.buttonContent}
          style={{ marginTop: 8 }}
        >
          Create account
        </Button>

        <View style={styles.footer}>
          <Text
            variant="bodyMedium"
            style={{ color: theme.colors.onSurfaceVariant }}
          >
            Already have an account?{" "}
          </Text>
          <Button compact onPress={() => navigation.navigate("Login")}>
            Sign in
          </Button>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { padding: 24, flexGrow: 1, justifyContent: "center" },
  header: { alignItems: "center", gap: 6, marginBottom: 20 },
  title: { fontWeight: "700", marginTop: 12 },
  input: { marginBottom: 12 },
  errorText: { marginBottom: 8 },
  strengthContainer: { marginBottom: 12, gap: 2 },
  progressBar: { height: 6, borderRadius: 3, marginBottom: 6 },
  buttonContent: { paddingVertical: 6 },
  footer: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    marginTop: 16,
  },
  successContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
});

export default RegisterScreen;
