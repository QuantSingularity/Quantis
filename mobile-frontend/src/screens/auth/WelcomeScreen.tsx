import React from "react";
import { StyleSheet, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Button, Text, useTheme } from "react-native-paper";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import Logo from "../../components/Logo";
import type { AuthStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<AuthStackParamList, "Welcome">;

const FEATURES = [
  "Unified dataset & model pipeline",
  "Real-time forecasting predictions",
  "Built-in financial risk tooling",
  "Enterprise-grade security & MFA",
];

const WelcomeScreen: React.FC<Props> = ({ navigation }) => {
  const theme = useTheme();

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: theme.colors.background }]}
    >
      <View style={styles.content}>
        <Logo size={56} />
        <Text
          variant="displaySmall"
          style={[styles.headline, { color: theme.colors.onBackground }]}
        >
          Financial forecasting, from data to decision.
        </Text>
        <Text
          variant="bodyLarge"
          style={[styles.subhead, { color: theme.colors.onSurfaceVariant }]}
        >
          Manage datasets, train models, and ship predictions - all from your
          pocket.
        </Text>

        <View style={styles.featureList}>
          {FEATURES.map((feature) => (
            <View style={styles.featureRow} key={feature}>
              <View
                style={[styles.dot, { backgroundColor: theme.colors.primary }]}
              />
              <Text
                variant="bodyMedium"
                style={{ color: theme.colors.onSurfaceVariant }}
              >
                {feature}
              </Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.actions}>
        <Button
          mode="contained"
          onPress={() => navigation.navigate("Register")}
          contentStyle={styles.buttonContent}
        >
          Get started
        </Button>
        <Button
          mode="outlined"
          onPress={() => navigation.navigate("Login")}
          contentStyle={styles.buttonContent}
        >
          I already have an account
        </Button>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "space-between", padding: 24 },
  content: { flex: 1, justifyContent: "center", gap: 16 },
  headline: { fontWeight: "800", lineHeight: 40 },
  subhead: { marginTop: 4 },
  featureList: { marginTop: 24, gap: 12 },
  featureRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  dot: { width: 6, height: 6, borderRadius: 3 },
  actions: { gap: 12, paddingBottom: 8 },
  buttonContent: { paddingVertical: 6 },
});

export default WelcomeScreen;
