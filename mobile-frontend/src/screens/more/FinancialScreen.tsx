import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import React, { useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import {
  Button,
  Dialog,
  HelperText,
  Menu,
  Portal,
  Text,
  TextInput,
  useTheme,
} from "react-native-paper";
import { financialAPI, getErrorMessage } from "../../api";
import { Transaction } from "../../api/types";
import EmptyState from "../../components/EmptyState";
import LoadingScreen from "../../components/LoadingScreen";
import ScreenContainer from "../../components/ScreenContainer";
import StatCard from "../../components/StatCard";
import StatusChip from "../../components/StatusChip";
import type { MoreStackParamList } from "../../navigation/types";

type Props = NativeStackScreenProps<MoreStackParamList, "Financial">;

const TRANSACTION_TYPES = [
  "deposit",
  "withdrawal",
  "transfer",
  "payment",
  "trade",
  "fee",
];

const formatCurrency = (value: unknown): string => {
  if (typeof value !== "number") return String(value ?? "—");
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(value);
};

const FinancialScreen: React.FC<Props> = () => {
  const theme = useTheme();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [amount, setAmount] = useState("");
  const [txType, setTxType] = useState("deposit");
  const [typeMenuOpen, setTypeMenuOpen] = useState(false);
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    const [txRes, summaryRes] = await Promise.allSettled([
      financialAPI.listTransactions({ limit: 25 }),
      financialAPI.summary(),
    ]);
    if (txRes.status === "fulfilled") {
      setTransactions(
        (txRes.value.data as unknown as { items?: Transaction[] })?.items ??
          (txRes.value.data as Transaction[]) ??
          [],
      );
    } else {
      setError(getErrorMessage(txRes.reason));
    }
    if (summaryRes.status === "fulfilled")
      setSummary(summaryRes.value.data as Record<string, unknown>);
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async () => {
    setCreating(true);
    try {
      await financialAPI.createTransaction({
        amount: Number(amount),
        transaction_type: txType,
        description: description || undefined,
      });
      setCreateOpen(false);
      setAmount("");
      setDescription("");
      load();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setCreating(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading financial data…" />;

  return (
    <ScreenContainer>
      <View style={styles.headerRow}>
        <Text variant="headlineSmall" style={{ fontWeight: "700" }}>
          Financial
        </Text>
        <Button mode="contained" compact onPress={() => setCreateOpen(true)}>
          New
        </Button>
      </View>

      {error && (
        <HelperText type="error" visible>
          {error}
        </HelperText>
      )}

      <View style={styles.statsGrid}>
        <StatCard
          label="Total volume"
          value={formatCurrency(summary?.total_volume)}
        />
        <StatCard
          label="Transactions"
          value={String(summary?.total_transactions ?? transactions.length)}
        />
      </View>

      {transactions.length === 0 ? (
        <EmptyState
          title="No transactions yet"
          description="Record your first transaction to get started."
        />
      ) : (
        transactions.map((tx) => (
          <View
            key={tx.id}
            style={[
              styles.txRow,
              { borderBottomColor: theme.colors.outlineVariant },
            ]}
          >
            <View style={{ flex: 1 }}>
              <Text
                variant="bodyMedium"
                style={{ fontWeight: "600", textTransform: "capitalize" }}
              >
                {tx.transaction_type}
              </Text>
              <Text
                variant="bodySmall"
                style={{ color: theme.colors.onSurfaceVariant }}
              >
                {tx.description ||
                  (tx.created_at
                    ? new Date(tx.created_at).toLocaleDateString()
                    : "")}
              </Text>
            </View>
            <View style={{ alignItems: "flex-end" }}>
              <Text variant="bodyMedium" style={{ fontWeight: "700" }}>
                {formatCurrency(tx.amount)}
              </Text>
              <StatusChip status={tx.status} />
            </View>
          </View>
        ))
      )}

      <Portal>
        <Dialog visible={createOpen} onDismiss={() => setCreateOpen(false)}>
          <Dialog.Title>New transaction</Dialog.Title>
          <Dialog.Content style={{ gap: 12 }}>
            <Menu
              visible={typeMenuOpen}
              onDismiss={() => setTypeMenuOpen(false)}
              anchor={
                <Button
                  mode="outlined"
                  onPress={() => setTypeMenuOpen(true)}
                  style={{ justifyContent: "flex-start" }}
                >
                  {txType}
                </Button>
              }
            >
              {TRANSACTION_TYPES.map((type) => (
                <Menu.Item
                  key={type}
                  title={type}
                  onPress={() => {
                    setTxType(type);
                    setTypeMenuOpen(false);
                  }}
                />
              ))}
            </Menu>
            <TextInput
              label="Amount (USD)"
              value={amount}
              onChangeText={setAmount}
              keyboardType="decimal-pad"
              mode="outlined"
            />
            <TextInput
              label="Description"
              value={description}
              onChangeText={setDescription}
              mode="outlined"
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setCreateOpen(false)}>Cancel</Button>
            <Button
              onPress={handleCreate}
              loading={creating}
              disabled={creating || !amount}
            >
              Submit
            </Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </ScreenContainer>
  );
};

const styles = StyleSheet.create({
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  statsGrid: { flexDirection: "row", gap: 12, marginBottom: 20 },
  txRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
});

export default FinancialScreen;
