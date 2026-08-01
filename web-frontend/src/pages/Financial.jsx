import AccountBalanceIcon from "@mui/icons-material/AccountBalanceOutlined";
import AddIcon from "@mui/icons-material/Add";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  MenuItem,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import { financialAPI, getErrorMessage } from "../api";
import EmptyState from "../components/common/EmptyState";
import LoadingScreen from "../components/common/LoadingScreen";
import PageHeader from "../components/common/PageHeader";
import StatusChip from "../components/common/StatusChip";

const TRANSACTION_TYPES = [
  "deposit",
  "withdrawal",
  "transfer",
  "payment",
  "trade",
  "fee",
];

const emptyForm = { amount: "", transaction_type: "deposit", description: "" };

const formatCurrency = (value) => {
  if (typeof value !== "number") return value ?? "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(value);
};

const Financial = () => {
  const [transactions, setTransactions] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [txRes, summaryRes] = await Promise.allSettled([
        financialAPI.listTransactions({ limit: 25 }),
        financialAPI.summary(),
      ]);
      if (txRes.status === "fulfilled") {
        setTransactions(txRes.value.data?.items || txRes.value.data || []);
      } else {
        setError(getErrorMessage(txRes.reason));
      }
      if (summaryRes.status === "fulfilled") setSummary(summaryRes.value.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    setCreating(true);
    setCreateError(null);
    try {
      await financialAPI.createTransaction({
        amount: Number(form.amount),
        transaction_type: form.transaction_type,
        description: form.description || undefined,
      });
      setCreateOpen(false);
      setForm(emptyForm);
      loadData();
    } catch (err) {
      setCreateError(getErrorMessage(err));
    } finally {
      setCreating(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading financial data…" />;

  return (
    <Box>
      <PageHeader
        title="Financial"
        description="Track transactions, risk levels, and compliance status."
        action={
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateOpen(true)}
            disableElevation
          >
            New transaction
          </Button>
        }
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={2.5} sx={{ mb: 3 }}>
        {[
          {
            label: "Total volume",
            value: formatCurrency(summary?.total_volume),
          },
          {
            label: "Total transactions",
            value: summary?.total_transactions ?? transactions.length,
          },
          {
            label: "Pending approval",
            value: summary?.pending_approval ?? "-",
          },
          {
            label: "Flagged (high risk)",
            value: summary?.high_risk_count ?? "-",
          },
        ].map((item) => (
          <Grid item xs={6} md={3} key={item.label}>
            <Card>
              <CardContent>
                <Typography variant="body2" color="text.secondary">
                  {item.label}
                </Typography>
                <Typography variant="h5" fontWeight={700} sx={{ mt: 0.5 }}>
                  {item.value}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {transactions.length === 0 ? (
        <Paper sx={{ p: 2 }}>
          <EmptyState
            icon={<AccountBalanceIcon fontSize="inherit" />}
            title="No transactions yet"
            description="Record your first transaction to start tracking risk and compliance."
            actionLabel="New transaction"
            onAction={() => setCreateOpen(true)}
          />
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Type</TableCell>
                <TableCell>Amount</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Risk</TableCell>
                <TableCell>Date</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {transactions.map((tx) => (
                <TableRow key={tx.id} hover>
                  <TableCell sx={{ textTransform: "capitalize" }}>
                    {tx.transaction_type}
                  </TableCell>
                  <TableCell sx={{ fontFamily: "monospace" }}>
                    {formatCurrency(tx.amount)}
                  </TableCell>
                  <TableCell>{tx.description || "-"}</TableCell>
                  <TableCell>
                    <StatusChip status={tx.status} />
                  </TableCell>
                  <TableCell>
                    {tx.risk_level ? (
                      <StatusChip status={tx.risk_level} />
                    ) : (
                      "-"
                    )}
                  </TableCell>
                  <TableCell>
                    {tx.created_at
                      ? new Date(tx.created_at).toLocaleDateString()
                      : "-"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Dialog
        open={createOpen}
        onClose={() => !creating && setCreateOpen(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle sx={{ fontWeight: 700 }}>New transaction</DialogTitle>
        <Box component="form" onSubmit={handleCreate}>
          <DialogContent>
            <Stack spacing={2}>
              {createError && <Alert severity="error">{createError}</Alert>}
              <TextField
                select
                label="Type"
                required
                fullWidth
                value={form.transaction_type}
                onChange={(e) =>
                  setForm((f) => ({ ...f, transaction_type: e.target.value }))
                }
              >
                {TRANSACTION_TYPES.map((type) => (
                  <MenuItem
                    key={type}
                    value={type}
                    sx={{ textTransform: "capitalize" }}
                  >
                    {type}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                label="Amount (USD)"
                type="number"
                required
                fullWidth
                inputProps={{ min: 0.01, step: 0.01, max: 1000000 }}
                value={form.amount}
                onChange={(e) =>
                  setForm((f) => ({ ...f, amount: e.target.value }))
                }
              />
              <TextField
                label="Description"
                fullWidth
                multiline
                rows={2}
                value={form.description}
                onChange={(e) =>
                  setForm((f) => ({ ...f, description: e.target.value }))
                }
              />
            </Stack>
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 2 }}>
            <Button
              onClick={() => setCreateOpen(false)}
              disabled={creating}
              color="inherit"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={creating}
              disableElevation
            >
              {creating ? "Submitting…" : "Submit"}
            </Button>
          </DialogActions>
        </Box>
      </Dialog>
    </Box>
  );
};

export default Financial;
