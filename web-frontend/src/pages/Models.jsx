import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ModelTrainingIcon from "@mui/icons-material/ModelTrainingOutlined";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
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
  Tooltip,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import { datasetsAPI, getErrorMessage, modelsAPI } from "../api";
import ConfirmDialog from "../components/common/ConfirmDialog";
import EmptyState from "../components/common/EmptyState";
import LoadingScreen from "../components/common/LoadingScreen";
import PageHeader from "../components/common/PageHeader";
import StatusChip from "../components/common/StatusChip";

const MODEL_TYPES = [
  { value: "tft", label: "Temporal Fusion Transformer" },
  { value: "lstm", label: "LSTM" },
  { value: "arima", label: "ARIMA" },
  { value: "prophet", label: "Prophet" },
  { value: "linear_regression", label: "Linear Regression" },
  { value: "random_forest", label: "Random Forest" },
  { value: "xgboost", label: "XGBoost" },
  { value: "ensemble", label: "Ensemble" },
];

const emptyForm = {
  name: "",
  description: "",
  model_type: "random_forest",
  dataset_id: "",
};

const Models = () => {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState(null);

  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [trainingIds, setTrainingIds] = useState(new Set());

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelsRes, datasetsRes] = await Promise.all([
        modelsAPI.list(),
        datasetsAPI.list(),
      ]);
      setModels(modelsRes.data?.items || modelsRes.data || []);
      setDatasets(datasetsRes.data?.items || datasetsRes.data || []);
    } catch (err) {
      setError(getErrorMessage(err));
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
      await modelsAPI.create({
        name: form.name,
        description: form.description || undefined,
        model_type: form.model_type,
        dataset_id: Number(form.dataset_id),
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

  const handleTrain = async (model) => {
    setTrainingIds((prev) => new Set(prev).add(model.id));
    try {
      await modelsAPI.train(model.id);
      await loadData();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setTrainingIds((prev) => {
        const next = new Set(prev);
        next.delete(model.id);
        return next;
      });
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await modelsAPI.remove(deleteTarget.id);
      setModels((prev) => prev.filter((m) => m.id !== deleteTarget.id));
      setDeleteTarget(null);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setDeleting(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading models…" />;

  return (
    <Box>
      <PageHeader
        title="Models"
        description="Train and manage forecasting models across your datasets."
        action={
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateOpen(true)}
            disableElevation
            disabled={datasets.length === 0}
          >
            Create model
          </Button>
        }
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {datasets.length === 0 && (
        <Alert severity="info" sx={{ mb: 2 }}>
          You need at least one dataset before creating a model.{" "}
          <RouterLink to="/app/datasets">Upload one now</RouterLink>.
        </Alert>
      )}

      {models.length === 0 ? (
        <Paper sx={{ p: 2 }}>
          <EmptyState
            icon={<ModelTrainingIcon fontSize="inherit" />}
            title="No models yet"
            description="Create a model from one of your datasets to start forecasting."
            actionLabel={
              datasets.length > 0 ? "Create your first model" : undefined
            }
            onAction={
              datasets.length > 0 ? () => setCreateOpen(true) : undefined
            }
          />
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Version</TableCell>
                <TableCell>Created</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.map((model) => {
                const isTraining =
                  trainingIds.has(model.id) || model.status === "training";
                const canTrain =
                  ["created", "failed"].includes(model.status) && !isTraining;
                return (
                  <TableRow key={model.id} hover>
                    <TableCell>
                      <Typography
                        component={RouterLink}
                        to={`/app/models/${model.id}`}
                        variant="body2"
                        fontWeight={600}
                        sx={{
                          color: "text.primary",
                          textDecoration: "none",
                          "&:hover": { color: "primary.main" },
                        }}
                      >
                        {model.name}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ textTransform: "capitalize" }}>
                      {String(model.model_type).replace(/_/g, " ")}
                    </TableCell>
                    <TableCell>
                      <StatusChip
                        status={isTraining ? "training" : model.status}
                      />
                    </TableCell>
                    <TableCell>{model.version}</TableCell>
                    <TableCell>
                      {model.created_at
                        ? new Date(model.created_at).toLocaleDateString()
                        : "—"}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip
                        title={
                          canTrain
                            ? "Train model"
                            : "Model must be in Created or Failed state"
                        }
                      >
                        <span>
                          <IconButton
                            size="small"
                            disabled={!canTrain}
                            onClick={() => handleTrain(model)}
                          >
                            <PlayArrowIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Delete model">
                        <IconButton
                          size="small"
                          onClick={() => setDeleteTarget(model)}
                        >
                          <DeleteOutlineIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Dialog
        open={createOpen}
        onClose={() => !creating && setCreateOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ fontWeight: 700 }}>Create model</DialogTitle>
        <Box component="form" onSubmit={handleCreate}>
          <DialogContent>
            <Stack spacing={2}>
              {createError && <Alert severity="error">{createError}</Alert>}
              <TextField
                label="Name"
                required
                fullWidth
                value={form.name}
                onChange={(e) =>
                  setForm((f) => ({ ...f, name: e.target.value }))
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
              <TextField
                select
                label="Model type"
                required
                fullWidth
                value={form.model_type}
                onChange={(e) =>
                  setForm((f) => ({ ...f, model_type: e.target.value }))
                }
              >
                {MODEL_TYPES.map((type) => (
                  <MenuItem key={type.value} value={type.value}>
                    {type.label}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                select
                label="Dataset"
                required
                fullWidth
                value={form.dataset_id}
                onChange={(e) =>
                  setForm((f) => ({ ...f, dataset_id: e.target.value }))
                }
              >
                {datasets.map((dataset) => (
                  <MenuItem key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </MenuItem>
                ))}
              </TextField>
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
              {creating ? "Creating…" : "Create model"}
            </Button>
          </DialogActions>
        </Box>
      </Dialog>

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Delete model?"
        description={`"${deleteTarget?.name}" will be permanently removed. This cannot be undone.`}
        confirmLabel="Delete"
        destructive
        loading={deleting}
        onConfirm={handleDelete}
        onClose={() => setDeleteTarget(null)}
      />
    </Box>
  );
};

export default Models;
