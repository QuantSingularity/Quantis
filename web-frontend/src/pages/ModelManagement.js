import {
  Add,
  DataUsage,
  Delete,
  PlayArrow,
  Visibility,
} from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  Grid,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  TextField,
  Tooltip,
  Typography,
  useTheme,
} from "@mui/material";
import { useEffect, useState } from "react";
import BarChart from "../components/charts/BarChart";
import { datasetsAPI, handleApiError, modelsAPI } from "../services/api";

const ModelManagement = () => {
  const theme = useTheme();
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelDetailsOpen, setModelDetailsOpen] = useState(false);
  const [trainingModels, setTrainingModels] = useState(new Set());

  const [newModel, setNewModel] = useState({
    name: "",
    description: "",
    model_type: "linear_regression",
    dataset_id: "",
    target_column: "",
    feature_columns: [],
    hyperparameters: {},
  });

  useEffect(() => {
    loadData();
  }, [loadData]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [modelsResponse, datasetsResponse] = await Promise.all([
        modelsAPI.getModels(),
        datasetsAPI.getDatasets(),
      ]);
      setModels(modelsResponse.data || []);
      setDatasets(datasetsResponse.data || []);
    } catch (error) {
      const apiError = handleApiError(error);
      setError(apiError.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateModel = async () => {
    try {
      const response = await modelsAPI.createModel(newModel);
      setModels((prev) => [...prev, response.data]);
      setCreateDialogOpen(false);
      setNewModel({
        name: "",
        description: "",
        model_type: "linear_regression",
        dataset_id: "",
        target_column: "",
        feature_columns: [],
        hyperparameters: {},
      });
    } catch (error) {
      const apiError = handleApiError(error);
      setError(apiError.message);
    }
  };

  const handleTrainModel = async (modelId) => {
    try {
      setTrainingModels((prev) => new Set([...prev, modelId]));
      await modelsAPI.trainModel(modelId);
      // Refresh models to get updated status
      await loadData();
    } catch (error) {
      const apiError = handleApiError(error);
      setError(apiError.message);
    } finally {
      setTrainingModels((prev) => {
        const newSet = new Set(prev);
        newSet.delete(modelId);
        return newSet;
      });
    }
  };

  const handleDeleteModel = async (modelId) => {
    if (window.confirm("Are you sure you want to delete this model?")) {
      try {
        await modelsAPI.deleteModel(modelId);
        setModels((prev) => prev.filter((model) => model.id !== modelId));
      } catch (error) {
        const apiError = handleApiError(error);
        setError(apiError.message);
      }
    }
  };

  const handleViewModelDetails = async (model) => {
    try {
      const [modelDetails, metrics] = await Promise.all([
        modelsAPI.getModel(model.id),
        modelsAPI.getModelMetrics(model.id).catch(() => ({ data: null })),
      ]);
      setSelectedModel({
        ...modelDetails.data,
        metrics: metrics.data,
      });
      setModelDetailsOpen(true);
    } catch (error) {
      const apiError = handleApiError(error);
      setError(apiError.message);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "trained":
        return "success";
      case "training":
        return "warning";
      case "failed":
        return "error";
      default:
        return "default";
    }
  };

  const modelTypes = [
    { value: "linear_regression", label: "Linear Regression" },
    { value: "random_forest", label: "Random Forest" },
    { value: "lstm", label: "LSTM Neural Network" },
    { value: "arima", label: "ARIMA" },
    { value: "tft", label: "Temporal Fusion Transformer" },
  ];

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography variant="h4">Model Management</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Model
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} md={6} lg={4} key={model.id}>
            <Card>
              <CardContent>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    mb: 2,
                  }}
                >
                  <Typography variant="h6" noWrap>
                    {model.name}
                  </Typography>
                  <Chip
                    label={model.status}
                    color={getStatusColor(model.status)}
                    size="small"
                  />
                </Box>

                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mb: 2 }}
                >
                  {model.description || "No description"}
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    Type: {model.model_type}
                  </Typography>
                  <br />
                  <Typography variant="caption" color="text.secondary">
                    Created: {new Date(model.created_at).toLocaleDateString()}
                  </Typography>
                </Box>

                {model.status === "training" && (
                  <LinearProgress sx={{ mb: 2 }} />
                )}

                {model.metrics && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Accuracy: {(model.metrics.accuracy * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                )}

                <Box sx={{ display: "flex", gap: 1 }}>
                  <Tooltip title="View Details">
                    <IconButton
                      size="small"
                      onClick={() => handleViewModelDetails(model)}
                    >
                      <Visibility />
                    </IconButton>
                  </Tooltip>

                  {model.status === "created" && (
                    <Tooltip title="Train Model">
                      <IconButton
                        size="small"
                        onClick={() => handleTrainModel(model.id)}
                        disabled={trainingModels.has(model.id)}
                      >
                        {trainingModels.has(model.id) ? (
                          <CircularProgress size={20} />
                        ) : (
                          <PlayArrow />
                        )}
                      </IconButton>
                    </Tooltip>
                  )}

                  <Tooltip title="Delete Model">
                    <IconButton
                      size="small"
                      onClick={() => handleDeleteModel(model.id)}
                      color="error"
                    >
                      <Delete />
                    </IconButton>
                  </Tooltip>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}

        {models.length === 0 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 4, textAlign: "center" }}>
              <DataUsage
                sx={{
                  fontSize: 64,
                  color: theme.palette.text.secondary,
                  mb: 2,
                }}
              />
              <Typography variant="h6" gutterBottom>
                No Models Yet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Create your first machine learning model to get started.
              </Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setCreateDialogOpen(true)}
              >
                Create Model
              </Button>
            </Paper>
          </Grid>
        )}
      </Grid>

      {/* Create Model Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Model</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Model Name"
                value={newModel.name}
                onChange={(e) =>
                  setNewModel((prev) => ({ ...prev, name: e.target.value }))
                }
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Model Type</InputLabel>
                <Select
                  value={newModel.model_type}
                  label="Model Type"
                  onChange={(e) =>
                    setNewModel((prev) => ({
                      ...prev,
                      model_type: e.target.value,
                    }))
                  }
                >
                  {modelTypes.map((type) => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                multiline
                rows={3}
                value={newModel.description}
                onChange={(e) =>
                  setNewModel((prev) => ({
                    ...prev,
                    description: e.target.value,
                  }))
                }
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Dataset</InputLabel>
                <Select
                  value={newModel.dataset_id}
                  label="Dataset"
                  onChange={(e) =>
                    setNewModel((prev) => ({
                      ...prev,
                      dataset_id: e.target.value,
                    }))
                  }
                  required
                >
                  {datasets.map((dataset) => (
                    <MenuItem key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.row_count} rows)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateModel}
            variant="contained"
            disabled={!newModel.name || !newModel.dataset_id}
          >
            Create Model
          </Button>
        </DialogActions>
      </Dialog>

      {/* Model Details Dialog */}
      <Dialog
        open={modelDetailsOpen}
        onClose={() => setModelDetailsOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>Model Details: {selectedModel?.name}</DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Model Information
                  </Typography>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Type
                    </Typography>
                    <Typography variant="body1">
                      {selectedModel.model_type}
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Status
                    </Typography>
                    <Chip
                      label={selectedModel.status}
                      color={getStatusColor(selectedModel.status)}
                      size="small"
                    />
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Created
                    </Typography>
                    <Typography variant="body1">
                      {new Date(selectedModel.created_at).toLocaleString()}
                    </Typography>
                  </Box>
                  {selectedModel.trained_at && (
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Trained
                      </Typography>
                      <Typography variant="body1">
                        {new Date(selectedModel.trained_at).toLocaleString()}
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Grid>

              {selectedModel.metrics && (
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Performance Metrics
                    </Typography>
                    {Object.entries(selectedModel.metrics).map(
                      ([key, value]) => (
                        <Box key={key} sx={{ mb: 1 }}>
                          <Typography variant="body2" color="text.secondary">
                            {key.replace(/_/g, " ").toUpperCase()}
                          </Typography>
                          <Typography variant="body1">
                            {typeof value === "number"
                              ? value.toFixed(4)
                              : value}
                          </Typography>
                        </Box>
                      ),
                    )}
                  </Paper>
                </Grid>
              )}

              {selectedModel.feature_importance && (
                <Grid item xs={12}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Feature Importance
                    </Typography>
                    <BarChart
                      data={selectedModel.feature_importance}
                      height={300}
                      dataKey="importance"
                      nameKey="feature"
                    />
                  </Paper>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setModelDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ModelManagement;
