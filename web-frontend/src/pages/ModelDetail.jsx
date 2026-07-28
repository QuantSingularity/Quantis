import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Grid,
  Stack,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import { Link as RouterLink, useParams } from "react-router-dom";
import { getErrorMessage, modelsAPI } from "../api";
import LoadingScreen from "../components/common/LoadingScreen";
import StatusChip from "../components/common/StatusChip";

const formatMetric = (value) => {
  if (typeof value !== "number") return value ?? "—";
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
};

const ModelDetail = () => {
  const { modelId } = useParams();
  const [model, setModel] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [training, setTraining] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelRes, metricsRes] = await Promise.allSettled([
        modelsAPI.get(modelId),
        modelsAPI.metrics(modelId),
      ]);
      if (modelRes.status === "fulfilled") setModel(modelRes.value.data);
      else setError(getErrorMessage(modelRes.reason));
      if (metricsRes.status === "fulfilled") setMetrics(metricsRes.value.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId]);

  const handleTrain = async () => {
    setTraining(true);
    try {
      await modelsAPI.train(modelId);
      await load();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setTraining(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading model…" />;
  if (error && !model) return <Alert severity="error">{error}</Alert>;

  const canTrain = ["created", "failed"].includes(model?.status) && !training;
  const metricEntries = Object.entries(metrics || model?.metrics || {}).filter(
    ([key]) => !["error"].includes(key),
  );

  return (
    <Box>
      <Button
        startIcon={<ArrowBackIcon />}
        component={RouterLink}
        to="/app/models"
        sx={{ mb: 2 }}
        color="inherit"
      >
        Back to models
      </Button>

      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="flex-start"
        sx={{ mb: 3 }}
        flexWrap="wrap"
        gap={2}
      >
        <Box>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <Typography variant="h4" fontWeight={700}>
              {model?.name}
            </Typography>
            <StatusChip status={model?.status} />
          </Stack>
          <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
            <Chip
              size="small"
              label={String(model?.model_type).replace(/_/g, " ")}
              sx={{ textTransform: "capitalize" }}
            />
            <Chip
              size="small"
              label={`v${model?.version}`}
              variant="outlined"
            />
          </Stack>
          {model?.description && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mt: 1.5, maxWidth: 560 }}
            >
              {model.description}
            </Typography>
          )}
        </Box>
        <Button
          variant="contained"
          startIcon={<PlayArrowIcon />}
          disabled={!canTrain}
          onClick={handleTrain}
          disableElevation
        >
          {training
            ? "Training…"
            : model?.status === "failed"
              ? "Retry training"
              : "Train model"}
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Typography variant="h6" sx={{ mb: 1.5 }}>
        Performance metrics
      </Typography>
      {metricEntries.length === 0 ? (
        <Alert severity="info" sx={{ mb: 3 }}>
          No metrics available yet. Train the model to generate performance
          metrics.
        </Alert>
      ) : (
        <Grid container spacing={2.5} sx={{ mb: 3 }}>
          {metricEntries.map(([key, value]) => (
            <Grid item xs={6} sm={4} md={3} key={key}>
              <Card>
                <CardContent>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ textTransform: "uppercase" }}
                  >
                    {key.replace(/_/g, " ")}
                  </Typography>
                  <Typography
                    variant="h5"
                    fontWeight={700}
                    sx={{ mt: 0.5, fontFamily: "monospace" }}
                  >
                    {formatMetric(value)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      <Grid container spacing={2.5}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1.5 }}>
                Configuration
              </Typography>
              <Stack spacing={1}>
                <Stack direction="row" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Dataset ID
                  </Typography>
                  <Typography variant="body2">{model?.dataset_id}</Typography>
                </Stack>
                <Stack direction="row" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Target column
                  </Typography>
                  <Typography variant="body2">
                    {model?.target_column || "—"}
                  </Typography>
                </Stack>
                <Stack direction="row" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Trained at
                  </Typography>
                  <Typography variant="body2">
                    {model?.trained_at
                      ? new Date(model.trained_at).toLocaleString()
                      : "Not trained yet"}
                  </Typography>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1.5 }}>
                Hyperparameters
              </Typography>
              {model?.hyperparameters &&
              Object.keys(model.hyperparameters).length > 0 ? (
                <Stack spacing={1}>
                  {Object.entries(model.hyperparameters).map(([key, value]) => (
                    <Stack
                      direction="row"
                      justifyContent="space-between"
                      key={key}
                    >
                      <Typography variant="body2" color="text.secondary">
                        {key}
                      </Typography>
                      <Typography variant="body2" fontFamily="monospace">
                        {String(value)}
                      </Typography>
                    </Stack>
                  ))}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Using default hyperparameters
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelDetail;
