import InsightsIcon from "@mui/icons-material/InsightsOutlined";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
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
import { getErrorMessage, modelsAPI, predictionsAPI } from "../api";
import EmptyState from "../components/common/EmptyState";
import LoadingScreen from "../components/common/LoadingScreen";
import PageHeader from "../components/common/PageHeader";

const Predictions = () => {
  const [models, setModels] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedModel, setSelectedModel] = useState("");
  const [inputJson, setInputJson] = useState(
    '{\n  "feature_1": 0.5,\n  "feature_2": 1.2\n}',
  );
  const [jsonError, setJsonError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  const trainedModels = models.filter(
    (m) => m.status === "trained" || m.status === "deployed",
  );

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelsRes, historyRes] = await Promise.all([
        modelsAPI.list(),
        predictionsAPI.history({ limit: 20 }),
      ]);
      const modelList = modelsRes.data?.items || modelsRes.data || [];
      setModels(modelList);
      setHistory(historyRes.data?.items || historyRes.data || []);
      if (modelList.length > 0) {
        const firstTrained = modelList.find(
          (m) => m.status === "trained" || m.status === "deployed",
        );
        if (firstTrained) setSelectedModel(String(firstTrained.id));
      }
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    setJsonError(null);
    setSubmitError(null);
    setLastResult(null);

    let parsedInput;
    try {
      parsedInput = JSON.parse(inputJson);
    } catch {
      setJsonError('Input must be valid JSON, e.g. {"feature_1": 0.5}');
      return;
    }

    setSubmitting(true);
    try {
      const { data } = await predictionsAPI.predict(
        Number(selectedModel),
        parsedInput,
      );
      setLastResult(data);
      loadData();
    } catch (err) {
      setSubmitError(getErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading predictions…" />;

  return (
    <Box>
      <PageHeader
        title="Predictions"
        description="Run predictions against your trained models and review history."
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={2.5} sx={{ mb: 4 }}>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
                New prediction
              </Typography>
              {trainedModels.length === 0 ? (
                <Alert severity="info">
                  No trained models available yet. Train a model first from the
                  Models page.
                </Alert>
              ) : (
                <Box component="form" onSubmit={handlePredict}>
                  <Stack spacing={2}>
                    {submitError && (
                      <Alert severity="error">{submitError}</Alert>
                    )}
                    <TextField
                      select
                      label="Model"
                      required
                      fullWidth
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                    >
                      {trainedModels.map((m) => (
                        <MenuItem key={m.id} value={m.id}>
                          {m.name} ({String(m.model_type).replace(/_/g, " ")})
                        </MenuItem>
                      ))}
                    </TextField>
                    <TextField
                      label="Input data (JSON)"
                      required
                      fullWidth
                      multiline
                      rows={6}
                      value={inputJson}
                      onChange={(e) => setInputJson(e.target.value)}
                      error={Boolean(jsonError)}
                      helperText={jsonError}
                      InputProps={{
                        sx: { fontFamily: "monospace", fontSize: 13 },
                      }}
                    />
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={<PlayArrowIcon />}
                      disabled={submitting}
                      disableElevation
                    >
                      {submitting ? "Running…" : "Run prediction"}
                    </Button>
                  </Stack>
                </Box>
              )}

              {lastResult && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Result
                  </Typography>
                  <Paper
                    variant="outlined"
                    sx={{ p: 2, bgcolor: "action.hover" }}
                  >
                    <Typography
                      component="pre"
                      variant="body2"
                      sx={{
                        fontFamily: "monospace",
                        whiteSpace: "pre-wrap",
                        m: 0,
                      }}
                    >
                      {JSON.stringify(
                        lastResult.prediction_result ?? lastResult,
                        null,
                        2,
                      )}
                    </Typography>
                    {typeof lastResult.confidence_score === "number" && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ display: "block", mt: 1 }}
                      >
                        Confidence:{" "}
                        {(lastResult.confidence_score * 100).toFixed(1)}%
                      </Typography>
                    )}
                  </Paper>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={7}>
          <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1.5 }}>
            Prediction history
          </Typography>
          {history.length === 0 ? (
            <Paper sx={{ p: 2 }}>
              <EmptyState
                icon={<InsightsIcon fontSize="inherit" />}
                title="No predictions yet"
                description="Run a prediction to see it appear here."
              />
            </Paper>
          ) : (
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Model</TableCell>
                    <TableCell>Result</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Date</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {history.map((p) => (
                    <TableRow key={p.id} hover>
                      <TableCell>#{p.model_id}</TableCell>
                      <TableCell
                        sx={{
                          fontFamily: "monospace",
                          fontSize: 12,
                          maxWidth: 220,
                        }}
                      >
                        {typeof p.prediction_result === "object"
                          ? JSON.stringify(p.prediction_result).slice(0, 60)
                          : String(p.prediction_result ?? "—")}
                      </TableCell>
                      <TableCell>
                        {typeof p.confidence_score === "number"
                          ? `${(p.confidence_score * 100).toFixed(0)}%`
                          : "—"}
                      </TableCell>
                      <TableCell>
                        {p.created_at
                          ? new Date(p.created_at).toLocaleDateString()
                          : "—"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Predictions;
