import AddIcon from "@mui/icons-material/Add";
import AccountBalanceIcon from "@mui/icons-material/AccountBalanceOutlined";
import DatasetIcon from "@mui/icons-material/StorageOutlined";
import ModelTrainingIcon from "@mui/icons-material/ModelTrainingOutlined";
import PredictionIcon from "@mui/icons-material/InsightsOutlined";
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Grid,
  List,
  ListItem,
  ListItemText,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Link as RouterLink } from "react-router-dom";
import { datasetsAPI, financialAPI, modelsAPI, predictionsAPI } from "../api";
import EmptyState from "../components/common/EmptyState";
import LoadingScreen from "../components/common/LoadingScreen";
import PageHeader from "../components/common/PageHeader";
import StatCard from "../components/common/StatCard";
import StatusChip from "../components/common/StatusChip";
import { useAuth } from "../context/AuthContext";

const Dashboard = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [predictionStats, setPredictionStats] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [financialSummary, setFinancialSummary] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const loadAll = async () => {
      const results = await Promise.allSettled([
        datasetsAPI.list({ limit: 5 }),
        modelsAPI.list({ limit: 5 }),
        predictionsAPI.stats(),
        predictionsAPI.history({ limit: 5 }),
        financialAPI.summary(),
      ]);
      if (cancelled) return;

      const [datasetsRes, modelsRes, statsRes, historyRes, financialRes] =
        results;
      if (datasetsRes.status === "fulfilled") {
        setDatasets(
          datasetsRes.value.data?.items || datasetsRes.value.data || [],
        );
      }
      if (modelsRes.status === "fulfilled") {
        setModels(modelsRes.value.data?.items || modelsRes.value.data || []);
      }
      if (statsRes.status === "fulfilled")
        setPredictionStats(statsRes.value.data);
      if (historyRes.status === "fulfilled") {
        setRecentPredictions(
          historyRes.value.data?.items || historyRes.value.data || [],
        );
      }
      if (financialRes.status === "fulfilled")
        setFinancialSummary(financialRes.value.data);

      setLoading(false);
    };

    loadAll();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <LoadingScreen label="Loading your workspace…" />;

  const trainedModels = models.filter(
    (m) => m.status === "trained" || m.status === "deployed",
  );
  const readyDatasets = datasets.filter((d) => d.status === "ready");

  const chartData = (
    predictionStats?.timeline ||
    predictionStats?.daily_counts ||
    []
  ).map((point, idx) => ({
    name: point.date || point.label || `Day ${idx + 1}`,
    predictions: point.count ?? point.value ?? 0,
  }));

  return (
    <Box>
      <PageHeader
        title={`Welcome back, ${user?.first_name || user?.username}`}
        description="Here's what's happening across your workspace."
        action={
          <Button
            component={RouterLink}
            to="/app/predictions"
            variant="contained"
            startIcon={<AddIcon />}
            disableElevation
          >
            New prediction
          </Button>
        }
      />

      <Grid container spacing={2.5} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            label="Datasets"
            value={datasets.length}
            icon={<DatasetIcon fontSize="small" />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            label="Trained models"
            value={trainedModels.length}
            icon={<ModelTrainingIcon fontSize="small" />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            label="Predictions made"
            value={
              predictionStats?.total_predictions ??
              predictionStats?.total ??
              recentPredictions.length
            }
            icon={<PredictionIcon fontSize="small" />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            label="Transaction volume"
            value={financialSummary?.total_volume ?? "$0"}
            icon={<AccountBalanceIcon fontSize="small" />}
          />
        </Grid>
      </Grid>

      <Grid container spacing={2.5}>
        <Grid item xs={12} md={7}>
          <Card sx={{ height: "100%" }}>
            <CardHeader
              title="Prediction activity"
              subheader="Volume over recent period"
            />
            <CardContent sx={{ pt: 0 }}>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="name" fontSize={12} tickLine={false} />
                    <YAxis
                      fontSize={12}
                      tickLine={false}
                      allowDecimals={false}
                    />
                    <RechartsTooltip />
                    <Line
                      type="monotone"
                      dataKey="predictions"
                      stroke="#6366F1"
                      strokeWidth={2.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState
                  icon={<PredictionIcon fontSize="inherit" />}
                  title="No predictions yet"
                  description="Run your first prediction to see activity here."
                  actionLabel="Make a prediction"
                  onAction={() => (window.location.href = "/app/predictions")}
                />
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          <Card sx={{ height: "100%" }}>
            <CardHeader
              title="Recent predictions"
              action={
                <Button
                  component={RouterLink}
                  to="/app/predictions"
                  size="small"
                >
                  View all
                </Button>
              }
            />
            <CardContent sx={{ pt: 0 }}>
              {recentPredictions.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No predictions yet.
                </Typography>
              ) : (
                <List disablePadding>
                  {recentPredictions.slice(0, 5).map((p) => (
                    <ListItem
                      key={p.id}
                      disableGutters
                      secondaryAction={
                        <StatusChip status={p.status || "completed"} />
                      }
                    >
                      <ListItemText
                        primary={`Model #${p.model_id}`}
                        secondary={
                          p.created_at
                            ? new Date(p.created_at).toLocaleString()
                            : "-"
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Datasets"
              action={
                <Button component={RouterLink} to="/app/datasets" size="small">
                  Manage
                </Button>
              }
            />
            <CardContent sx={{ pt: 0 }}>
              {readyDatasets.length === 0 && datasets.length === 0 ? (
                <EmptyState
                  icon={<DatasetIcon fontSize="inherit" />}
                  title="No datasets yet"
                  description="Upload a dataset to start training models."
                  actionLabel="Upload dataset"
                  onAction={() => (window.location.href = "/app/datasets")}
                />
              ) : (
                <List disablePadding>
                  {datasets.slice(0, 5).map((d) => (
                    <ListItem
                      key={d.id}
                      disableGutters
                      secondaryAction={<StatusChip status={d.status} />}
                    >
                      <ListItemText
                        primary={d.name}
                        secondary={`${d.row_count ?? "-"} rows`}
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Models"
              action={
                <Button component={RouterLink} to="/app/models" size="small">
                  Manage
                </Button>
              }
            />
            <CardContent sx={{ pt: 0 }}>
              {models.length === 0 ? (
                <EmptyState
                  icon={<ModelTrainingIcon fontSize="inherit" />}
                  title="No models yet"
                  description="Create a model from one of your datasets."
                  actionLabel="Create model"
                  onAction={() => (window.location.href = "/app/models")}
                />
              ) : (
                <List disablePadding>
                  {models.slice(0, 5).map((m) => (
                    <ListItem
                      key={m.id}
                      disableGutters
                      secondaryAction={<StatusChip status={m.status} />}
                    >
                      <ListItemText primary={m.name} secondary={m.model_type} />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
