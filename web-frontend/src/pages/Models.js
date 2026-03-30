import {
  Assessment,
  DataUsage,
  Delete,
  Search,
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
  Grid,
  IconButton,
  InputAdornment,
  LinearProgress,
  Paper,
  TextField,
  Tooltip,
  Typography,
  useTheme,
} from "@mui/material";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { handleApiError, modelsAPI } from "../services/api";

const Models = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const loadModels = async () => {
    try {
      setLoading(true);
      const response = await modelsAPI.getModels();
      setModels(response.data || []);
    } catch (error) {
      const apiError = handleApiError(error);
      setError(apiError.message);
    } finally {
      setLoading(false);
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

  const handleViewDetails = (_modelId) => {
    navigate(`/model-management`);
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

  const filteredModels = models.filter((model) => {
    const matchesSearch =
      model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      model.description?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter =
      filterStatus === "all" || model.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box className="fade-in">
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Box>
          <Typography variant="h4" gutterBottom fontWeight="600">
            Models
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Browse and manage your machine learning models
          </Typography>
        </Box>
        <Button
          variant="contained"
          onClick={() => navigate("/model-management")}
        >
          Create Model
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                placeholder="Search models..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                <Chip
                  label="All"
                  onClick={() => setFilterStatus("all")}
                  color={filterStatus === "all" ? "primary" : "default"}
                  variant={filterStatus === "all" ? "filled" : "outlined"}
                />
                <Chip
                  label="Trained"
                  onClick={() => setFilterStatus("trained")}
                  color={filterStatus === "trained" ? "success" : "default"}
                  variant={filterStatus === "trained" ? "filled" : "outlined"}
                />
                <Chip
                  label="Training"
                  onClick={() => setFilterStatus("training")}
                  color={filterStatus === "training" ? "warning" : "default"}
                  variant={filterStatus === "training" ? "filled" : "outlined"}
                />
                <Chip
                  label="Created"
                  onClick={() => setFilterStatus("created")}
                  color={filterStatus === "created" ? "info" : "default"}
                  variant={filterStatus === "created" ? "filled" : "outlined"}
                />
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {filteredModels.map((model) => (
          <Grid item xs={12} sm={6} md={4} key={model.id}>
            <Card
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                transition: "transform 0.2s, box-shadow 0.2s",
                "&:hover": {
                  transform: "translateY(-4px)",
                  boxShadow: theme.shadows[8],
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    mb: 2,
                  }}
                >
                  <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                    <Typography
                      variant="h6"
                      noWrap
                      sx={{ fontWeight: 600, mb: 0.5 }}
                    >
                      {model.name}
                    </Typography>
                    <Chip
                      label={model.status}
                      color={getStatusColor(model.status)}
                      size="small"
                    />
                  </Box>
                  <Assessment
                    sx={{ color: theme.palette.primary.main, ml: 1 }}
                  />
                </Box>

                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{
                    mb: 2,
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {model.description || "No description available"}
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    display="block"
                  >
                    Type: {model.model_type?.replace(/_/g, " ").toUpperCase()}
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    display="block"
                  >
                    Created: {new Date(model.created_at).toLocaleDateString()}
                  </Typography>
                  {model.trained_at && (
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      display="block"
                    >
                      Trained: {new Date(model.trained_at).toLocaleDateString()}
                    </Typography>
                  )}
                </Box>

                {model.status === "training" && (
                  <LinearProgress sx={{ mb: 2 }} />
                )}

                {model.metrics && model.status === "trained" && (
                  <Paper
                    sx={{
                      p: 1.5,
                      mb: 2,
                      backgroundColor: theme.palette.background.default,
                    }}
                  >
                    <Typography variant="caption" color="text.secondary">
                      Performance Metrics
                    </Typography>
                    {Object.entries(model.metrics)
                      .slice(0, 2)
                      .map(([key, value]) => (
                        <Typography
                          key={key}
                          variant="body2"
                          sx={{ fontWeight: 500 }}
                        >
                          {key.replace(/_/g, " ")}:{" "}
                          {typeof value === "number" ? value.toFixed(4) : value}
                        </Typography>
                      ))}
                  </Paper>
                )}

                <Box
                  sx={{ display: "flex", gap: 1, justifyContent: "flex-end" }}
                >
                  <Tooltip title="View Details">
                    <IconButton
                      size="small"
                      onClick={() => handleViewDetails(model.id)}
                    >
                      <Visibility />
                    </IconButton>
                  </Tooltip>
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

        {filteredModels.length === 0 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 6, textAlign: "center" }}>
              <DataUsage
                sx={{
                  fontSize: 80,
                  color: theme.palette.text.secondary,
                  mb: 2,
                }}
              />
              <Typography variant="h5" gutterBottom>
                {searchTerm || filterStatus !== "all"
                  ? "No Models Found"
                  : "No Models Yet"}
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                {searchTerm || filterStatus !== "all"
                  ? "Try adjusting your search or filter criteria"
                  : "Create your first machine learning model to get started"}
              </Typography>
              {!searchTerm && filterStatus === "all" && (
                <Button
                  variant="contained"
                  onClick={() => navigate("/model-management")}
                >
                  Create Model
                </Button>
              )}
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Models;
