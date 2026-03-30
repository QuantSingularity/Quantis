import {
  CloudUpload,
  Delete,
  InsertChart,
  Search,
  Storage as StorageIcon,
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
  Grid,
  IconButton,
  InputAdornment,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
  useTheme,
} from "@mui/material";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { datasetsAPI, handleApiError } from "../services/api";

const Datasets = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [previewDialog, setPreviewDialog] = useState({
    open: false,
    dataset: null,
    preview: null,
  });

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await datasetsAPI.getDatasets();
      setDatasets(response.data || []);
    } catch (error) {
      const apiError = handleApiError(error);
      setError(apiError.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDataset = async (datasetId) => {
    if (window.confirm("Are you sure you want to delete this dataset?")) {
      try {
        await datasetsAPI.deleteDataset(datasetId);
        setDatasets((prev) =>
          prev.filter((dataset) => dataset.id !== datasetId),
        );
      } catch (error) {
        const apiError = handleApiError(error);
        setError(apiError.message);
      }
    }
  };

  const handleViewPreview = async (dataset) => {
    try {
      const preview = await datasetsAPI.getDatasetPreview(dataset.id, 10);
      setPreviewDialog({
        open: true,
        dataset,
        preview: preview.data,
      });
    } catch (error) {
      const apiError = handleApiError(error);
      setError(apiError.message);
    }
  };

  const handleClosePreview = () => {
    setPreviewDialog({
      open: false,
      dataset: null,
      preview: null,
    });
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return "N/A";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`;
  };

  const filteredDatasets = datasets.filter(
    (dataset) =>
      dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      dataset.description?.toLowerCase().includes(searchTerm.toLowerCase()),
  );

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
            Datasets
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage your uploaded datasets
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<CloudUpload />}
          onClick={() => navigate("/dataset-upload")}
        >
          Upload Dataset
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <TextField
            fullWidth
            placeholder="Search datasets..."
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
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {filteredDatasets.map((dataset) => (
          <Grid item xs={12} md={6} lg={4} key={dataset.id}>
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
                      {dataset.name}
                    </Typography>
                    <Chip
                      label={dataset.format || "CSV"}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </Box>
                  <StorageIcon
                    sx={{ color: theme.palette.secondary.main, ml: 1 }}
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
                  {dataset.description || "No description available"}
                </Typography>

                <Paper
                  sx={{
                    p: 1.5,
                    mb: 2,
                    backgroundColor: theme.palette.background.default,
                  }}
                >
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">
                        Rows
                      </Typography>
                      <Typography variant="body2" fontWeight="500">
                        {dataset.row_count?.toLocaleString() || "N/A"}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">
                        Columns
                      </Typography>
                      <Typography variant="body2" fontWeight="500">
                        {dataset.columns?.length ||
                          dataset.column_count ||
                          "N/A"}
                      </Typography>
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="caption" color="text.secondary">
                        Size
                      </Typography>
                      <Typography variant="body2" fontWeight="500">
                        {formatFileSize(dataset.file_size)}
                      </Typography>
                    </Grid>
                  </Grid>
                </Paper>

                <Typography
                  variant="caption"
                  color="text.secondary"
                  display="block"
                  sx={{ mb: 2 }}
                >
                  Uploaded: {new Date(dataset.created_at).toLocaleDateString()}
                </Typography>

                <Box
                  sx={{ display: "flex", gap: 1, justifyContent: "flex-end" }}
                >
                  <Tooltip title="View Preview">
                    <IconButton
                      size="small"
                      onClick={() => handleViewPreview(dataset)}
                    >
                      <Visibility />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Delete Dataset">
                    <IconButton
                      size="small"
                      onClick={() => handleDeleteDataset(dataset.id)}
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

        {filteredDatasets.length === 0 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 6, textAlign: "center" }}>
              <InsertChart
                sx={{
                  fontSize: 80,
                  color: theme.palette.text.secondary,
                  mb: 2,
                }}
              />
              <Typography variant="h5" gutterBottom>
                {searchTerm ? "No Datasets Found" : "No Datasets Yet"}
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                {searchTerm
                  ? "Try adjusting your search criteria"
                  : "Upload your first dataset to get started with ML models"}
              </Typography>
              {!searchTerm && (
                <Button
                  variant="contained"
                  startIcon={<CloudUpload />}
                  onClick={() => navigate("/dataset-upload")}
                >
                  Upload Dataset
                </Button>
              )}
            </Paper>
          </Grid>
        )}
      </Grid>

      <Dialog
        open={previewDialog.open}
        onClose={handleClosePreview}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Dataset Preview: {previewDialog.dataset?.name}
        </DialogTitle>
        <DialogContent>
          {previewDialog.preview && (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {previewDialog.preview.columns.map((column) => (
                      <TableCell key={column}>
                        <Typography variant="subtitle2" fontWeight="600">
                          {column}
                        </Typography>
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {previewDialog.preview.data.map((row, index) => (
                    <TableRow key={index}>
                      {previewDialog.preview.columns.map((column) => (
                        <TableCell key={column}>
                          {row[column]?.toString() || ""}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClosePreview}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Datasets;
