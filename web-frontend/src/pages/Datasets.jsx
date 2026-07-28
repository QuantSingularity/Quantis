import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import StorageIcon from "@mui/icons-material/StorageOutlined";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  LinearProgress,
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
import { datasetsAPI, getErrorMessage } from "../api";
import ConfirmDialog from "../components/common/ConfirmDialog";
import EmptyState from "../components/common/EmptyState";
import LoadingScreen from "../components/common/LoadingScreen";
import PageHeader from "../components/common/PageHeader";
import StatusChip from "../components/common/StatusChip";

const FREQUENCIES = [
  "daily",
  "weekly",
  "monthly",
  "quarterly",
  "yearly",
  "irregular",
];

const emptyUploadForm = {
  name: "",
  description: "",
  frequency: "daily",
  file: null,
};

const Datasets = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadForm, setUploadForm] = useState(emptyUploadForm);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const loadDatasets = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await datasetsAPI.list();
      setDatasets(data?.items || data || []);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!uploadForm.file) {
      setUploadError("Please choose a file to upload");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      await datasetsAPI.upload(uploadForm.file, uploadForm, (progressEvent) => {
        const percent = Math.round(
          (progressEvent.loaded * 100) / (progressEvent.total || 1),
        );
        setUploadProgress(percent);
      });
      setUploadOpen(false);
      setUploadForm(emptyUploadForm);
      setUploadProgress(0);
      loadDatasets();
    } catch (err) {
      setUploadError(getErrorMessage(err));
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await datasetsAPI.remove(deleteTarget.id);
      setDatasets((prev) => prev.filter((d) => d.id !== deleteTarget.id));
      setDeleteTarget(null);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setDeleting(false);
    }
  };

  if (loading) return <LoadingScreen label="Loading datasets…" />;

  return (
    <Box>
      <PageHeader
        title="Datasets"
        description="Upload and manage the time-series data behind your models."
        action={
          <Button
            variant="contained"
            startIcon={<UploadFileIcon />}
            onClick={() => setUploadOpen(true)}
            disableElevation
          >
            Upload dataset
          </Button>
        }
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {datasets.length === 0 ? (
        <Paper sx={{ p: 2 }}>
          <EmptyState
            icon={<StorageIcon fontSize="inherit" />}
            title="No datasets yet"
            description="Upload a CSV or Excel file to get started with training models."
            actionLabel="Upload your first dataset"
            onAction={() => setUploadOpen(true)}
          />
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Rows</TableCell>
                <TableCell>Frequency</TableCell>
                <TableCell>Uploaded</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {datasets.map((dataset) => (
                <TableRow key={dataset.id} hover>
                  <TableCell>
                    <Typography
                      component={RouterLink}
                      to={`/app/datasets/${dataset.id}`}
                      variant="body2"
                      fontWeight={600}
                      sx={{
                        color: "text.primary",
                        textDecoration: "none",
                        "&:hover": { color: "primary.main" },
                      }}
                    >
                      {dataset.name}
                    </Typography>
                    {dataset.description && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        display="block"
                      >
                        {dataset.description}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <StatusChip status={dataset.status} />
                  </TableCell>
                  <TableCell>{dataset.row_count ?? "—"}</TableCell>
                  <TableCell sx={{ textTransform: "capitalize" }}>
                    {dataset.frequency ?? "—"}
                  </TableCell>
                  <TableCell>
                    {dataset.created_at
                      ? new Date(dataset.created_at).toLocaleDateString()
                      : "—"}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Delete dataset">
                      <IconButton
                        size="small"
                        onClick={() => setDeleteTarget(dataset)}
                      >
                        <DeleteOutlineIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Upload dialog */}
      <Dialog
        open={uploadOpen}
        onClose={() => !uploading && setUploadOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ fontWeight: 700 }}>Upload dataset</DialogTitle>
        <Box component="form" onSubmit={handleUpload}>
          <DialogContent>
            <Stack spacing={2}>
              {uploadError && <Alert severity="error">{uploadError}</Alert>}
              <Button
                variant="outlined"
                component="label"
                startIcon={<AddIcon />}
              >
                {uploadForm.file
                  ? uploadForm.file.name
                  : "Choose file (CSV, XLSX, JSON)"}
                <input
                  type="file"
                  hidden
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={(e) =>
                    setUploadForm((f) => ({
                      ...f,
                      file: e.target.files?.[0] || null,
                    }))
                  }
                />
              </Button>
              <TextField
                label="Name"
                required
                fullWidth
                value={uploadForm.name}
                onChange={(e) =>
                  setUploadForm((f) => ({ ...f, name: e.target.value }))
                }
              />
              <TextField
                label="Description"
                fullWidth
                multiline
                rows={2}
                value={uploadForm.description}
                onChange={(e) =>
                  setUploadForm((f) => ({ ...f, description: e.target.value }))
                }
              />
              <TextField
                select
                label="Frequency"
                fullWidth
                value={uploadForm.frequency}
                onChange={(e) =>
                  setUploadForm((f) => ({ ...f, frequency: e.target.value }))
                }
              >
                {FREQUENCIES.map((freq) => (
                  <MenuItem
                    key={freq}
                    value={freq}
                    sx={{ textTransform: "capitalize" }}
                  >
                    {freq}
                  </MenuItem>
                ))}
              </TextField>
              {uploading && (
                <LinearProgress variant="determinate" value={uploadProgress} />
              )}
            </Stack>
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 2 }}>
            <Button
              onClick={() => setUploadOpen(false)}
              disabled={uploading}
              color="inherit"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={uploading}
              disableElevation
            >
              {uploading ? `Uploading… ${uploadProgress}%` : "Upload"}
            </Button>
          </DialogActions>
        </Box>
      </Dialog>

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Delete dataset?"
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

export default Datasets;
