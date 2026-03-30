import { CheckCircle, CloudUpload, Description } from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Grid,
  LinearProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
  useTheme,
} from "@mui/material";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { datasetsAPI, handleApiError } from "../services/api";

const DatasetUpload = () => {
  const theme = useTheme();
  const [uploadState, setUploadState] = useState({
    file: null,
    name: "",
    description: "",
    isUploading: false,
    uploadProgress: 0,
    error: null,
    success: false,
    uploadedDataset: null,
    preview: null,
  });

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      setUploadState((prev) => ({
        ...prev,
        error: "Invalid file type. Please upload CSV, JSON, or Excel files.",
      }));
      return;
    }

    const file = acceptedFiles[0];
    if (file) {
      setUploadState((prev) => ({
        ...prev,
        file,
        name: file.name.replace(/\.[^/.]+$/, ""), // Remove extension
        error: null,
        success: false,
      }));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/json": [".json"],
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const handleUpload = async () => {
    if (!uploadState.file || !uploadState.name.trim()) {
      setUploadState((prev) => ({
        ...prev,
        error: "Please select a file and provide a name.",
      }));
      return;
    }

    setUploadState((prev) => ({
      ...prev,
      isUploading: true,
      uploadProgress: 0,
      error: null,
    }));

    try {
      const formData = new FormData();
      formData.append("file", uploadState.file);
      formData.append("name", uploadState.name);
      formData.append("description", uploadState.description);

      const response = await datasetsAPI.uploadDataset(
        formData,
        (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total,
          );
          setUploadState((prev) => ({
            ...prev,
            uploadProgress: progress,
          }));
        },
      );

      // Get preview of uploaded dataset
      const preview = await datasetsAPI.getDatasetPreview(response.data.id, 5);

      setUploadState((prev) => ({
        ...prev,
        isUploading: false,
        success: true,
        uploadedDataset: response.data,
        preview: preview.data,
      }));
    } catch (error) {
      const apiError = handleApiError(error);
      setUploadState((prev) => ({
        ...prev,
        isUploading: false,
        error: apiError.message,
      }));
    }
  };

  const handleReset = () => {
    setUploadState({
      file: null,
      name: "",
      description: "",
      isUploading: false,
      uploadProgress: 0,
      error: null,
      success: false,
      uploadedDataset: null,
      preview: null,
    });
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`;
  };

  if (uploadState.success && uploadState.uploadedDataset) {
    return (
      <Box sx={{ p: 3 }}>
        <Card>
          <CardContent>
            <Box sx={{ textAlign: "center", mb: 3 }}>
              <CheckCircle
                sx={{ fontSize: 64, color: theme.palette.success.main, mb: 2 }}
              />
              <Typography variant="h5" gutterBottom>
                Dataset Uploaded Successfully!
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Your dataset "{uploadState.uploadedDataset.name}" has been
                processed and is ready to use.
              </Typography>
            </Box>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Dataset Information
                  </Typography>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Name
                    </Typography>
                    <Typography variant="body1">
                      {uploadState.uploadedDataset.name}
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Rows
                    </Typography>
                    <Typography variant="body1">
                      {uploadState.uploadedDataset.row_count?.toLocaleString()}
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Columns
                    </Typography>
                    <Typography variant="body1">
                      {uploadState.uploadedDataset.columns?.length}
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      File Size
                    </Typography>
                    <Typography variant="body1">
                      {formatFileSize(uploadState.uploadedDataset.file_size)}
                    </Typography>
                  </Box>
                </Paper>
              </Grid>

              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Column Types
                  </Typography>
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                    {uploadState.uploadedDataset.metadata?.numeric_columns?.map(
                      (col) => (
                        <Chip
                          key={col}
                          label={col}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      ),
                    )}
                    {uploadState.uploadedDataset.metadata?.categorical_columns?.map(
                      (col) => (
                        <Chip
                          key={col}
                          label={col}
                          size="small"
                          color="secondary"
                          variant="outlined"
                        />
                      ),
                    )}
                  </Box>
                </Paper>
              </Grid>

              {uploadState.preview && (
                <Grid item xs={12}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Data Preview
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            {uploadState.preview.columns.map((column) => (
                              <TableCell key={column}>{column}</TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {uploadState.preview.data.map((row, index) => (
                            <TableRow key={index}>
                              {uploadState.preview.columns.map((column) => (
                                <TableCell key={column}>
                                  {row[column]?.toString() || ""}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Paper>
                </Grid>
              )}
            </Grid>

            <Box sx={{ mt: 3, textAlign: "center" }}>
              <Button variant="contained" onClick={handleReset} sx={{ mr: 2 }}>
                Upload Another Dataset
              </Button>
              <Button variant="outlined" href="/datasets">
                View All Datasets
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Upload Dataset
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload your dataset to start building machine learning models. Supported
        formats: CSV, JSON, Excel (.xlsx, .xls)
      </Typography>

      <Card>
        <CardContent>
          {uploadState.error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {uploadState.error}
            </Alert>
          )}

          <Box
            {...getRootProps()}
            sx={{
              border: `2px dashed ${
                isDragActive
                  ? theme.palette.primary.main
                  : theme.palette.divider
              }`,
              borderRadius: 2,
              p: 4,
              textAlign: "center",
              cursor: "pointer",
              backgroundColor: isDragActive
                ? theme.palette.action.hover
                : "transparent",
              transition: "all 0.2s ease-in-out",
              mb: 3,
            }}
          >
            <input {...getInputProps()} />
            <CloudUpload
              sx={{
                fontSize: 48,
                color: theme.palette.text.secondary,
                mb: 2,
              }}
            />
            <Typography variant="h6" gutterBottom>
              {isDragActive
                ? "Drop the file here"
                : "Drag & drop a file here, or click to select"}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Maximum file size: 10MB
            </Typography>
          </Box>

          {uploadState.file && (
            <Paper
              sx={{
                p: 2,
                mb: 3,
                backgroundColor: theme.palette.background.default,
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <Description
                  sx={{ mr: 1, color: theme.palette.text.secondary }}
                />
                <Typography variant="body1">{uploadState.file.name}</Typography>
                <Chip
                  label={formatFileSize(uploadState.file.size)}
                  size="small"
                  sx={{ ml: 2 }}
                />
              </Box>
            </Paper>
          )}

          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Dataset Name"
                value={uploadState.name}
                onChange={(e) =>
                  setUploadState((prev) => ({ ...prev, name: e.target.value }))
                }
                required
                disabled={uploadState.isUploading}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Description (Optional)"
                value={uploadState.description}
                onChange={(e) =>
                  setUploadState((prev) => ({
                    ...prev,
                    description: e.target.value,
                  }))
                }
                disabled={uploadState.isUploading}
              />
            </Grid>
          </Grid>

          {uploadState.isUploading && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="body2" gutterBottom>
                Uploading... {uploadState.uploadProgress}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={uploadState.uploadProgress}
              />
            </Box>
          )}

          <Box sx={{ textAlign: "center" }}>
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={
                !uploadState.file ||
                !uploadState.name.trim() ||
                uploadState.isUploading
              }
              size="large"
            >
              {uploadState.isUploading ? "Uploading..." : "Upload Dataset"}
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default DatasetUpload;
