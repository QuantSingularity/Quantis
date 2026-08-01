import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import DownloadIcon from "@mui/icons-material/Download";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import { Link as RouterLink, useParams } from "react-router-dom";
import { datasetsAPI, getErrorMessage } from "../api";
import LoadingScreen from "../components/common/LoadingScreen";
import StatusChip from "../components/common/StatusChip";

const DatasetDetail = () => {
  const { datasetId } = useParams();

  const [dataset, setDataset] = useState(null);
  const [stats, setStats] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [datasetRes, statsRes, previewRes] = await Promise.allSettled([
          datasetsAPI.get(datasetId),
          datasetsAPI.stats(datasetId),
          datasetsAPI.preview(datasetId, 15),
        ]);
        if (cancelled) return;
        if (datasetRes.status === "fulfilled")
          setDataset(datasetRes.value.data);
        else setError(getErrorMessage(datasetRes.reason));
        if (statsRes.status === "fulfilled") setStats(statsRes.value.data);
        if (previewRes.status === "fulfilled")
          setPreview(previewRes.value.data);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  if (loading) return <LoadingScreen label="Loading dataset…" />;
  if (error && !dataset) return <Alert severity="error">{error}</Alert>;

  const previewRows = preview?.rows || preview?.data || [];
  const previewColumns =
    preview?.columns || (previewRows[0] ? Object.keys(previewRows[0]) : []);

  return (
    <Box>
      <Button
        startIcon={<ArrowBackIcon />}
        component={RouterLink}
        to="/app/datasets"
        sx={{ mb: 2 }}
        color="inherit"
      >
        Back to datasets
      </Button>

      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="flex-start"
        sx={{ mb: 3 }}
      >
        <Box>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <Typography variant="h4" fontWeight={700}>
              {dataset?.name}
            </Typography>
            <StatusChip status={dataset?.status} />
          </Stack>
          {dataset?.description && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              {dataset.description}
            </Typography>
          )}
        </Box>
        <Button
          variant="outlined"
          startIcon={<DownloadIcon />}
          component="a"
          href={datasetsAPI.downloadUrl(datasetId)}
          target="_blank"
          rel="noopener noreferrer"
        >
          Download
        </Button>
      </Stack>

      <Grid container spacing={2.5} sx={{ mb: 3 }}>
        {[
          {
            label: "Rows",
            value: stats?.row_count ?? dataset?.row_count ?? "-",
          },
          { label: "Columns", value: stats?.column_count ?? "-" },
          { label: "Missing values", value: stats?.missing_values ?? "-" },
          { label: "Frequency", value: dataset?.frequency ?? "-" },
        ].map((item) => (
          <Grid item xs={6} md={3} key={item.label}>
            <Card>
              <CardContent>
                <Typography variant="body2" color="text.secondary">
                  {item.label}
                </Typography>
                <Typography
                  variant="h5"
                  fontWeight={700}
                  sx={{ mt: 0.5, textTransform: "capitalize" }}
                >
                  {item.value}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h6" sx={{ mb: 1.5 }}>
        Data preview
      </Typography>
      {previewRows.length === 0 ? (
        <Paper sx={{ p: 3 }}>
          <Typography variant="body2" color="text.secondary">
            No preview available for this dataset yet.
          </Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                {previewColumns.map((col) => (
                  <TableCell key={col} sx={{ fontWeight: 700 }}>
                    {col}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {previewRows.map((row, idx) => (
                <TableRow key={idx} hover>
                  {previewColumns.map((col) => (
                    <TableCell
                      key={col}
                      sx={{ fontFamily: "monospace", fontSize: 12.5 }}
                    >
                      {String(row[col] ?? "")}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default DatasetDetail;
