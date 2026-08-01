import HistoryIcon from "@mui/icons-material/History";
import MonitorHeartIcon from "@mui/icons-material/MonitorHeartOutlined";
import {
  Alert,
  Box,
  Card,
  CardContent,
  Grid,
  LinearProgress,
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
import { getErrorMessage, monitoringAPI } from "../api";
import EmptyState from "../components/common/EmptyState";
import LoadingScreen from "../components/common/LoadingScreen";
import PageHeader from "../components/common/PageHeader";
import StatusChip from "../components/common/StatusChip";

const UsageBar = ({ label, value }) => (
  <Box sx={{ mb: 2 }}>
    <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" fontWeight={600}>
        {value ?? 0}%
      </Typography>
    </Stack>
    <LinearProgress
      variant="determinate"
      value={Math.min(value ?? 0, 100)}
      color={value > 85 ? "error" : value > 65 ? "warning" : "success"}
      sx={{ height: 8, borderRadius: 4 }}
    />
  </Box>
);

const AdminSystem = () => {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const results = await Promise.allSettled([
        monitoringAPI.health(),
        monitoringAPI.stats(),
        monitoringAPI.auditLogs({ limit: 20 }),
      ]);
      if (cancelled) return;
      const [healthRes, statsRes, auditRes] = results;
      if (healthRes.status === "fulfilled") setHealth(healthRes.value.data);
      else setError(getErrorMessage(healthRes.reason));
      if (statsRes.status === "fulfilled") setStats(statsRes.value.data);
      if (auditRes.status === "fulfilled") {
        setAuditLogs(auditRes.value.data?.items || auditRes.value.data || []);
      }
      setLoading(false);
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <LoadingScreen label="Loading system health…" />;

  return (
    <Box>
      <PageHeader
        title="System health"
        description="Live infrastructure status and recent audit activity."
        breadcrumb="Admin"
      />

      {error && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={2.5} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Stack
                direction="row"
                justifyContent="space-between"
                alignItems="center"
                sx={{ mb: 1.5 }}
              >
                <Typography variant="subtitle1" fontWeight={700}>
                  Overall status
                </Typography>
                <MonitorHeartIcon color="primary" />
              </Stack>
              <StatusChip status={health?.status || "unknown"} size="medium" />
              <Stack spacing={0.5} sx={{ mt: 2 }}>
                <Stack direction="row" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    Database
                  </Typography>
                  <StatusChip status={health?.database_status || "unknown"} />
                </Stack>
                <Stack direction="row" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    API
                  </Typography>
                  <StatusChip status={health?.api_status || "unknown"} />
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
                Resource usage
              </Typography>
              <UsageBar label="CPU" value={health?.cpu_usage} />
              <UsageBar label="Memory" value={health?.memory_usage?.percent} />
              <UsageBar label="Disk" value={health?.disk_usage?.percent} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {stats && (
        <Grid container spacing={2.5} sx={{ mb: 3 }}>
          {Object.entries(stats)
            .filter(([, v]) => typeof v === "number" || typeof v === "string")
            .slice(0, 4)
            .map(([key, value]) => (
              <Grid item xs={6} md={3} key={key}>
                <Card>
                  <CardContent>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ textTransform: "capitalize" }}
                    >
                      {key.replace(/_/g, " ")}
                    </Typography>
                    <Typography variant="h5" fontWeight={700} sx={{ mt: 0.5 }}>
                      {value}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
        </Grid>
      )}

      <Typography variant="h6" sx={{ mb: 1.5 }}>
        Recent audit activity
      </Typography>
      {auditLogs.length === 0 ? (
        <Paper sx={{ p: 2 }}>
          <EmptyState
            icon={<HistoryIcon fontSize="inherit" />}
            title="No audit events yet"
          />
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Action</TableCell>
                <TableCell>Resource</TableCell>
                <TableCell>User</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Time</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {auditLogs.map((log) => (
                <TableRow key={log.id} hover>
                  <TableCell sx={{ textTransform: "capitalize" }}>
                    {String(log.action || "").replace(/_/g, " ")}
                  </TableCell>
                  <TableCell>
                    {log.resource_name || log.resource_type}
                  </TableCell>
                  <TableCell>{log.user_id ?? "system"}</TableCell>
                  <TableCell>
                    <StatusChip
                      status={
                        log.status_code && log.status_code < 400
                          ? "success"
                          : "error"
                      }
                    />
                  </TableCell>
                  <TableCell>
                    {log.created_at
                      ? new Date(log.created_at).toLocaleString()
                      : "-"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default AdminSystem;
