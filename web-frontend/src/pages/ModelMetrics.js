import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  useTheme,
} from "@mui/material";
import BarChart from "../components/charts/BarChart";
import LineChart from "../components/charts/LineChart";

const ModelMetrics = () => {
  const theme = useTheme();

  // Mock data for metrics
  const metrics = [
    { name: "Accuracy", value: "92.7%" },
    { name: "Precision", value: "91.3%" },
    { name: "Recall", value: "89.5%" },
    { name: "F1 Score", value: "90.4%" },
    { name: "RMSE", value: "0.042" },
    { name: "MAE", value: "0.037" },
  ];

  // Mock data for feature importance
  const featureImportance = [
    { feature: "Feature 1", value: 0.85 },
    { feature: "Feature 2", value: 0.72 },
    { feature: "Feature 3", value: 0.68 },
    { feature: "Feature 4", value: 0.56 },
    { feature: "Feature 5", value: 0.43 },
  ];

  // Mock data for model versions
  const modelVersions = [
    {
      version: "2.1.0",
      date: "2025-04-01",
      accuracy: "92.7%",
      status: "Active",
    },
    {
      version: "2.0.1",
      date: "2025-03-15",
      accuracy: "91.2%",
      status: "Archived",
    },
    {
      version: "2.0.0",
      date: "2025-02-28",
      accuracy: "90.5%",
      status: "Archived",
    },
    {
      version: "1.9.0",
      date: "2025-01-10",
      accuracy: "88.3%",
      status: "Archived",
    },
  ];

  return (
    <Box className="fade-in">
      <Typography variant="h4" gutterBottom fontWeight="600">
        Model Metrics
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Performance metrics and analytics for the current model
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Performance Metrics"
              subheader="Key metrics for model evaluation"
            />
            <Divider />
            <CardContent>
              <TableContainer component={Paper} elevation={0}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>
                        <Typography variant="subtitle2">Metric</Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="subtitle2">Value</Typography>
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {metrics.map((metric) => (
                      <TableRow key={metric.name}>
                        <TableCell component="th" scope="row">
                          {metric.name}
                        </TableCell>
                        <TableCell align="right">{metric.value}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Feature Importance"
              subheader="Top 5 influential features"
            />
            <Divider />
            <CardContent sx={{ height: 300 }}>
              <BarChart data={featureImportance} />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Model Versions"
              subheader="History of model deployments"
              action={
                <Button size="small" color="primary">
                  View Details
                </Button>
              }
            />
            <Divider />
            <CardContent>
              <TableContainer component={Paper} elevation={0}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>
                        <Typography variant="subtitle2">Version</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="subtitle2">
                          Deployment Date
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="subtitle2">Accuracy</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="subtitle2">Status</Typography>
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {modelVersions.map((version) => (
                      <TableRow key={version.version}>
                        <TableCell component="th" scope="row">
                          {version.version}
                        </TableCell>
                        <TableCell>{version.date}</TableCell>
                        <TableCell>{version.accuracy}</TableCell>
                        <TableCell>
                          <Box
                            sx={{
                              display: "inline-block",
                              px: 1,
                              py: 0.5,
                              borderRadius: 1,
                              backgroundColor:
                                version.status === "Active"
                                  ? `${theme.palette.success.main}20`
                                  : `${theme.palette.text.secondary}20`,
                              color:
                                version.status === "Active"
                                  ? theme.palette.success.main
                                  : theme.palette.text.secondary,
                            }}
                          >
                            {version.status}
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Performance Over Time"
              subheader="Model accuracy trends"
            />
            <Divider />
            <CardContent sx={{ height: 400 }}>
              <LineChart data={[]} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelMetrics;
