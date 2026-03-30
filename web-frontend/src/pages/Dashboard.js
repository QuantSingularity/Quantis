import AssessmentIcon from "@mui/icons-material/Assessment";
import TimelineIcon from "@mui/icons-material/Timeline";
import TrendingDownIcon from "@mui/icons-material/TrendingDown";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Grid,
  Typography,
  useTheme,
} from "@mui/material";
import BarChart from "../components/charts/BarChart";
import LineChart from "../components/charts/LineChart";
import StatCard from "../components/StatCard";

const Dashboard = () => {
  const theme = useTheme();

  // Mock data for charts
  const lineChartData = [
    {
      id: "Predictions",
      color: theme.palette.primary.main,
      data: [
        { x: "Jan", y: 45 },
        { x: "Feb", y: 52 },
        { x: "Mar", y: 48 },
        { x: "Apr", y: 61 },
        { x: "May", y: 55 },
        { x: "Jun", y: 67 },
        { x: "Jul", y: 71 },
      ],
    },
    {
      id: "Actual",
      color: theme.palette.secondary.main,
      data: [
        { x: "Jan", y: 42 },
        { x: "Feb", y: 49 },
        { x: "Mar", y: 51 },
        { x: "Apr", y: 58 },
        { x: "May", y: 53 },
        { x: "Jun", y: 65 },
        { x: "Jul", y: 68 },
      ],
    },
  ];

  const barChartData = [
    { feature: "Feature 1", value: 0.85 },
    { feature: "Feature 2", value: 0.72 },
    { feature: "Feature 3", value: 0.68 },
    { feature: "Feature 4", value: 0.56 },
    { feature: "Feature 5", value: 0.43 },
  ];

  return (
    <Box className="fade-in">
      <Typography variant="h4" gutterBottom fontWeight="600">
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Overview of your model performance and predictions
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Accuracy"
            value="92.7%"
            change="+2.1%"
            positive={true}
            icon={<AssessmentIcon />}
            color={theme.palette.primary.main}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Predictions"
            value="1,284"
            change="+12.5%"
            positive={true}
            icon={<TimelineIcon />}
            color={theme.palette.secondary.main}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="RMSE"
            value="0.042"
            change="-0.8%"
            positive={true}
            icon={<TrendingDownIcon />}
            color={theme.palette.success.main}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Drift"
            value="0.031"
            change="+0.4%"
            positive={false}
            icon={<TrendingUpIcon />}
            color={theme.palette.warning.main}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardHeader
              title="Prediction Performance"
              subheader="Comparison of predicted vs actual values"
              action={
                <Button size="small" color="primary">
                  View Details
                </Button>
              }
            />
            <Divider />
            <CardContent sx={{ height: 400 }}>
              <LineChart data={lineChartData} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} lg={4}>
          <Card>
            <CardHeader
              title="Feature Importance"
              subheader="Top 5 influential features"
            />
            <Divider />
            <CardContent sx={{ height: 400 }}>
              <BarChart data={barChartData} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Recent Predictions"
              subheader="Latest model predictions and their accuracy"
              action={
                <Button size="small" color="primary">
                  View All
                </Button>
              }
            />
            <Divider />
            <CardContent>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ py: 5, textAlign: "center" }}
              >
                Recent prediction data will appear here when available
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
