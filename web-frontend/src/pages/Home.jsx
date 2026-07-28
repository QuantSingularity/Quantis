import BoltIcon from "@mui/icons-material/BoltOutlined";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import InsightsIcon from "@mui/icons-material/InsightsOutlined";
import SecurityIcon from "@mui/icons-material/GppGoodOutlined";
import ShowChartIcon from "@mui/icons-material/ShowChartOutlined";
import StorageIcon from "@mui/icons-material/StorageOutlined";
import TimelineIcon from "@mui/icons-material/TimelineOutlined";
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Grid,
  Stack,
  Typography,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

const FEATURES = [
  {
    icon: StorageIcon,
    title: "Unified data pipeline",
    description:
      "Ingest, validate, and version time-series datasets with automated quality scoring before they ever reach a model.",
  },
  {
    icon: TimelineIcon,
    title: "Multiple model families",
    description:
      "Train and compare TFT, LSTM, ARIMA, Prophet, Random Forest, XGBoost and ensemble models from a single workspace.",
  },
  {
    icon: InsightsIcon,
    title: "Real-time predictions",
    description:
      "Serve single or batch predictions with sub-second latency, full audit trails, and confidence intervals.",
  },
  {
    icon: ShowChartIcon,
    title: "Financial risk tooling",
    description:
      "Built-in transaction risk scoring, compliance limits, interest and NPV calculators for finance teams.",
  },
  {
    icon: SecurityIcon,
    title: "Enterprise-grade security",
    description:
      "Role-based access control, MFA, API key scoping, and a full audit log out of the box.",
  },
  {
    icon: BoltIcon,
    title: "Built to integrate",
    description:
      "A documented REST API plus native web and mobile clients so your whole team works from the same source of truth.",
  },
];

const STEPS = [
  {
    title: "Connect your data",
    description: "Upload a dataset or connect a live source in minutes.",
  },
  {
    title: "Train & compare models",
    description:
      "Spin up multiple model types and compare metrics side by side.",
  },
  {
    title: "Predict with confidence",
    description:
      "Ship predictions to your product with a documented, versioned API.",
  },
];

const Home = () => {
  return (
    <Box>
      {/* Hero */}
      <Container
        maxWidth="lg"
        sx={{ pt: { xs: 8, md: 12 }, pb: { xs: 8, md: 10 } }}
      >
        <Stack spacing={3} alignItems="center" textAlign="center">
          <Chip
            label="Now with real-time model health monitoring"
            size="small"
            sx={{ fontWeight: 600 }}
            color="primary"
            variant="outlined"
          />
          <Typography
            variant="h1"
            sx={{
              fontSize: { xs: 36, sm: 48, md: 60 },
              maxWidth: 820,
              lineHeight: 1.1,
            }}
          >
            Financial forecasting, from raw data to production prediction.
          </Typography>
          <Typography
            variant="h6"
            color="text.secondary"
            sx={{ maxWidth: 640, fontWeight: 400 }}
          >
            Quantis is the platform quant and data teams use to manage datasets,
            train forecasting models, and serve predictions — all with the audit
            trail and compliance controls finance requires.
          </Typography>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={2}
            sx={{ pt: 1 }}
          >
            <Button
              component={RouterLink}
              to="/register"
              variant="contained"
              size="large"
              disableElevation
              sx={{ px: 4 }}
            >
              Get started free
            </Button>
            <Button
              component={RouterLink}
              to="/login"
              variant="outlined"
              size="large"
              sx={{ px: 4 }}
            >
              Sign in
            </Button>
          </Stack>
          <Stack
            direction="row"
            spacing={3}
            sx={{ pt: 2 }}
            flexWrap="wrap"
            justifyContent="center"
          >
            {[
              "No credit card required",
              "Free tier included",
              "Cancel anytime",
            ].map((t) => (
              <Stack key={t} direction="row" spacing={0.75} alignItems="center">
                <CheckCircleIcon sx={{ fontSize: 16, color: "success.main" }} />
                <Typography variant="body2" color="text.secondary">
                  {t}
                </Typography>
              </Stack>
            ))}
          </Stack>
        </Stack>
      </Container>

      {/* Features */}
      <Box
        id="product"
        sx={{
          bgcolor: "background.paper",
          borderTop: "1px solid",
          borderBottom: "1px solid",
          borderColor: "divider",
        }}
      >
        <Container maxWidth="lg" sx={{ py: { xs: 8, md: 10 } }}>
          <Stack
            spacing={1}
            alignItems="center"
            textAlign="center"
            sx={{ mb: 6 }}
          >
            <Typography
              variant="overline"
              color="primary.main"
              fontWeight={700}
            >
              PLATFORM
            </Typography>
            <Typography variant="h3" sx={{ fontSize: { xs: 28, md: 36 } }}>
              Everything your forecasting workflow needs
            </Typography>
          </Stack>
          <Grid container spacing={3}>
            {FEATURES.map((feature) => {
              const Icon = feature.icon;
              return (
                <Grid item xs={12} sm={6} md={4} key={feature.title}>
                  <Card sx={{ height: "100%" }} elevation={0}>
                    <CardContent sx={{ p: 3 }}>
                      <Box
                        sx={{
                          width: 44,
                          height: 44,
                          borderRadius: 2,
                          bgcolor: "action.hover",
                          color: "primary.main",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          mb: 2,
                        }}
                      >
                        <Icon />
                      </Box>
                      <Typography variant="h6" sx={{ mb: 1 }}>
                        {feature.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {feature.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Container>
      </Box>

      {/* How it works */}
      <Container maxWidth="lg" sx={{ py: { xs: 8, md: 10 } }} id="how-it-works">
        <Stack
          spacing={1}
          alignItems="center"
          textAlign="center"
          sx={{ mb: 6 }}
        >
          <Typography variant="overline" color="primary.main" fontWeight={700}>
            HOW IT WORKS
          </Typography>
          <Typography variant="h3" sx={{ fontSize: { xs: 28, md: 36 } }}>
            Three steps to production
          </Typography>
        </Stack>
        <Grid container spacing={4}>
          {STEPS.map((step, index) => (
            <Grid item xs={12} md={4} key={step.title}>
              <Stack spacing={1.5}>
                <Typography
                  variant="h3"
                  sx={{ color: "primary.main", opacity: 0.35, fontWeight: 800 }}
                >
                  {String(index + 1).padStart(2, "0")}
                </Typography>
                <Typography variant="h6">{step.title}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {step.description}
                </Typography>
              </Stack>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Security */}
      <Box
        id="security"
        sx={{
          bgcolor: "background.paper",
          borderTop: "1px solid",
          borderColor: "divider",
        }}
      >
        <Container
          maxWidth="md"
          sx={{ py: { xs: 8, md: 10 }, textAlign: "center" }}
        >
          <SecurityIcon sx={{ fontSize: 40, color: "primary.main", mb: 2 }} />
          <Typography variant="h4" sx={{ mb: 2 }}>
            Security and compliance, built in
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
            Role-based access control, multi-factor authentication, scoped API
            keys, and a complete audit log ship with every Quantis workspace —
            not as an add-on.
          </Typography>
          <Button
            component={RouterLink}
            to="/register"
            variant="contained"
            size="large"
            disableElevation
          >
            Create your workspace
          </Button>
        </Container>
      </Box>
    </Box>
  );
};

export default Home;
