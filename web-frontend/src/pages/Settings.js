import SaveIcon from "@mui/icons-material/Save";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Divider,
  FormControlLabel,
  Grid,
  Snackbar,
  Switch,
  TextField,
  Typography,
  useTheme,
} from "@mui/material";
import { useState } from "react";

const Settings = () => {
  const _theme = useTheme();
  const [formData, setFormData] = useState({
    apiKey: "sk_test_***********************",
    apiEndpoint: "https://api.quantis.ai/v2",
    enableNotifications: true,
    enableAutoUpdates: true,
    dataRefreshInterval: 5,
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: "",
    severity: "success",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSwitchChange = (e) => {
    const { name, checked } = e.target;
    setFormData({
      ...formData,
      [name]: checked,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simulate saving settings
    setSnackbar({
      open: true,
      message: "Settings saved successfully!",
      severity: "success",
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({
      ...snackbar,
      open: false,
    });
  };

  return (
    <Box className="fade-in">
      <Typography variant="h4" gutterBottom fontWeight="600">
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure application settings and preferences
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="API Configuration"
              subheader="Manage API connection settings"
            />
            <Divider />
            <CardContent>
              <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="apiKey"
                  label="API Key"
                  name="apiKey"
                  value={formData.apiKey}
                  onChange={handleChange}
                  type="password"
                />

                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="apiEndpoint"
                  label="API Endpoint"
                  name="apiEndpoint"
                  value={formData.apiEndpoint}
                  onChange={handleChange}
                />

                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="dataRefreshInterval"
                  label="Data Refresh Interval (minutes)"
                  name="dataRefreshInterval"
                  type="number"
                  value={formData.dataRefreshInterval}
                  onChange={handleChange}
                  inputProps={{ min: 1, max: 60 }}
                />

                <Button
                  type="submit"
                  variant="contained"
                  startIcon={<SaveIcon />}
                  sx={{ mt: 3 }}
                >
                  Save API Settings
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Application Preferences"
              subheader="Customize application behavior"
            />
            <Divider />
            <CardContent>
              <Box component="form" sx={{ mt: 1 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={formData.enableNotifications}
                      onChange={handleSwitchChange}
                      name="enableNotifications"
                      color="primary"
                    />
                  }
                  label="Enable Notifications"
                />

                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ ml: 4, mb: 2 }}
                >
                  Receive alerts for model performance changes and prediction
                  results
                </Typography>

                <FormControlLabel
                  control={
                    <Switch
                      checked={formData.enableAutoUpdates}
                      onChange={handleSwitchChange}
                      name="enableAutoUpdates"
                      color="primary"
                    />
                  }
                  label="Enable Automatic Updates"
                />

                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ ml: 4, mb: 2 }}
                >
                  Automatically update to the latest model versions when
                  available
                </Typography>

                <Divider sx={{ my: 2 }} />

                <Alert severity="info" sx={{ mt: 2 }}>
                  Your current model version is 2.1.0. No updates available at
                  this time.
                </Alert>

                <Button variant="outlined" sx={{ mt: 3 }}>
                  Check for Updates
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Account Information"
              subheader="View and manage your account details"
            />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Account Type
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    Enterprise
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Subscription Status
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    Active
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    API Requests (This Month)
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    1,284 / 10,000
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Next Billing Date
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    May 1, 2025
                  </Typography>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ display: "flex", gap: 2, mt: 2 }}>
                <Button variant="outlined">Manage Subscription</Button>
                <Button variant="outlined" color="error">
                  Reset API Key
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          sx={{ width: "100%" }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Settings;
