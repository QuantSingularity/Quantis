import DarkModeIcon from "@mui/icons-material/DarkModeOutlined";
import LogoutIcon from "@mui/icons-material/LogoutOutlined";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  FormControlLabel,
  Stack,
  Switch,
  Typography,
} from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useThemeMode } from "../context/ThemeModeContext";
import PageHeader from "../components/common/PageHeader";
import ConfirmDialog from "../components/common/ConfirmDialog";

const Settings = () => {
  const { mode, toggleMode } = useThemeMode();
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [logoutConfirmOpen, setLogoutConfirmOpen] = useState(false);

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <Box>
      <PageHeader
        title="Settings"
        description="Configure how Quantis looks and behaves for you."
      />

      <Stack spacing={3} sx={{ maxWidth: 640 }}>
        <Card>
          <CardContent>
            <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
              Appearance
            </Typography>
            <FormControlLabel
              control={
                <Switch checked={mode === "dark"} onChange={toggleMode} />
              }
              label={
                <Stack direction="row" spacing={1} alignItems="center">
                  <DarkModeIcon fontSize="small" />
                  <Typography variant="body2">Dark mode</Typography>
                </Stack>
              }
            />
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
              Notifications
            </Typography>
            <Stack spacing={1.5}>
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Model training updates"
              />
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Prediction completion alerts"
              />
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Compliance & risk alerts"
              />
              <FormControlLabel
                control={<Switch />}
                label="Weekly summary email"
              />
            </Stack>
            <Alert severity="info" sx={{ mt: 2 }}>
              Notification preferences are applied to in-app notifications
              immediately.
            </Alert>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Typography
              variant="subtitle1"
              fontWeight={700}
              sx={{ mb: 2, color: "error.main" }}
            >
              Session
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Sign out of Quantis on this device.
            </Typography>
            <Button
              color="error"
              variant="outlined"
              startIcon={<LogoutIcon />}
              onClick={() => setLogoutConfirmOpen(true)}
            >
              Sign out
            </Button>
          </CardContent>
        </Card>
      </Stack>

      <ConfirmDialog
        open={logoutConfirmOpen}
        title="Sign out?"
        description="You'll need to sign in again to access your workspace."
        confirmLabel="Sign out"
        destructive
        onConfirm={handleLogout}
        onClose={() => setLogoutConfirmOpen(false)}
      />
    </Box>
  );
};

export default Settings;
