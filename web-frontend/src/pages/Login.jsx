import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import {
  Alert,
  Box,
  Button,
  Link as MuiLink,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useState } from "react";
import { Link as RouterLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const Login = () => {
  const { login, authError, setAuthError } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || "/app/dashboard";

  const [form, setForm] = useState({ username: "", password: "", mfaCode: "" });
  const [needsMfa, setNeedsMfa] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const handleChange = (field) => (e) =>
    setForm((f) => ({ ...f, [field]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAuthError(null);
    setSubmitting(true);
    const result = await login(
      form.username,
      form.password,
      form.mfaCode || undefined,
    );
    setSubmitting(false);

    if (result.success) {
      navigate(from, { replace: true });
      return;
    }
    if (result.status === 403 && /mfa/i.test(result.error || "")) {
      setNeedsMfa(true);
    }
  };

  return (
    <Box>
      <Stack alignItems="center" spacing={1} sx={{ mb: 3 }}>
        <LockOutlinedIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>
          Welcome back
        </Typography>
        <Typography variant="body2" color="text.secondary" textAlign="center">
          Sign in to access your Quantis workspace
        </Typography>
      </Stack>

      {authError && (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          onClose={() => setAuthError(null)}
        >
          {authError}
        </Alert>
      )}

      <Box component="form" onSubmit={handleSubmit} noValidate>
        <Stack spacing={2}>
          <TextField
            label="Username or email"
            value={form.username}
            onChange={handleChange("username")}
            required
            fullWidth
            autoFocus
            autoComplete="username"
          />
          <TextField
            label="Password"
            type="password"
            value={form.password}
            onChange={handleChange("password")}
            required
            fullWidth
            autoComplete="current-password"
          />
          {needsMfa && (
            <TextField
              label="Authenticator code"
              value={form.mfaCode}
              onChange={handleChange("mfaCode")}
              required
              fullWidth
              inputProps={{ inputMode: "numeric", maxLength: 6 }}
              helperText="Enter the 6-digit code from your authenticator app"
            />
          )}

          <Box sx={{ textAlign: "right" }}>
            <MuiLink
              component={RouterLink}
              to="/forgot-password"
              variant="body2"
            >
              Forgot password?
            </MuiLink>
          </Box>

          <Button
            type="submit"
            variant="contained"
            size="large"
            disabled={submitting}
            disableElevation
          >
            {submitting ? "Signing in…" : "Sign in"}
          </Button>
        </Stack>
      </Box>

      <Typography
        variant="body2"
        color="text.secondary"
        textAlign="center"
        sx={{ mt: 3 }}
      >
        Don&apos;t have an account?{" "}
        <MuiLink component={RouterLink} to="/register" fontWeight={600}>
          Create one
        </MuiLink>
      </Typography>
    </Box>
  );
};

export default Login;
