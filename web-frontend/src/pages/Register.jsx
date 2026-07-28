import PersonAddOutlinedIcon from "@mui/icons-material/PersonAddOutlined";
import {
  Alert,
  Box,
  Button,
  Grid,
  LinearProgress,
  Link as MuiLink,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useMemo, useState } from "react";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const initialForm = {
  username: "",
  email: "",
  firstName: "",
  lastName: "",
  password: "",
  confirmPassword: "",
};

const PASSWORD_RULES = [
  { test: (v) => v.length >= 12, label: "At least 12 characters" },
  { test: (v) => /[A-Z]/.test(v), label: "One uppercase letter" },
  { test: (v) => /[a-z]/.test(v), label: "One lowercase letter" },
  { test: (v) => /\d/.test(v), label: "One digit" },
  {
    test: (v) => /[!@#$%^&*()_+\-=[\]{}|;:,.<>?]/.test(v),
    label: "One special character",
  },
];

const Register = () => {
  const { register, authError, setAuthError } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState(initialForm);
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [fieldErrors, setFieldErrors] = useState({});

  const handleChange = (field) => (e) =>
    setForm((f) => ({ ...f, [field]: e.target.value }));

  const passedRules = useMemo(
    () => PASSWORD_RULES.filter((rule) => rule.test(form.password)).length,
    [form.password],
  );
  const passwordStrength = (passedRules / PASSWORD_RULES.length) * 100;
  const passwordsMatch =
    form.password && form.password === form.confirmPassword;

  const validate = () => {
    const errors = {};
    if (form.username.length < 3) errors.username = "Minimum 3 characters";
    if (!/^[a-zA-Z0-9_]+$/.test(form.username)) {
      errors.username = "Letters, numbers and underscores only";
    }
    if (passedRules < PASSWORD_RULES.length)
      errors.password = "Password does not meet all requirements";
    if (!passwordsMatch) errors.confirmPassword = "Passwords do not match";
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAuthError(null);
    if (!validate()) return;

    setSubmitting(true);
    const result = await register({
      username: form.username,
      email: form.email,
      first_name: form.firstName || undefined,
      last_name: form.lastName || undefined,
      password: form.password,
      confirm_password: form.confirmPassword,
    });
    setSubmitting(false);

    if (result.success) {
      setSuccess(true);
      setTimeout(() => navigate("/login"), 1800);
    }
  };

  if (success) {
    return (
      <Stack alignItems="center" spacing={2} sx={{ py: 3 }}>
        <Typography variant="h5" fontWeight={700}>
          Account created 🎉
        </Typography>
        <Typography variant="body2" color="text.secondary" textAlign="center">
          Redirecting you to sign in…
        </Typography>
      </Stack>
    );
  }

  return (
    <Box>
      <Stack alignItems="center" spacing={1} sx={{ mb: 3 }}>
        <PersonAddOutlinedIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>
          Create your account
        </Typography>
        <Typography variant="body2" color="text.secondary" textAlign="center">
          Start forecasting in minutes — no credit card required
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
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                label="First name"
                value={form.firstName}
                onChange={handleChange("firstName")}
                fullWidth
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                label="Last name"
                value={form.lastName}
                onChange={handleChange("lastName")}
                fullWidth
              />
            </Grid>
          </Grid>

          <TextField
            label="Username"
            value={form.username}
            onChange={handleChange("username")}
            required
            fullWidth
            autoComplete="username"
            error={Boolean(fieldErrors.username)}
            helperText={
              fieldErrors.username ||
              "3-50 characters, letters/numbers/underscore"
            }
          />
          <TextField
            label="Email"
            type="email"
            value={form.email}
            onChange={handleChange("email")}
            required
            fullWidth
            autoComplete="email"
          />
          <TextField
            label="Password"
            type="password"
            value={form.password}
            onChange={handleChange("password")}
            required
            fullWidth
            autoComplete="new-password"
            error={Boolean(fieldErrors.password)}
          />
          {form.password && (
            <Box>
              <LinearProgress
                variant="determinate"
                value={passwordStrength}
                color={
                  passwordStrength === 100
                    ? "success"
                    : passwordStrength >= 60
                      ? "warning"
                      : "error"
                }
                sx={{ height: 6, borderRadius: 3, mb: 1 }}
              />
              <Stack spacing={0.25}>
                {PASSWORD_RULES.map((rule) => (
                  <Typography
                    key={rule.label}
                    variant="caption"
                    sx={{
                      color: rule.test(form.password)
                        ? "success.main"
                        : "text.secondary",
                    }}
                  >
                    {rule.test(form.password) ? "✓" : "○"} {rule.label}
                  </Typography>
                ))}
              </Stack>
            </Box>
          )}
          <TextField
            label="Confirm password"
            type="password"
            value={form.confirmPassword}
            onChange={handleChange("confirmPassword")}
            required
            fullWidth
            autoComplete="new-password"
            error={Boolean(fieldErrors.confirmPassword)}
            helperText={fieldErrors.confirmPassword}
          />

          <Button
            type="submit"
            variant="contained"
            size="large"
            disabled={submitting}
            disableElevation
          >
            {submitting ? "Creating account…" : "Create account"}
          </Button>
        </Stack>
      </Box>

      <Typography
        variant="body2"
        color="text.secondary"
        textAlign="center"
        sx={{ mt: 3 }}
      >
        Already have an account?{" "}
        <MuiLink component={RouterLink} to="/login" fontWeight={600}>
          Sign in
        </MuiLink>
      </Typography>
    </Box>
  );
};

export default Register;
