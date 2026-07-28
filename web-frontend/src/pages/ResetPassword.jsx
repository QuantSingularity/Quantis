import LockResetIcon from "@mui/icons-material/LockReset";
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
import {
  Link as RouterLink,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import { authAPI, getErrorMessage } from "../api";

const ResetPassword = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const tokenFromUrl = searchParams.get("token") || "";

  const [form, setForm] = useState({
    token: tokenFromUrl,
    newPassword: "",
    confirmPassword: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const handleChange = (field) => (e) =>
    setForm((f) => ({ ...f, [field]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    if (form.newPassword !== form.confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    setSubmitting(true);
    try {
      await authAPI.resetPassword(form.token, form.newPassword);
      setSuccess(true);
      setTimeout(() => navigate("/login"), 1800);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  if (success) {
    return (
      <Stack alignItems="center" spacing={2} textAlign="center">
        <Typography variant="h5" fontWeight={700}>
          Password updated ✅
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Redirecting you to sign in…
        </Typography>
      </Stack>
    );
  }

  return (
    <Box>
      <Stack alignItems="center" spacing={1} sx={{ mb: 3 }}>
        <LockResetIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>
          Choose a new password
        </Typography>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box component="form" onSubmit={handleSubmit}>
        <Stack spacing={2}>
          {!tokenFromUrl && (
            <TextField
              label="Reset token"
              value={form.token}
              onChange={handleChange("token")}
              required
              fullWidth
              helperText="Paste the token from your reset email"
            />
          )}
          <TextField
            label="New password"
            type="password"
            value={form.newPassword}
            onChange={handleChange("newPassword")}
            required
            fullWidth
            helperText="At least 12 characters, mixed case, digit, and symbol"
          />
          <TextField
            label="Confirm new password"
            type="password"
            value={form.confirmPassword}
            onChange={handleChange("confirmPassword")}
            required
            fullWidth
          />
          <Button
            type="submit"
            variant="contained"
            size="large"
            disabled={submitting}
            disableElevation
          >
            {submitting ? "Updating…" : "Update password"}
          </Button>
        </Stack>
      </Box>

      <Typography
        variant="body2"
        color="text.secondary"
        textAlign="center"
        sx={{ mt: 3 }}
      >
        <MuiLink component={RouterLink} to="/login" fontWeight={600}>
          Back to sign in
        </MuiLink>
      </Typography>
    </Box>
  );
};

export default ResetPassword;
