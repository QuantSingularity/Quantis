import MailOutlineIcon from "@mui/icons-material/MailOutline";
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
import { Link as RouterLink } from "react-router-dom";
import { authAPI, getErrorMessage } from "../api";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState(null);
  const [devToken, setDevToken] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const { data } = await authAPI.forgotPassword(email);
      setSent(true);
      if (data?.reset_token) setDevToken(data.reset_token);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  if (sent) {
    return (
      <Stack alignItems="center" spacing={2} textAlign="center">
        <MailOutlineIcon color="primary" sx={{ fontSize: 36 }} />
        <Typography variant="h5" fontWeight={700}>
          Check your email
        </Typography>
        <Typography variant="body2" color="text.secondary">
          If an account exists for <strong>{email}</strong>, we&apos;ve sent a
          link to reset your password. The link expires in 30 minutes.
        </Typography>
        {devToken && (
          <Alert severity="info" sx={{ textAlign: "left", width: "100%" }}>
            Dev mode — no SMTP configured. Use this link directly:
            <br />
            <MuiLink
              component={RouterLink}
              to={`/reset-password?token=${devToken}`}
            >
              Reset your password
            </MuiLink>
          </Alert>
        )}
        <MuiLink
          component={RouterLink}
          to="/login"
          variant="body2"
          fontWeight={600}
        >
          Back to sign in
        </MuiLink>
      </Stack>
    );
  }

  return (
    <Box>
      <Stack alignItems="center" spacing={1} sx={{ mb: 3 }}>
        <MailOutlineIcon color="primary" />
        <Typography variant="h5" fontWeight={700}>
          Reset your password
        </Typography>
        <Typography variant="body2" color="text.secondary" textAlign="center">
          Enter your email and we&apos;ll send you a reset link
        </Typography>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box component="form" onSubmit={handleSubmit}>
        <Stack spacing={2}>
          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            fullWidth
            autoFocus
          />
          <Button
            type="submit"
            variant="contained"
            size="large"
            disabled={submitting}
            disableElevation
          >
            {submitting ? "Sending…" : "Send reset link"}
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

export default ForgotPassword;
