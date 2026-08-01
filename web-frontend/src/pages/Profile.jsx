import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import KeyIcon from "@mui/icons-material/KeyOutlined";
import PersonIcon from "@mui/icons-material/PersonOutlined";
import SecurityIcon from "@mui/icons-material/SecurityOutlined";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Stack,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import { authAPI, getErrorMessage } from "../api";
import ConfirmDialog from "../components/common/ConfirmDialog";
import PageHeader from "../components/common/PageHeader";
import { useAuth } from "../context/AuthContext";

const AccountTab = () => {
  const { user, refreshUser } = useAuth();
  const [form, setForm] = useState({
    first_name: user?.first_name || "",
    last_name: user?.last_name || "",
    phone_number: user?.phone_number || "",
    timezone: user?.timezone || "UTC",
  });
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      await authAPI.updateProfile(form);
      await refreshUser();
      setMessage("Profile updated successfully");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
          Account information
        </Typography>
        {message && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {message}
          </Alert>
        )}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <Box component="form" onSubmit={handleSave}>
          <Stack spacing={2} sx={{ maxWidth: 480 }}>
            <TextField
              label="Username"
              value={user?.username || ""}
              disabled
              fullWidth
            />
            <TextField
              label="Email"
              value={user?.email || ""}
              disabled
              fullWidth
            />
            <Stack direction="row" spacing={2}>
              <TextField
                label="First name"
                fullWidth
                value={form.first_name}
                onChange={(e) =>
                  setForm((f) => ({ ...f, first_name: e.target.value }))
                }
              />
              <TextField
                label="Last name"
                fullWidth
                value={form.last_name}
                onChange={(e) =>
                  setForm((f) => ({ ...f, last_name: e.target.value }))
                }
              />
            </Stack>
            <TextField
              label="Phone number"
              fullWidth
              value={form.phone_number}
              onChange={(e) =>
                setForm((f) => ({ ...f, phone_number: e.target.value }))
              }
            />
            <Box>
              <Button
                type="submit"
                variant="contained"
                disabled={saving}
                disableElevation
              >
                {saving ? "Saving…" : "Save changes"}
              </Button>
            </Box>
          </Stack>
        </Box>
      </CardContent>
    </Card>
  );
};

const SecurityTab = () => {
  const { user, refreshUser } = useAuth();
  const [pwForm, setPwForm] = useState({ current: "", next: "", confirm: "" });
  const [pwSaving, setPwSaving] = useState(false);
  const [pwMessage, setPwMessage] = useState(null);
  const [pwError, setPwError] = useState(null);

  const [mfaSetup, setMfaSetup] = useState(null);
  const [mfaPassword, setMfaPassword] = useState("");
  const [mfaCode, setMfaCode] = useState("");
  const [mfaError, setMfaError] = useState(null);
  const [mfaSubmitting, setMfaSubmitting] = useState(false);
  const [disableCode, setDisableCode] = useState("");
  const [disableOpen, setDisableOpen] = useState(false);

  const handleChangePassword = async (e) => {
    e.preventDefault();
    setPwError(null);
    setPwMessage(null);
    if (pwForm.next !== pwForm.confirm) {
      setPwError("New passwords do not match");
      return;
    }
    setPwSaving(true);
    try {
      await authAPI.changePassword(pwForm.current, pwForm.next);
      setPwMessage("Password changed successfully");
      setPwForm({ current: "", next: "", confirm: "" });
    } catch (err) {
      setPwError(getErrorMessage(err));
    } finally {
      setPwSaving(false);
    }
  };

  const startMfaSetup = async () => {
    setMfaError(null);
    try {
      const { data } = await authAPI.setupMfa();
      setMfaSetup(data);
    } catch (err) {
      setMfaError(getErrorMessage(err));
    }
  };

  const confirmMfaEnable = async (e) => {
    e.preventDefault();
    setMfaSubmitting(true);
    setMfaError(null);
    try {
      await authAPI.enableMfa(mfaPassword, mfaCode);
      await refreshUser();
      setMfaSetup(null);
      setMfaPassword("");
      setMfaCode("");
    } catch (err) {
      setMfaError(getErrorMessage(err));
    } finally {
      setMfaSubmitting(false);
    }
  };

  const handleDisableMfa = async () => {
    setMfaError(null);
    try {
      await authAPI.disableMfa(disableCode);
      await refreshUser();
      setDisableOpen(false);
      setDisableCode("");
    } catch (err) {
      setMfaError(getErrorMessage(err));
    }
  };

  return (
    <Stack spacing={3}>
      <Card>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 2 }}>
            Change password
          </Typography>
          {pwMessage && (
            <Alert severity="success" sx={{ mb: 2 }}>
              {pwMessage}
            </Alert>
          )}
          {pwError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {pwError}
            </Alert>
          )}
          <Box component="form" onSubmit={handleChangePassword}>
            <Stack spacing={2} sx={{ maxWidth: 420 }}>
              <TextField
                label="Current password"
                type="password"
                required
                fullWidth
                value={pwForm.current}
                onChange={(e) =>
                  setPwForm((f) => ({ ...f, current: e.target.value }))
                }
              />
              <TextField
                label="New password"
                type="password"
                required
                fullWidth
                value={pwForm.next}
                onChange={(e) =>
                  setPwForm((f) => ({ ...f, next: e.target.value }))
                }
                helperText="At least 12 characters, mixed case, digit, and symbol"
              />
              <TextField
                label="Confirm new password"
                type="password"
                required
                fullWidth
                value={pwForm.confirm}
                onChange={(e) =>
                  setPwForm((f) => ({ ...f, confirm: e.target.value }))
                }
              />
              <Box>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={pwSaving}
                  disableElevation
                >
                  {pwSaving ? "Updating…" : "Update password"}
                </Button>
              </Box>
            </Stack>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
            sx={{ mb: 2 }}
          >
            <Typography variant="subtitle1" fontWeight={700}>
              Two-factor authentication
            </Typography>
            <Chip
              label={user?.is_mfa_enabled ? "Enabled" : "Disabled"}
              color={user?.is_mfa_enabled ? "success" : "default"}
              size="small"
            />
          </Stack>

          {mfaError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {mfaError}
            </Alert>
          )}

          {user?.is_mfa_enabled ? (
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Two-factor authentication is protecting your account.
              </Typography>
              <Button
                color="error"
                variant="outlined"
                onClick={() => setDisableOpen(true)}
              >
                Disable 2FA
              </Button>
            </Box>
          ) : mfaSetup ? (
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Scan this QR code with your authenticator app (Google
                Authenticator, Authy, 1Password…), then enter the 6-digit code
                to confirm.
              </Typography>
              {mfaSetup.qr_code_svg && (
                <Box
                  component="img"
                  src={`data:image/png;base64,${mfaSetup.qr_code_svg}`}
                  alt="MFA QR code"
                  sx={{
                    width: 180,
                    height: 180,
                    mb: 2,
                    border: "1px solid",
                    borderColor: "divider",
                    borderRadius: 1,
                    p: 1,
                  }}
                />
              )}
              <Typography
                variant="caption"
                color="text.secondary"
                display="block"
                sx={{ mb: 2 }}
              >
                Manual entry key: <code>{mfaSetup.secret}</code>
              </Typography>
              <Box component="form" onSubmit={confirmMfaEnable}>
                <Stack spacing={2} sx={{ maxWidth: 320 }}>
                  <TextField
                    label="Account password"
                    type="password"
                    required
                    fullWidth
                    value={mfaPassword}
                    onChange={(e) => setMfaPassword(e.target.value)}
                  />
                  <TextField
                    label="6-digit code"
                    required
                    fullWidth
                    inputProps={{ inputMode: "numeric", maxLength: 6 }}
                    value={mfaCode}
                    onChange={(e) => setMfaCode(e.target.value)}
                  />
                  <Stack direction="row" spacing={1}>
                    <Button
                      type="submit"
                      variant="contained"
                      disabled={mfaSubmitting}
                      disableElevation
                    >
                      {mfaSubmitting ? "Confirming…" : "Confirm & enable"}
                    </Button>
                    <Button color="inherit" onClick={() => setMfaSetup(null)}>
                      Cancel
                    </Button>
                  </Stack>
                </Stack>
              </Box>
            </Box>
          ) : (
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Add an extra layer of security to your account with an
                authenticator app.
              </Typography>
              <Button
                variant="contained"
                onClick={startMfaSetup}
                disableElevation
              >
                Set up 2FA
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      <ConfirmDialog
        open={disableOpen}
        title="Disable two-factor authentication?"
        description="This will make your account less secure."
        confirmLabel="Disable"
        destructive
        onConfirm={handleDisableMfa}
        onClose={() => setDisableOpen(false)}
      />
    </Stack>
  );
};

const ApiKeysTab = () => {
  const [keys, setKeys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [keyName, setKeyName] = useState("");
  const [creating, setCreating] = useState(false);
  const [newKey, setNewKey] = useState(null);
  const [revokeTarget, setRevokeTarget] = useState(null);

  const load = async () => {
    setLoading(true);
    try {
      const { data } = await authAPI.listApiKeys();
      setKeys(data || []);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    setCreating(true);
    try {
      const { data } = await authAPI.createApiKey({ name: keyName });
      setNewKey(data.key);
      setKeyName("");
      load();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async () => {
    if (!revokeTarget) return;
    try {
      await authAPI.revokeApiKey(revokeTarget.id);
      setKeys((prev) => prev.filter((k) => k.id !== revokeTarget.id));
      setRevokeTarget(null);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  return (
    <Card>
      <CardContent>
        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="center"
          sx={{ mb: 2 }}
        >
          <Typography variant="subtitle1" fontWeight={700}>
            API keys
          </Typography>
          <Button
            variant="contained"
            size="small"
            onClick={() => setCreateOpen(true)}
            disableElevation
          >
            New API key
          </Button>
        </Stack>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {!loading && keys.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            You don&apos;t have any API keys yet.
          </Typography>
        )}

        <List disablePadding>
          {keys.map((key) => (
            <ListItem
              key={key.id}
              divider
              disableGutters
              secondaryAction={
                <Tooltip title="Revoke">
                  <IconButton edge="end" onClick={() => setRevokeTarget(key)}>
                    <DeleteOutlineIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              }
            >
              <ListItemText
                primary={key.name}
                secondary={`${key.key_preview} · Created ${new Date(key.created_at).toLocaleDateString()}`}
              />
            </ListItem>
          ))}
        </List>
      </CardContent>

      <Dialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle sx={{ fontWeight: 700 }}>
          {newKey ? "API key created" : "New API key"}
        </DialogTitle>
        <DialogContent>
          {newKey ? (
            <Stack spacing={2}>
              <Alert severity="warning">
                Copy this key now - you won&apos;t be able to see it again.
              </Alert>
              <Stack direction="row" spacing={1} alignItems="center">
                <TextField
                  value={newKey}
                  fullWidth
                  InputProps={{
                    readOnly: true,
                    sx: { fontFamily: "monospace" },
                  }}
                />
                <IconButton
                  onClick={() => navigator.clipboard.writeText(newKey)}
                >
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Stack>
            </Stack>
          ) : (
            <Box component="form" id="create-key-form" onSubmit={handleCreate}>
              <TextField
                label="Key name"
                required
                fullWidth
                autoFocus
                sx={{ mt: 1 }}
                value={keyName}
                onChange={(e) => setKeyName(e.target.value)}
                placeholder="e.g. Production server"
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          {newKey ? (
            <Button
              onClick={() => {
                setNewKey(null);
                setCreateOpen(false);
              }}
              variant="contained"
              disableElevation
            >
              Done
            </Button>
          ) : (
            <>
              <Button onClick={() => setCreateOpen(false)} color="inherit">
                Cancel
              </Button>
              <Button
                type="submit"
                form="create-key-form"
                variant="contained"
                disabled={creating}
                disableElevation
              >
                {creating ? "Creating…" : "Create"}
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>

      <ConfirmDialog
        open={Boolean(revokeTarget)}
        title="Revoke API key?"
        description={`"${revokeTarget?.name}" will stop working immediately.`}
        confirmLabel="Revoke"
        destructive
        onConfirm={handleRevoke}
        onClose={() => setRevokeTarget(null)}
      />
    </Card>
  );
};

const Profile = () => {
  const [tab, setTab] = useState(0);

  return (
    <Box>
      <PageHeader
        title="Profile"
        description="Manage your account, security, and API access."
      />

      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        sx={{ mb: 3, borderBottom: "1px solid", borderColor: "divider" }}
      >
        <Tab
          icon={<PersonIcon fontSize="small" />}
          iconPosition="start"
          label="Account"
        />
        <Tab
          icon={<SecurityIcon fontSize="small" />}
          iconPosition="start"
          label="Security"
        />
        <Tab
          icon={<KeyIcon fontSize="small" />}
          iconPosition="start"
          label="API keys"
        />
      </Tabs>

      {tab === 0 && <AccountTab />}
      {tab === 1 && <SecurityTab />}
      {tab === 2 && <ApiKeysTab />}
    </Box>
  );
};

export default Profile;
