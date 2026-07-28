import DoneAllIcon from "@mui/icons-material/DoneAll";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import NotificationsIcon from "@mui/icons-material/NotificationsOutlined";
import {
  Alert,
  Box,
  Button,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Paper,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import { getErrorMessage, notificationsAPI } from "../api";
import EmptyState from "../components/common/EmptyState";
import LoadingScreen from "../components/common/LoadingScreen";
import PageHeader from "../components/common/PageHeader";

const Notifications = () => {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await notificationsAPI.list({ limit: 50 });
      setNotifications(data?.items || data || []);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleMarkRead = async (id) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, is_read: true } : n)),
    );
    try {
      await notificationsAPI.markRead(id);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const handleMarkAllRead = async () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
    try {
      await notificationsAPI.markAllRead();
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const handleDelete = async (id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
    try {
      await notificationsAPI.remove(id);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  if (loading) return <LoadingScreen label="Loading notifications…" />;

  const unreadCount = notifications.filter((n) => !n.is_read).length;

  return (
    <Box>
      <PageHeader
        title="Notifications"
        description={
          unreadCount > 0 ? `${unreadCount} unread` : "You're all caught up"
        }
        action={
          unreadCount > 0 && (
            <Button
              startIcon={<DoneAllIcon />}
              onClick={handleMarkAllRead}
              color="inherit"
            >
              Mark all as read
            </Button>
          )
        }
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {notifications.length === 0 ? (
        <Paper sx={{ p: 2 }}>
          <EmptyState
            icon={<NotificationsIcon fontSize="inherit" />}
            title="No notifications"
            description="You'll see model training updates, prediction alerts, and account activity here."
          />
        </Paper>
      ) : (
        <Paper>
          <List disablePadding>
            {notifications.map((n, idx) => (
              <ListItem
                key={n.id}
                divider={idx < notifications.length - 1}
                sx={{ bgcolor: n.is_read ? "transparent" : "action.hover" }}
                secondaryAction={
                  <Stack direction="row" spacing={0.5}>
                    {!n.is_read && (
                      <Tooltip title="Mark as read">
                        <IconButton
                          size="small"
                          onClick={() => handleMarkRead(n.id)}
                        >
                          <DoneAllIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        onClick={() => handleDelete(n.id)}
                      >
                        <DeleteOutlineIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                }
              >
                <ListItemText
                  primary={
                    <Typography
                      variant="body2"
                      fontWeight={n.is_read ? 400 : 700}
                    >
                      {n.title || n.notification_type}
                    </Typography>
                  }
                  secondary={
                    <>
                      {n.message}
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        display="block"
                      >
                        {n.created_at
                          ? new Date(n.created_at).toLocaleString()
                          : ""}
                      </Typography>
                    </>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
    </Box>
  );
};

export default Notifications;
