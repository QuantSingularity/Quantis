import AccountBalanceIcon from "@mui/icons-material/AccountBalanceOutlined";
import AdminPanelSettingsIcon from "@mui/icons-material/AdminPanelSettingsOutlined";
import DarkModeIcon from "@mui/icons-material/DarkModeOutlined";
import DashboardIcon from "@mui/icons-material/DashboardOutlined";
import DatasetIcon from "@mui/icons-material/StorageOutlined";
import LightModeIcon from "@mui/icons-material/LightModeOutlined";
import LogoutIcon from "@mui/icons-material/LogoutOutlined";
import MenuIcon from "@mui/icons-material/Menu";
import ModelTrainingIcon from "@mui/icons-material/ModelTrainingOutlined";
import MonitorHeartIcon from "@mui/icons-material/MonitorHeartOutlined";
import NotificationsIcon from "@mui/icons-material/NotificationsOutlined";
import PersonIcon from "@mui/icons-material/PersonOutlined";
import PredictionIcon from "@mui/icons-material/InsightsOutlined";
import SettingsIcon from "@mui/icons-material/SettingsOutlined";
import GroupIcon from "@mui/icons-material/GroupOutlined";
import {
  AppBar,
  Avatar,
  Badge,
  Box,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Toolbar,
  Tooltip,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import {
  Link as RouterLink,
  Outlet,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { notificationsAPI } from "../../api";
import { useAuth } from "../../context/AuthContext";
import { useThemeMode } from "../../context/ThemeModeContext";
import Logo from "./Logo";

const DRAWER_WIDTH = 248;

const NAV_ITEMS = [
  { label: "Dashboard", to: "/app/dashboard", icon: DashboardIcon },
  { label: "Datasets", to: "/app/datasets", icon: DatasetIcon },
  { label: "Models", to: "/app/models", icon: ModelTrainingIcon },
  { label: "Predictions", to: "/app/predictions", icon: PredictionIcon },
  { label: "Financial", to: "/app/financial", icon: AccountBalanceIcon },
];

const ADMIN_NAV_ITEMS = [
  { label: "Users", to: "/app/admin/users", icon: GroupIcon },
  { label: "System health", to: "/app/admin/system", icon: MonitorHeartIcon },
];

const DashboardLayout = () => {
  const { user, isAdmin, logout } = useAuth();
  const { mode, toggleMode } = useThemeMode();
  const location = useLocation();
  const navigate = useNavigate();

  const [mobileOpen, setMobileOpen] = useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const loadUnread = async () => {
      try {
        const { data } = await notificationsAPI.list({
          is_read: false,
          limit: 1,
        });
        if (!cancelled) {
          setUnreadCount(
            Array.isArray(data) ? data.length : (data?.total ?? 0),
          );
        }
      } catch {
        // Non-critical — silently ignore
      }
    };
    loadUnread();
    const interval = setInterval(loadUnread, 60000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [location.pathname]);

  const handleLogout = async () => {
    setUserMenuAnchor(null);
    await logout();
    navigate("/login");
  };

  const initials = (user?.username || "?").slice(0, 2).toUpperCase();

  const renderNavList = (items) => (
    <List sx={{ px: 1.5 }}>
      {items.map((item) => {
        const Icon = item.icon;
        const selected = location.pathname.startsWith(item.to);
        return (
          <ListItemButton
            key={item.to}
            component={RouterLink}
            to={item.to}
            selected={selected}
            onClick={() => setMobileOpen(false)}
            sx={{
              borderRadius: 1.5,
              mb: 0.5,
              "&.Mui-selected": {
                bgcolor: "primary.main",
                color: "#fff",
                "& .MuiListItemIcon-root": { color: "#fff" },
                "&:hover": { bgcolor: "primary.main" },
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 36 }}>
              <Icon fontSize="small" />
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{ fontSize: 14, fontWeight: 500 }}
            />
          </ListItemButton>
        );
      })}
    </List>
  );

  const drawerContent = (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <Toolbar sx={{ px: 2.5, py: 2 }}>
        <Logo />
      </Toolbar>
      <Divider />
      {renderNavList(NAV_ITEMS)}

      {isAdmin && (
        <>
          <Divider sx={{ mx: 2, my: 1 }} />
          <Typography
            variant="caption"
            sx={{
              px: 3,
              color: "text.secondary",
              fontWeight: 600,
              letterSpacing: 0.5,
            }}
          >
            ADMIN
          </Typography>
          {renderNavList(ADMIN_NAV_ITEMS)}
        </>
      )}

      <Box sx={{ flexGrow: 1 }} />
      <Divider />
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Quantis v2.0 · {mode === "dark" ? "Dark" : "Light"} mode
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: "block", md: "none" },
          "& .MuiDrawer-paper": {
            width: DRAWER_WIDTH,
            boxSizing: "border-box",
          },
        }}
      >
        {drawerContent}
      </Drawer>

      {/* Persistent desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: "none", md: "block" },
          width: DRAWER_WIDTH,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: DRAWER_WIDTH,
            boxSizing: "border-box",
          },
        }}
        open
      >
        {drawerContent}
      </Drawer>

      <Box
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
            ml: { md: `${DRAWER_WIDTH}px` },
          }}
        >
          <Toolbar sx={{ gap: 1 }}>
            <IconButton
              edge="start"
              sx={{ display: { xs: "flex", md: "none" } }}
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              <MenuIcon />
            </IconButton>
            <Box sx={{ flexGrow: 1 }} />

            <Tooltip title="Toggle theme">
              <IconButton onClick={toggleMode} size="small">
                {mode === "dark" ? (
                  <LightModeIcon fontSize="small" />
                ) : (
                  <DarkModeIcon fontSize="small" />
                )}
              </IconButton>
            </Tooltip>

            <Tooltip title="Notifications">
              <IconButton
                size="small"
                component={RouterLink}
                to="/app/notifications"
                aria-label="Notifications"
              >
                <Badge badgeContent={unreadCount} color="error" max={99}>
                  <NotificationsIcon fontSize="small" />
                </Badge>
              </IconButton>
            </Tooltip>

            <Tooltip title="Account">
              <IconButton
                onClick={(e) => setUserMenuAnchor(e.currentTarget)}
                size="small"
                sx={{ ml: 0.5 }}
              >
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    fontSize: 13,
                    bgcolor: "primary.main",
                  }}
                >
                  {initials}
                </Avatar>
              </IconButton>
            </Tooltip>

            <Menu
              anchorEl={userMenuAnchor}
              open={Boolean(userMenuAnchor)}
              onClose={() => setUserMenuAnchor(null)}
              transformOrigin={{ horizontal: "right", vertical: "top" }}
              anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
            >
              <Box sx={{ px: 2, py: 1 }}>
                <Typography variant="subtitle2" fontWeight={700}>
                  {user?.username}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {user?.email}
                </Typography>
              </Box>
              <Divider />
              <MenuItem
                component={RouterLink}
                to="/app/profile"
                onClick={() => setUserMenuAnchor(null)}
              >
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                Profile
              </MenuItem>
              <MenuItem
                component={RouterLink}
                to="/app/settings"
                onClick={() => setUserMenuAnchor(null)}
              >
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                Settings
              </MenuItem>
              {isAdmin && (
                <MenuItem
                  component={RouterLink}
                  to="/app/admin/users"
                  onClick={() => setUserMenuAnchor(null)}
                >
                  <ListItemIcon>
                    <AdminPanelSettingsIcon fontSize="small" />
                  </ListItemIcon>
                  Admin console
                </MenuItem>
              )}
              <Divider />
              <MenuItem onClick={handleLogout} sx={{ color: "error.main" }}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" color="error" />
                </ListItemIcon>
                Sign out
              </MenuItem>
            </Menu>
          </Toolbar>
        </AppBar>

        <Box component="main" sx={{ flexGrow: 1, p: { xs: 2, sm: 3, md: 4 } }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
};

export default DashboardLayout;
