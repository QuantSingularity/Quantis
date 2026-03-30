import {
  CloudUpload,
  Dashboard as DashboardIcon,
  Logout,
  ModelTraining,
  Psychology,
  Storage,
  TrendingUp,
} from "@mui/icons-material";
import {
  Avatar,
  Box,
  Divider,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  useTheme,
} from "@mui/material";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const drawerWidth = 240;

const Sidebar = ({ open, onClose }) => {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const menuItems = [
    {
      text: "Dashboard",
      icon: <DashboardIcon />,
      path: "/",
      section: "main",
    },
    {
      text: "Predictions",
      icon: <TrendingUp />,
      path: "/predictions",
      section: "main",
    },
    {
      text: "Models",
      icon: <Psychology />,
      path: "/models",
      section: "main",
    },
    {
      text: "Model Management",
      icon: <ModelTraining />,
      path: "/model-management",
      section: "management",
    },
    {
      text: "Datasets",
      icon: <Storage />,
      path: "/datasets",
      section: "management",
    },
    {
      text: "Upload Dataset",
      icon: <CloudUpload />,
      path: "/dataset-upload",
      section: "management",
    },
  ];

  const handleNavigation = (path) => {
    navigate(path);
    if (onClose) {
      onClose();
    }
  };

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  const isSelected = (path) => {
    if (path === "/") {
      return location.pathname === "/" || location.pathname === "/dashboard";
    }
    return location.pathname === path;
  };

  const renderMenuSection = (sectionItems, title) => (
    <>
      {title && (
        <Typography
          variant="overline"
          sx={{
            px: 2,
            py: 1,
            color: theme.palette.text.secondary,
            fontSize: "0.75rem",
            fontWeight: 600,
          }}
        >
          {title}
        </Typography>
      )}
      <List sx={{ py: 0 }}>
        {sectionItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={isSelected(item.path)}
              sx={{
                mx: 1,
                borderRadius: 1,
                "&.Mui-selected": {
                  backgroundColor: theme.palette.primary.main,
                  color: theme.palette.primary.contrastText,
                  "&:hover": {
                    backgroundColor: theme.palette.primary.dark,
                  },
                  "& .MuiListItemIcon-root": {
                    color: theme.palette.primary.contrastText,
                  },
                },
                "&:hover": {
                  backgroundColor: theme.palette.action.hover,
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color: isSelected(item.path)
                    ? theme.palette.primary.contrastText
                    : theme.palette.text.secondary,
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: "0.875rem",
                  fontWeight: isSelected(item.path) ? 600 : 400,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </>
  );

  const drawerContent = (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Typography
          variant="h5"
          sx={{
            fontWeight: 700,
            color: theme.palette.primary.main,
            textAlign: "center",
          }}
        >
          Quantis
        </Typography>
      </Box>

      {/* User Info */}
      {user && (
        <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Avatar
              sx={{
                bgcolor: theme.palette.primary.main,
                width: 40,
                height: 40,
              }}
            >
              {user.username?.charAt(0).toUpperCase() || "U"}
            </Avatar>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }} noWrap>
                {user.username || "User"}
              </Typography>
              <Typography variant="caption" color="text.secondary" noWrap>
                {user.email || ""}
              </Typography>
            </Box>
          </Box>
        </Box>
      )}

      {/* Navigation */}
      <Box sx={{ flex: 1, overflow: "auto", py: 1 }}>
        {renderMenuSection(
          menuItems.filter((item) => item.section === "main"),
          null,
        )}

        <Divider sx={{ my: 1, mx: 2 }} />

        {renderMenuSection(
          menuItems.filter((item) => item.section === "management"),
          "Management",
        )}
      </Box>

      {/* Footer */}
      <Box sx={{ borderTop: `1px solid ${theme.palette.divider}` }}>
        <List sx={{ py: 0 }}>
          <ListItem disablePadding>
            <ListItemButton
              onClick={handleLogout}
              sx={{
                mx: 1,
                borderRadius: 1,
                "&:hover": {
                  backgroundColor: theme.palette.error.light,
                  color: theme.palette.error.contrastText,
                  "& .MuiListItemIcon-root": {
                    color: theme.palette.error.contrastText,
                  },
                },
              }}
            >
              <ListItemIcon>
                <Logout />
              </ListItemIcon>
              <ListItemText
                primary="Logout"
                primaryTypographyProps={{
                  fontSize: "0.875rem",
                }}
              />
            </ListItemButton>
          </ListItem>
        </List>
      </Box>
    </Box>
  );

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: drawerWidth,
          boxSizing: "border-box",
          borderRight: `1px solid ${theme.palette.divider}`,
        },
      }}
    >
      {drawerContent}
    </Drawer>
  );
};

export default Sidebar;
