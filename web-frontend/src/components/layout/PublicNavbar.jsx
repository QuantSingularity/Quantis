import DarkModeIcon from "@mui/icons-material/DarkModeOutlined";
import LightModeIcon from "@mui/icons-material/LightModeOutlined";
import MenuIcon from "@mui/icons-material/Menu";
import {
  AppBar,
  Box,
  Button,
  Drawer,
  IconButton,
  Stack,
  Toolbar,
} from "@mui/material";
import { useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import { useThemeMode } from "../../context/ThemeModeContext";
import Logo from "./Logo";

const NAV_LINKS = [
  { label: "Product", href: "#product" },
  { label: "How it works", href: "#how-it-works" },
  { label: "Security", href: "#security" },
];

const PublicNavbar = () => {
  const { mode, toggleMode } = useThemeMode();
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <AppBar
      position="sticky"
      elevation={0}
      color="transparent"
      sx={{ backdropFilter: "blur(10px)" }}
    >
      <Toolbar sx={{ maxWidth: 1200, width: "100%", mx: "auto", py: 1 }}>
        <Logo />
        <Box sx={{ flexGrow: 1 }} />

        <Stack
          direction="row"
          spacing={3}
          alignItems="center"
          sx={{ display: { xs: "none", md: "flex" } }}
        >
          {NAV_LINKS.map((link) => (
            <Button
              key={link.label}
              href={link.href}
              color="inherit"
              sx={{ fontWeight: 500 }}
            >
              {link.label}
            </Button>
          ))}
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center" sx={{ ml: 2 }}>
          <IconButton
            onClick={toggleMode}
            size="small"
            aria-label="Toggle color mode"
          >
            {mode === "dark" ? (
              <LightModeIcon fontSize="small" />
            ) : (
              <DarkModeIcon fontSize="small" />
            )}
          </IconButton>
          <Box sx={{ display: { xs: "none", sm: "flex" }, gap: 1 }}>
            <Button component={RouterLink} to="/login" color="inherit">
              Sign in
            </Button>
            <Button
              component={RouterLink}
              to="/register"
              variant="contained"
              disableElevation
            >
              Get started
            </Button>
          </Box>
          <IconButton
            sx={{ display: { xs: "flex", sm: "none" } }}
            onClick={() => setDrawerOpen(true)}
            aria-label="Open menu"
          >
            <MenuIcon />
          </IconButton>
        </Stack>
      </Toolbar>

      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      >
        <Stack spacing={1} sx={{ p: 3, width: 240 }}>
          {NAV_LINKS.map((link) => (
            <Button
              key={link.label}
              href={link.href}
              color="inherit"
              sx={{ justifyContent: "flex-start" }}
            >
              {link.label}
            </Button>
          ))}
          <Button
            component={RouterLink}
            to="/login"
            sx={{ justifyContent: "flex-start" }}
          >
            Sign in
          </Button>
          <Button
            component={RouterLink}
            to="/register"
            variant="contained"
            disableElevation
          >
            Get started
          </Button>
        </Stack>
      </Drawer>
    </AppBar>
  );
};

export default PublicNavbar;
