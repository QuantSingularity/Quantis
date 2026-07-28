import { Box, Container, Paper, Stack } from "@mui/material";
import { Outlet } from "react-router-dom";
import Logo from "./Logo";

/**
 * Centered card layout used for Login / Register / Forgot / Reset password.
 */
const AuthLayout = () => (
  <Box
    sx={{
      minHeight: "100vh",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      background: (theme) =>
        theme.palette.mode === "dark"
          ? "radial-gradient(circle at 20% 20%, rgba(99,102,241,0.14), transparent 40%), radial-gradient(circle at 80% 80%, rgba(20,184,166,0.12), transparent 40%)"
          : "radial-gradient(circle at 20% 20%, rgba(79,70,229,0.06), transparent 40%), radial-gradient(circle at 80% 80%, rgba(13,148,136,0.06), transparent 40%)",
      py: 6,
      px: 2,
    }}
  >
    <Container maxWidth="xs">
      <Stack spacing={4} alignItems="center">
        <Logo size={36} />
        <Paper
          sx={{ p: { xs: 3, sm: 4 }, width: "100%", borderRadius: 3 }}
          elevation={0}
        >
          <Outlet />
        </Paper>
      </Stack>
    </Container>
  </Box>
);

export default AuthLayout;
