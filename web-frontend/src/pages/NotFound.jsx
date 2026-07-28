import { Box, Button, Container, Stack, Typography } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const NotFound = () => {
  const { isAuthenticated } = useAuth();
  return (
    <Container maxWidth="sm" sx={{ py: 12 }}>
      <Stack alignItems="center" spacing={2} textAlign="center">
        <Typography
          variant="h1"
          sx={{ fontSize: 96, fontWeight: 800, color: "primary.main" }}
        >
          404
        </Typography>
        <Typography variant="h5" fontWeight={700}>
          Page not found
        </Typography>
        <Typography variant="body1" color="text.secondary">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </Typography>
        <Box sx={{ pt: 2 }}>
          <Button
            component={RouterLink}
            to={isAuthenticated ? "/app/dashboard" : "/"}
            variant="contained"
            disableElevation
          >
            {isAuthenticated ? "Back to dashboard" : "Back to homepage"}
          </Button>
        </Box>
      </Stack>
    </Container>
  );
};

export default NotFound;
