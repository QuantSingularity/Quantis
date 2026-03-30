import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import { Box, Button, Typography, useTheme } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

const NotFound = () => {
  const theme = useTheme();

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        py: 8,
        px: 2,
        height: "80vh",
      }}
      className="fade-in"
    >
      <ErrorOutlineIcon
        sx={{ fontSize: 100, color: theme.palette.text.secondary, mb: 4 }}
      />

      <Typography variant="h2" component="h1" gutterBottom fontWeight="bold">
        404
      </Typography>

      <Typography variant="h4" gutterBottom>
        Page Not Found
      </Typography>

      <Typography
        variant="body1"
        color="text.secondary"
        sx={{ maxWidth: 500, mb: 4 }}
      >
        The page you are looking for might have been removed, had its name
        changed, or is temporarily unavailable.
      </Typography>

      <Button variant="contained" component={RouterLink} to="/" size="large">
        Back to Dashboard
      </Button>
    </Box>
  );
};

export default NotFound;
