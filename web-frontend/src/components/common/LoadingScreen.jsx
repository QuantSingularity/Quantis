import { Box, CircularProgress, Typography } from "@mui/material";

const LoadingScreen = ({ label = "Loading…" }) => (
  <Box
    sx={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: 2,
      minHeight: "60vh",
      width: "100%",
    }}
  >
    <CircularProgress size={36} thickness={4} />
    <Typography variant="body2" color="text.secondary">
      {label}
    </Typography>
  </Box>
);

export default LoadingScreen;
