import { Box, Typography } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

const Logo = ({ to = "/", size = 28, showWordmark = true }) => (
  <Box
    component={RouterLink}
    to={to}
    sx={{
      display: "inline-flex",
      alignItems: "center",
      gap: 1.2,
      textDecoration: "none",
      color: "inherit",
    }}
  >
    <Box
      sx={{
        width: size,
        height: size,
        borderRadius: `${size * 0.28}px`,
        background: "linear-gradient(135deg, #6366F1 0%, #14B8A6 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      <svg
        width={size * 0.6}
        height={size * 0.6}
        viewBox="0 0 32 32"
        fill="none"
      >
        <path
          d="M8 21L13 12L17 18L24 8"
          stroke="white"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="24" cy="8" r="2.4" fill="white" />
      </svg>
    </Box>
    {showWordmark && (
      <Typography
        variant="h6"
        sx={{ fontWeight: 800, letterSpacing: "-0.02em" }}
      >
        Quantis
      </Typography>
    )}
  </Box>
);

export default Logo;
