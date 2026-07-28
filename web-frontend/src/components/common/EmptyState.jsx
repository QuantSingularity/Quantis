import { Box, Button, Typography } from "@mui/material";

const EmptyState = ({ icon, title, description, actionLabel, onAction }) => (
  <Box
    sx={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      textAlign: "center",
      py: 8,
      px: 3,
      color: "text.secondary",
    }}
  >
    {icon && (
      <Box
        sx={{ fontSize: 48, mb: 2, color: "text.disabled", display: "flex" }}
      >
        {icon}
      </Box>
    )}
    <Typography variant="h6" sx={{ color: "text.primary", mb: 0.5 }}>
      {title}
    </Typography>
    {description && (
      <Typography
        variant="body2"
        sx={{ maxWidth: 420, mb: actionLabel ? 3 : 0 }}
      >
        {description}
      </Typography>
    )}
    {actionLabel && onAction && (
      <Button variant="contained" onClick={onAction} sx={{ mt: 1 }}>
        {actionLabel}
      </Button>
    )}
  </Box>
);

export default EmptyState;
