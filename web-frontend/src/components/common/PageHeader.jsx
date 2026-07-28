import { Box, Stack, Typography } from "@mui/material";

const PageHeader = ({ title, description, action, breadcrumb }) => (
  <Stack
    direction={{ xs: "column", sm: "row" }}
    justifyContent="space-between"
    alignItems={{ xs: "flex-start", sm: "center" }}
    spacing={2}
    sx={{ mb: 3 }}
  >
    <Box>
      {breadcrumb && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mb: 0.5 }}
        >
          {breadcrumb}
        </Typography>
      )}
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        {title}
      </Typography>
      {description && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
          {description}
        </Typography>
      )}
    </Box>
    {action && <Box>{action}</Box>}
  </Stack>
);

export default PageHeader;
