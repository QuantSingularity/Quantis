import BlockIcon from "@mui/icons-material/BlockOutlined";
import { Box, Button, Container, Stack, Typography } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

const Forbidden = () => (
  <Container maxWidth="sm" sx={{ py: 12 }}>
    <Stack alignItems="center" spacing={2} textAlign="center">
      <BlockIcon sx={{ fontSize: 64, color: "error.main" }} />
      <Typography variant="h4" fontWeight={700}>
        Access denied
      </Typography>
      <Typography variant="body1" color="text.secondary">
        You don&apos;t have permission to view this page. If you believe this is
        a mistake, contact your workspace administrator.
      </Typography>
      <Box sx={{ pt: 2 }}>
        <Button
          component={RouterLink}
          to="/app/dashboard"
          variant="contained"
          disableElevation
        >
          Back to dashboard
        </Button>
      </Box>
    </Stack>
  </Container>
);

export default Forbidden;
