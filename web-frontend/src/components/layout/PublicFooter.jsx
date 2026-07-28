import { Box, Container, Divider, Stack, Typography } from "@mui/material";
import Logo from "./Logo";

const PublicFooter = () => (
  <Box
    component="footer"
    sx={{ borderTop: "1px solid", borderColor: "divider", mt: 10 }}
  >
    <Container maxWidth="lg" sx={{ py: 5 }}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        justifyContent="space-between"
        alignItems={{ xs: "flex-start", sm: "center" }}
        spacing={2}
      >
        <Logo size={22} />
        <Typography variant="body2" color="text.secondary">
          © {new Date().getFullYear()} Quantis. Built for quantitative finance
          teams.
        </Typography>
      </Stack>
      <Divider sx={{ my: 3 }} />
      <Typography variant="caption" color="text.secondary">
        Quantis is a forecasting and decision-support tool. It does not
        constitute financial advice.
      </Typography>
    </Container>
  </Box>
);

export default PublicFooter;
