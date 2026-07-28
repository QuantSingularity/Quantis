import { Box } from "@mui/material";
import { Outlet } from "react-router-dom";
import PublicFooter from "./PublicFooter";
import PublicNavbar from "./PublicNavbar";

const PublicLayout = () => (
  <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
    <PublicNavbar />
    <Box component="main" sx={{ flexGrow: 1 }}>
      <Outlet />
    </Box>
    <PublicFooter />
  </Box>
);

export default PublicLayout;
