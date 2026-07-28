import { SnackbarProvider } from "notistack";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AdminRoute from "./components/common/AdminRoute";
import ProtectedRoute from "./components/common/ProtectedRoute";
import AuthLayout from "./components/layout/AuthLayout";
import DashboardLayout from "./components/layout/DashboardLayout";
import PublicLayout from "./components/layout/PublicLayout";
import { AuthProvider } from "./context/AuthContext";
import { ThemeModeProvider } from "./context/ThemeModeContext";
import AdminSystem from "./pages/AdminSystem";
import AdminUsers from "./pages/AdminUsers";
import Dashboard from "./pages/Dashboard";
import DatasetDetail from "./pages/DatasetDetail";
import Datasets from "./pages/Datasets";
import Financial from "./pages/Financial";
import Forbidden from "./pages/Forbidden";
import ForgotPassword from "./pages/ForgotPassword";
import Home from "./pages/Home";
import Login from "./pages/Login";
import ModelDetail from "./pages/ModelDetail";
import Models from "./pages/Models";
import NotFound from "./pages/NotFound";
import Notifications from "./pages/Notifications";
import Predictions from "./pages/Predictions";
import Profile from "./pages/Profile";
import Register from "./pages/Register";
import ResetPassword from "./pages/ResetPassword";
import Settings from "./pages/Settings";

function App() {
  return (
    <ThemeModeProvider>
      <SnackbarProvider
        maxSnack={3}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      >
        <BrowserRouter>
          <AuthProvider>
            <Routes>
              {/* Public marketing site */}
              <Route element={<PublicLayout />}>
                <Route path="/" element={<Home />} />
              </Route>

              {/* Auth flow */}
              <Route element={<AuthLayout />}>
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/forgot-password" element={<ForgotPassword />} />
                <Route path="/reset-password" element={<ResetPassword />} />
              </Route>

              {/* Authenticated app */}
              <Route element={<ProtectedRoute />}>
                <Route element={<DashboardLayout />}>
                  <Route
                    path="/app"
                    element={<Navigate to="/app/dashboard" replace />}
                  />
                  <Route path="/app/dashboard" element={<Dashboard />} />
                  <Route path="/app/datasets" element={<Datasets />} />
                  <Route
                    path="/app/datasets/:datasetId"
                    element={<DatasetDetail />}
                  />
                  <Route path="/app/models" element={<Models />} />
                  <Route
                    path="/app/models/:modelId"
                    element={<ModelDetail />}
                  />
                  <Route path="/app/predictions" element={<Predictions />} />
                  <Route path="/app/financial" element={<Financial />} />
                  <Route
                    path="/app/notifications"
                    element={<Notifications />}
                  />
                  <Route path="/app/profile" element={<Profile />} />
                  <Route path="/app/settings" element={<Settings />} />

                  {/* Admin-only */}
                  <Route element={<AdminRoute />}>
                    <Route path="/app/admin/users" element={<AdminUsers />} />
                    <Route path="/app/admin/system" element={<AdminSystem />} />
                  </Route>
                </Route>
              </Route>

              <Route path="/403" element={<Forbidden />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </AuthProvider>
        </BrowserRouter>
      </SnackbarProvider>
    </ThemeModeProvider>
  );
}

export default App;
