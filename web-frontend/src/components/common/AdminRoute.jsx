import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import LoadingScreen from "./LoadingScreen";

const AdminRoute = () => {
  const { isAdmin, isLoading } = useAuth();

  if (isLoading) return <LoadingScreen />;
  if (!isAdmin) return <Navigate to="/403" replace />;

  return <Outlet />;
};

export default AdminRoute;
