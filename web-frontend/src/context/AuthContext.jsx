import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { authAPI, getErrorMessage, tokenStorage } from "../api";

const AuthContext = createContext(null);

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  const loadCurrentUser = useCallback(async () => {
    if (!tokenStorage.isAuthenticated()) {
      setIsLoading(false);
      return;
    }
    try {
      const { data } = await authAPI.getCurrentUser();
      setUser(data);
    } catch (error) {
      tokenStorage.clear();
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCurrentUser();
    const onSessionExpired = () => {
      setUser(null);
      setAuthError("Your session expired. Please sign in again.");
    };
    window.addEventListener("quantis:session-expired", onSessionExpired);
    return () =>
      window.removeEventListener("quantis:session-expired", onSessionExpired);
  }, [loadCurrentUser]);

  const login = useCallback(async (username, password, mfaCode) => {
    setAuthError(null);
    try {
      await authAPI.login(username, password, mfaCode);
      const { data } = await authAPI.getCurrentUser();
      setUser(data);
      return { success: true };
    } catch (error) {
      const message = getErrorMessage(error);
      setAuthError(message);
      return {
        success: false,
        error: message,
        status: error?.response?.status,
      };
    }
  }, []);

  const register = useCallback(async (payload) => {
    setAuthError(null);
    try {
      await authAPI.register(payload);
      return { success: true };
    } catch (error) {
      const message = getErrorMessage(error);
      setAuthError(message);
      return { success: false, error: message };
    }
  }, []);

  const logout = useCallback(async () => {
    await authAPI.logout();
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    try {
      const { data } = await authAPI.getCurrentUser();
      setUser(data);
    } catch (error) {
      // Ignore - interceptor will handle 401 -> session expiry
    }
  }, []);

  const isAdmin = user?.role === "admin";

  const value = useMemo(
    () => ({
      user,
      isLoading,
      isAuthenticated: Boolean(user),
      isAdmin,
      authError,
      setAuthError,
      login,
      register,
      logout,
      refreshUser,
    }),
    [user, isLoading, isAdmin, authError, login, register, logout, refreshUser],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;
