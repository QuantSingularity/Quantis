import { createContext, useContext, useEffect, useState } from "react";
import {
  authAPI,
  clearAuth,
  isAuthenticated,
  setApiKey,
  setAuthToken,
} from "../services/api";

const AuthContext = createContext({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  login: () => {},
  logout: () => {},
  register: () => {},
});

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check if user is authenticated on app load
  useEffect(() => {
    const checkAuth = async () => {
      if (isAuthenticated()) {
        try {
          const userData = await authAPI.getCurrentUser();
          setUser(userData.data);
        } catch (error) {
          console.error("Failed to get current user:", error);
          clearAuth();
        }
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const login = async (credentials) => {
    try {
      const response = await authAPI.login(credentials);

      if (response.data.access_token) {
        setAuthToken(response.data.access_token);
      } else if (response.data.api_key) {
        setApiKey(response.data.api_key);
      }

      setUser(response.data.user);
      localStorage.setItem("user", JSON.stringify(response.data.user));

      return { success: true, data: response.data };
    } catch (error) {
      console.error("Login failed:", error);
      return {
        success: false,
        error: error.response?.data?.message || "Login failed",
      };
    }
  };

  const register = async (userData) => {
    try {
      const response = await authAPI.register(userData);

      if (response.data.access_token) {
        setAuthToken(response.data.access_token);
      } else if (response.data.api_key) {
        setApiKey(response.data.api_key);
      }

      setUser(response.data.user);
      localStorage.setItem("user", JSON.stringify(response.data.user));

      return { success: true, data: response.data };
    } catch (error) {
      console.error("Registration failed:", error);
      return {
        success: false,
        error: error.response?.data?.message || "Registration failed",
      };
    }
  };

  const logout = async () => {
    try {
      await authAPI.logout();
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      clearAuth();
      setUser(null);
    }
  };

  const value = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    logout,
    register,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
