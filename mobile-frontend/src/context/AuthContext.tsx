import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  authAPI,
  getErrorMessage,
  onSessionExpired,
  tokenStorage,
} from "../api";
import { User } from "../api/types";

interface LoginResult {
  success: boolean;
  error?: string;
  status?: number;
}

interface AuthContextValue {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  authError: string | null;
  setAuthError: (msg: string | null) => void;
  login: (
    username: string,
    password: string,
    mfaCode?: string,
  ) => Promise<LoginResult>;
  register: (payload: Record<string, unknown>) => Promise<LoginResult>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export const useAuth = (): AuthContextValue => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [authError, setAuthError] = useState<string | null>(null);

  const loadCurrentUser = useCallback(async () => {
    const token = await tokenStorage.getAccessToken();
    if (!token) {
      setIsLoading(false);
      return;
    }
    try {
      const { data } = await authAPI.getCurrentUser();
      setUser(data);
    } catch {
      await tokenStorage.clear();
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCurrentUser();
    const unsubscribe = onSessionExpired(() => {
      setUser(null);
      setAuthError("Your session expired. Please sign in again.");
    });
    return unsubscribe;
  }, [loadCurrentUser]);

  const login = useCallback(
    async (
      username: string,
      password: string,
      mfaCode?: string,
    ): Promise<LoginResult> => {
      setAuthError(null);
      try {
        await authAPI.login(username, password, mfaCode);
        const { data } = await authAPI.getCurrentUser();
        setUser(data);
        return { success: true };
      } catch (error) {
        const message = getErrorMessage(error);
        setAuthError(message);
        const status = (error as { response?: { status?: number } })?.response
          ?.status;
        return { success: false, error: message, status };
      }
    },
    [],
  );

  const register = useCallback(
    async (payload: Record<string, unknown>): Promise<LoginResult> => {
      setAuthError(null);
      try {
        await authAPI.register(payload as never);
        return { success: true };
      } catch (error) {
        const message = getErrorMessage(error);
        setAuthError(message);
        return { success: false, error: message };
      }
    },
    [],
  );

  const logout = useCallback(async () => {
    await authAPI.logout();
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    try {
      const { data } = await authAPI.getCurrentUser();
      setUser(data);
    } catch {
      // Interceptor handles 401 -> session expiry
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
