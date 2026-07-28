import AsyncStorage from "@react-native-async-storage/async-storage";
import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";
import Constants from "expo-constants";

export const API_BASE_URL: string =
  process.env.EXPO_PUBLIC_API_URL ||
  (Constants.expoConfig?.extra?.apiUrl as string | undefined) ||
  "http://localhost:8000";

const ACCESS_TOKEN_KEY = "quantis_access_token";
const REFRESH_TOKEN_KEY = "quantis_refresh_token";

export const tokenStorage = {
  getAccessToken: () => AsyncStorage.getItem(ACCESS_TOKEN_KEY),
  getRefreshToken: () => AsyncStorage.getItem(REFRESH_TOKEN_KEY),
  setTokens: async (
    accessToken?: string | null,
    refreshToken?: string | null,
  ) => {
    if (accessToken) await AsyncStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
    if (refreshToken)
      await AsyncStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  },
  clear: async () => {
    await AsyncStorage.multiRemove([ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY]);
  },
};

type SessionExpiredListener = () => void;
let sessionExpiredListeners: SessionExpiredListener[] = [];

export const onSessionExpired = (listener: SessionExpiredListener) => {
  sessionExpiredListeners.push(listener);
  return () => {
    sessionExpiredListeners = sessionExpiredListeners.filter(
      (l) => l !== listener,
    );
  };
};

const emitSessionExpired = () => sessionExpiredListeners.forEach((l) => l());

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 30000,
});

apiClient.interceptors.request.use(
  async (config: InternalAxiosRequestConfig) => {
    const token = await tokenStorage.getAccessToken();
    if (token) {
      config.headers.set("Authorization", `Bearer ${token}`);
    }
    return config;
  },
);

let refreshPromise: Promise<string> | null = null;

const performRefresh = async (): Promise<string> => {
  const refreshToken = await tokenStorage.getRefreshToken();
  if (!refreshToken) throw new Error("No refresh token available");
  const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
    refresh_token: refreshToken,
  });
  await tokenStorage.setTokens(
    response.data.access_token,
    response.data.refresh_token,
  );
  return response.data.access_token;
};

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as
      (InternalAxiosRequestConfig & { _retry?: boolean }) | undefined;
    const status = error.response?.status;
    const refreshToken = await tokenStorage.getRefreshToken();

    if (
      status === 401 &&
      originalRequest &&
      !originalRequest._retry &&
      !originalRequest.url?.includes("/auth/login") &&
      !originalRequest.url?.includes("/auth/refresh") &&
      refreshToken
    ) {
      originalRequest._retry = true;
      try {
        if (!refreshPromise) {
          refreshPromise = performRefresh().finally(() => {
            refreshPromise = null;
          });
        }
        const newAccessToken = await refreshPromise;
        originalRequest.headers.set(
          "Authorization",
          `Bearer ${newAccessToken}`,
        );
        return apiClient(originalRequest);
      } catch (refreshError) {
        await tokenStorage.clear();
        emitSessionExpired();
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  },
);

/** Extracts a human-readable message from a FastAPI/axios error. */
export const getErrorMessage = (error: unknown): string => {
  const axiosError = error as AxiosError<{ detail?: unknown }>;
  const detail = axiosError?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (typeof d === "string" ? d : (d as { msg?: string })?.msg))
      .filter(Boolean)
      .join(" · ");
  }
  if (axiosError?.message) return axiosError.message;
  return "Something went wrong. Please try again.";
};

export default apiClient;
