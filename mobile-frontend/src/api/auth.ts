import apiClient, { tokenStorage } from "./client";
import { ApiKey, TokenResponse, User } from "./types";

export const authAPI = {
  register: (payload: {
    username: string;
    email: string;
    password: string;
    confirm_password: string;
    first_name?: string;
    last_name?: string;
  }) => apiClient.post<User>("/auth/register", payload),

  login: async (username: string, password: string, mfaCode?: string) => {
    const response = await apiClient.post<TokenResponse>("/auth/login", {
      username,
      password,
      mfa_code: mfaCode || undefined,
    });
    await tokenStorage.setTokens(
      response.data.access_token,
      response.data.refresh_token,
    );
    return response.data;
  },

  logout: async () => {
    const refreshToken = await tokenStorage.getRefreshToken();
    try {
      if (refreshToken) {
        await apiClient.post("/auth/logout", { refresh_token: refreshToken });
      }
    } finally {
      await tokenStorage.clear();
    }
  },

  getCurrentUser: () => apiClient.get<User>("/auth/me"),

  updateProfile: (payload: Partial<User>) =>
    apiClient.put<User>("/auth/me", payload),

  changePassword: (currentPassword: string, newPassword: string) =>
    apiClient.post("/auth/change-password", {
      current_password: currentPassword,
      new_password: newPassword,
    }),

  forgotPassword: (email: string) =>
    apiClient.post<{ message: string; reset_token?: string }>(
      "/auth/forgot-password",
      {
        email,
      },
    ),

  resetPassword: (token: string, newPassword: string) =>
    apiClient.post("/auth/reset-password", {
      token,
      new_password: newPassword,
      confirm_password: newPassword,
    }),

  setupMfa: () =>
    apiClient.post<{ qr_code_svg: string; secret: string; message: string }>(
      "/auth/mfa/setup",
    ),

  enableMfa: (password: string, otpCode: string) =>
    apiClient.post<User>("/auth/mfa/enable", { password, otp_code: otpCode }),

  disableMfa: (otpCode: string) =>
    apiClient.post<User>("/auth/mfa/disable", { otp_code: otpCode }),

  listApiKeys: () => apiClient.get<ApiKey[]>("/auth/api-keys"),

  createApiKey: (name: string) =>
    apiClient.post<ApiKey & { key: string }>("/auth/api-keys", { name }),

  revokeApiKey: (keyId: number) => apiClient.delete(`/auth/api-keys/${keyId}`),
};

export default authAPI;
