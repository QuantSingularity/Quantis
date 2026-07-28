import apiClient, { tokenStorage } from "./client";

export const authAPI = {
  register: (payload) => apiClient.post("/auth/register", payload),

  login: async (username, password, mfaCode) => {
    const response = await apiClient.post("/auth/login", {
      username,
      password,
      mfa_code: mfaCode || undefined,
    });
    tokenStorage.setTokens(
      response.data.access_token,
      response.data.refresh_token,
    );
    return response.data;
  },

  logout: async () => {
    const refreshToken = tokenStorage.getRefreshToken();
    try {
      if (refreshToken) {
        await apiClient.post("/auth/logout", { refresh_token: refreshToken });
      }
    } finally {
      tokenStorage.clear();
    }
  },

  getCurrentUser: () => apiClient.get("/auth/me"),

  updateProfile: (payload) => apiClient.put("/auth/me", payload),

  changePassword: (currentPassword, newPassword) =>
    apiClient.post("/auth/change-password", {
      current_password: currentPassword,
      new_password: newPassword,
    }),

  setupMfa: () => apiClient.post("/auth/mfa/setup"),

  enableMfa: (password, otpCode) =>
    apiClient.post("/auth/mfa/enable", { password, otp_code: otpCode }),

  disableMfa: (otpCode) =>
    apiClient.post("/auth/mfa/disable", { otp_code: otpCode }),

  listApiKeys: () => apiClient.get("/auth/api-keys"),

  createApiKey: (payload) => apiClient.post("/auth/api-keys", payload),

  revokeApiKey: (keyId) => apiClient.delete(`/auth/api-keys/${keyId}`),

  forgotPassword: (email) => apiClient.post("/auth/forgot-password", { email }),

  resetPassword: (token, newPassword) =>
    apiClient.post("/auth/reset-password", {
      token,
      new_password: newPassword,
      confirm_password: newPassword,
    }),
};

export default authAPI;
