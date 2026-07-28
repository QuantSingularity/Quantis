import apiClient from "./client";

export const monitoringAPI = {
  health: () => apiClient.get("/monitoring/health"),
  stats: () => apiClient.get("/monitoring/stats"),
  auditLogs: (params?: Record<string, unknown>) =>
    apiClient.get("/monitoring/audit-logs", { params }),
};

export default monitoringAPI;
