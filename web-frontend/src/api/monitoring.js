import apiClient from "./client";

export const monitoringAPI = {
  health: () => apiClient.get("/monitoring/health"),
  stats: () => apiClient.get("/monitoring/stats"),
  auditLogs: (params) => apiClient.get("/monitoring/audit-logs", { params }),
  metrics: (params) => apiClient.get("/monitoring/metrics", { params }),
  predictionAnalytics: (params) =>
    apiClient.get("/monitoring/analytics/predictions", { params }),
  modelAnalytics: (params) =>
    apiClient.get("/monitoring/analytics/models", { params }),
};

export default monitoringAPI;
