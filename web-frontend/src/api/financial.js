import apiClient from "./client";

export const financialAPI = {
  listTransactions: (params) =>
    apiClient.get("/financial/transactions", { params }),
  getTransaction: (id) => apiClient.get(`/financial/transactions/${id}`),
  createTransaction: (payload) =>
    apiClient.post("/financial/transactions", payload),
  approveTransaction: (id) =>
    apiClient.post(`/financial/transactions/${id}/approve`),
  rejectTransaction: (id, reason) =>
    apiClient.post(`/financial/transactions/${id}/reject`, { reason }),
  summary: (params) =>
    apiClient.get("/financial/financial-summary", { params }),
  complianceLimits: () => apiClient.get("/financial/compliance/limits"),
  calculateInterest: (payload) =>
    apiClient.post("/financial/financial/calculate-interest", payload),
  calculateNpv: (payload) =>
    apiClient.post("/financial/financial/calculate-npv", payload),
};

export default financialAPI;
