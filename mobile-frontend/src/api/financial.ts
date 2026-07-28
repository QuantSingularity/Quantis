import apiClient from "./client";
import { Transaction } from "./types";

export const financialAPI = {
  listTransactions: (params?: Record<string, unknown>) =>
    apiClient.get<Transaction[]>("/financial/transactions", { params }),
  createTransaction: (payload: {
    amount: number;
    transaction_type: string;
    description?: string;
  }) => apiClient.post<Transaction>("/financial/transactions", payload),
  summary: (params?: Record<string, unknown>) =>
    apiClient.get("/financial/financial-summary", { params }),
};

export default financialAPI;
