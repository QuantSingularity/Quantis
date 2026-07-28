import apiClient from "./client";
import { Model } from "./types";

// Verified real paths (backend mounts "/models" prefix over routes that are
// themselves named "/models", producing "/models/models/...").
export const modelsAPI = {
  list: (params?: Record<string, unknown>) =>
    apiClient.get<Model[]>("/models/models", { params }),
  get: (id: number | string) => apiClient.get<Model>(`/models/models/${id}`),
  create: (payload: {
    name: string;
    description?: string;
    model_type: string;
    dataset_id: number;
  }) => apiClient.post<Model>("/models/models", payload),
  remove: (id: number | string) => apiClient.delete(`/models/models/${id}`),
  train: (id: number | string) =>
    apiClient.post(`/models/models/${id}/train`, {}),
  metrics: (id: number | string) =>
    apiClient.get(`/models/models/${id}/metrics`),
};

export default modelsAPI;
