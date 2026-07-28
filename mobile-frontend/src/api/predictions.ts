import apiClient from "./client";
import { Prediction } from "./types";

// Verified real paths: POST /predictions/predict, GET /predictions/predictions/history,
// GET /predictions/predictions/stats, GET /predictions/predictions/{id}.
export const predictionsAPI = {
  predict: (modelId: number, inputData: Record<string, unknown>) =>
    apiClient.post<Prediction>("/predictions/predict", {
      model_id: modelId,
      input_data: inputData,
    }),
  history: (params?: Record<string, unknown>) =>
    apiClient.get<Prediction[]>("/predictions/predictions/history", { params }),
  stats: (params?: Record<string, unknown>) =>
    apiClient.get("/predictions/predictions/stats", { params }),
  get: (id: number | string) =>
    apiClient.get<Prediction>(`/predictions/predictions/${id}`),
};

export default predictionsAPI;
