import apiClient from "./client";

// Verified real paths: POST /predictions/predict, POST /predictions/predict/batch,
// GET /predictions/predictions/history, GET /predictions/predictions/stats,
// GET /predictions/predictions/{id}, GET /predictions/models/health.
export const predictionsAPI = {
  predict: (modelId, inputData, options = {}) =>
    apiClient.post("/predictions/predict", {
      model_id: modelId,
      input_data: inputData,
      tags: options.tags || [],
      notes: options.notes,
    }),
  predictBatch: (modelId, inputDataList, tags) =>
    apiClient.post("/predictions/predict/batch", {
      model_id: modelId,
      input_data: inputDataList,
      tags: tags || [],
    }),
  history: (params) =>
    apiClient.get("/predictions/predictions/history", { params }),
  stats: (params) =>
    apiClient.get("/predictions/predictions/stats", { params }),
  get: (id) => apiClient.get(`/predictions/predictions/${id}`),
  modelsHealth: () => apiClient.get("/predictions/models/health"),
  modelHealth: (modelId) =>
    apiClient.get(`/predictions/models/${modelId}/health`),
};

export default predictionsAPI;
