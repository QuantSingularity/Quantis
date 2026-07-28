import apiClient from "./client";

// NOTE: the backend mounts the models router at prefix "/models" while the
// router's own routes are also named "/models", so the real paths are
// "/models/models", "/models/models/{id}", etc. (verified against the live
// OpenAPI schema).
export const modelsAPI = {
  list: (params) => apiClient.get("/models/models", { params }),
  get: (id) => apiClient.get(`/models/models/${id}`),
  create: (payload) => apiClient.post("/models/models", payload),
  update: (id, payload) => apiClient.put(`/models/models/${id}`, payload),
  remove: (id) => apiClient.delete(`/models/models/${id}`),
  train: (id, payload) =>
    apiClient.post(`/models/models/${id}/train`, payload || {}),
  trainingStatus: (id) => apiClient.get(`/models/models/${id}/training-status`),
  metrics: (id) => apiClient.get(`/models/models/${id}/metrics`),
  types: () => apiClient.get("/models/models/types"),
  compare: (modelIds) =>
    apiClient.get("/models/models/compare", {
      params: { model_ids: modelIds },
    }),
};

export default modelsAPI;
