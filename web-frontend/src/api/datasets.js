import apiClient from "./client";

export const datasetsAPI = {
  list: (params) => apiClient.get("/datasets/", { params }),
  get: (id) => apiClient.get(`/datasets/${id}`),
  create: (payload) => apiClient.post("/datasets/", payload),
  update: (id, payload) => apiClient.put(`/datasets/${id}`, payload),
  remove: (id) => apiClient.delete(`/datasets/${id}`),
  upload: (file, metadata, onUploadProgress) => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", metadata.name);
    if (metadata.description) form.append("description", metadata.description);
    if (metadata.source) form.append("source", metadata.source);
    if (metadata.frequency) form.append("frequency", metadata.frequency);
    (metadata.tags || []).forEach((tag) => form.append("tags", tag));
    return apiClient.post("/datasets/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress,
    });
  },
  stats: (id) => apiClient.get(`/datasets/${id}/stats`),
  preview: (id, rows = 25) =>
    apiClient.get(`/datasets/${id}/preview`, { params: { rows } }),
  downloadUrl: (id) => `${apiClient.defaults.baseURL}/datasets/${id}/download`,
};

export default datasetsAPI;
