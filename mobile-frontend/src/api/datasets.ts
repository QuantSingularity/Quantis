import apiClient from "./client";
import { Dataset } from "./types";

export const datasetsAPI = {
  list: (params?: Record<string, unknown>) =>
    apiClient.get<Dataset[]>("/datasets/", { params }),
  get: (id: number | string) => apiClient.get<Dataset>(`/datasets/${id}`),
  remove: (id: number | string) => apiClient.delete(`/datasets/${id}`),
  stats: (id: number | string) => apiClient.get(`/datasets/${id}/stats`),
  preview: (id: number | string, rows = 15) =>
    apiClient.get(`/datasets/${id}/preview`, { params: { rows } }),
  upload: (
    fileUri: string,
    fileName: string,
    mimeType: string,
    metadata: { name: string; description?: string; frequency?: string },
  ) => {
    const form = new FormData();
    // React Native's FormData accepts this object shape for file uploads.
    form.append("file", {
      uri: fileUri,
      name: fileName,
      type: mimeType,
    } as unknown as Blob);
    form.append("name", metadata.name);
    if (metadata.description) form.append("description", metadata.description);
    if (metadata.frequency) form.append("frequency", metadata.frequency);
    return apiClient.post("/datasets/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
};

export default datasetsAPI;
