import apiClient from "./client";

export const notificationsAPI = {
  list: (params) => apiClient.get("/notifications/", { params }),
  get: (id) => apiClient.get(`/notifications/${id}`),
  markRead: (id) => apiClient.patch(`/notifications/${id}/read`),
  markAllRead: () => apiClient.post("/notifications/mark-all-read"),
  remove: (id) => apiClient.delete(`/notifications/${id}`),
};

export default notificationsAPI;
