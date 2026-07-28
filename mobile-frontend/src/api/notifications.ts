import apiClient from "./client";
import { Notification } from "./types";

export const notificationsAPI = {
  list: (params?: Record<string, unknown>) =>
    apiClient.get<Notification[]>("/notifications/", { params }),
  markRead: (id: number | string) =>
    apiClient.patch(`/notifications/${id}/read`),
  markAllRead: () => apiClient.post("/notifications/mark-all-read"),
  remove: (id: number | string) => apiClient.delete(`/notifications/${id}`),
};

export default notificationsAPI;
