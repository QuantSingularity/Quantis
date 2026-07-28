import apiClient from "./client";
import { User } from "./types";

export const usersAPI = {
  list: (params?: Record<string, unknown>) =>
    apiClient.get<User[]>("/users/", { params }),
  remove: (id: number | string) => apiClient.delete(`/users/${id}`),
};

export default usersAPI;
