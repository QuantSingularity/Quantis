import apiClient from "./client";

export const usersAPI = {
  list: (params) => apiClient.get("/users/", { params }),
  get: (id) => apiClient.get(`/users/${id}`),
  update: (id, payload) => apiClient.put(`/users/${id}`, payload),
  remove: (id) => apiClient.delete(`/users/${id}`),

  listRoles: () => apiClient.get("/users/roles"),
  createRole: (payload) => apiClient.post("/users/roles", payload),
  updateRole: (id, payload) => apiClient.put(`/users/roles/${id}`, payload),
  removeRole: (id) => apiClient.delete(`/users/roles/${id}`),

  listPermissions: () => apiClient.get("/users/permissions"),
  createPermission: (payload) => apiClient.post("/users/permissions", payload),
  removePermission: (id) => apiClient.delete(`/users/permissions/${id}`),
};

export default usersAPI;
