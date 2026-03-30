import {
  clearAuth,
  handleApiError,
  isAuthenticated,
  setApiKey,
  setAuthToken,
} from "../../services/api";

describe("API Service", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  test("setAuthToken stores token in localStorage", () => {
    const token = "test-token-123";
    setAuthToken(token);
    expect(localStorage.getItem("authToken")).toBe(token);
  });

  test("setApiKey stores API key in localStorage", () => {
    const apiKey = "test-api-key-456";
    setApiKey(apiKey);
    expect(localStorage.getItem("apiKey")).toBe(apiKey);
  });

  test("clearAuth removes all auth data", () => {
    localStorage.setItem("authToken", "token");
    localStorage.setItem("apiKey", "key");
    localStorage.setItem("user", "user-data");

    clearAuth();

    expect(localStorage.getItem("authToken")).toBeNull();
    expect(localStorage.getItem("apiKey")).toBeNull();
    expect(localStorage.getItem("user")).toBeNull();
  });

  test("isAuthenticated returns true when token exists", () => {
    localStorage.setItem("authToken", "token");
    expect(isAuthenticated()).toBe(true);
  });

  test("isAuthenticated returns true when API key exists", () => {
    localStorage.setItem("apiKey", "key");
    expect(isAuthenticated()).toBe(true);
  });

  test("isAuthenticated returns false when no auth data exists", () => {
    expect(isAuthenticated()).toBe(false);
  });

  test("handleApiError processes response error correctly", () => {
    const error = {
      response: {
        status: 400,
        data: { message: "Bad Request" },
      },
    };

    const result = handleApiError(error);
    expect(result.status).toBe(400);
    expect(result.message).toBe("Bad Request");
  });

  test("handleApiError processes network error correctly", () => {
    const error = {
      request: {},
    };

    const result = handleApiError(error);
    expect(result.status).toBe(0);
    expect(result.message).toBe("Network error - please check your connection");
  });
});
