import { beforeEach, describe, expect, it } from "vitest";
import { getErrorMessage, tokenStorage } from "../client";

describe("tokenStorage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("reports not authenticated when no token is stored", () => {
    expect(tokenStorage.isAuthenticated()).toBe(false);
  });

  it("stores and retrieves access/refresh tokens", () => {
    tokenStorage.setTokens("access-123", "refresh-456");
    expect(tokenStorage.getAccessToken()).toBe("access-123");
    expect(tokenStorage.getRefreshToken()).toBe("refresh-456");
    expect(tokenStorage.isAuthenticated()).toBe(true);
  });

  it("clears tokens", () => {
    tokenStorage.setTokens("a", "b");
    tokenStorage.clear();
    expect(tokenStorage.getAccessToken()).toBeNull();
    expect(tokenStorage.getRefreshToken()).toBeNull();
    expect(tokenStorage.isAuthenticated()).toBe(false);
  });
});

describe("getErrorMessage", () => {
  it("extracts a string detail from a FastAPI error response", () => {
    const error = { response: { data: { detail: "Invalid credentials" } } };
    expect(getErrorMessage(error)).toBe("Invalid credentials");
  });

  it("joins a FastAPI validation error array into one message", () => {
    const error = {
      response: {
        data: {
          detail: [
            { msg: "field required" },
            { msg: "value is not a valid email" },
          ],
        },
      },
    };
    expect(getErrorMessage(error)).toBe(
      "field required · value is not a valid email",
    );
  });

  it("falls back to a generic message when nothing is available", () => {
    expect(getErrorMessage({})).toBe("Something went wrong. Please try again.");
  });
});
