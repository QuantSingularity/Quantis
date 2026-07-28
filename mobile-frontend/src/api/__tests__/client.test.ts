import AsyncStorage from "@react-native-async-storage/async-storage";
import { getErrorMessage, tokenStorage } from "../client";

describe("tokenStorage", () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  it("returns null when no token is stored", async () => {
    expect(await tokenStorage.getAccessToken()).toBeNull();
    expect(await tokenStorage.getRefreshToken()).toBeNull();
  });

  it("stores and retrieves access/refresh tokens", async () => {
    await tokenStorage.setTokens("access-123", "refresh-456");
    expect(await tokenStorage.getAccessToken()).toBe("access-123");
    expect(await tokenStorage.getRefreshToken()).toBe("refresh-456");
  });

  it("clears tokens", async () => {
    await tokenStorage.setTokens("a", "b");
    await tokenStorage.clear();
    expect(await tokenStorage.getAccessToken()).toBeNull();
    expect(await tokenStorage.getRefreshToken()).toBeNull();
  });

  it("does not overwrite an existing token when passed null/undefined", async () => {
    await tokenStorage.setTokens("access-1", "refresh-1");
    await tokenStorage.setTokens(undefined, undefined);
    expect(await tokenStorage.getAccessToken()).toBe("access-1");
    expect(await tokenStorage.getRefreshToken()).toBe("refresh-1");
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
