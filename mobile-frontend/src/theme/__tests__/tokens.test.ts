import { statusColor, colorTokens } from "../tokens";

describe("statusColor", () => {
  it("maps success-like statuses to the success color", () => {
    expect(statusColor("dark", "trained")).toBe(colorTokens.dark.success);
    expect(statusColor("dark", "deployed")).toBe(colorTokens.dark.success);
    expect(statusColor("light", "compliant")).toBe(colorTokens.light.success);
  });

  it("maps warning-like statuses to the warning color", () => {
    expect(statusColor("dark", "training")).toBe(colorTokens.dark.warning);
    expect(statusColor("dark", "pending")).toBe(colorTokens.dark.warning);
  });

  it("maps error-like statuses to the error color", () => {
    expect(statusColor("dark", "failed")).toBe(colorTokens.dark.error);
    expect(statusColor("dark", "rejected")).toBe(colorTokens.dark.error);
  });

  it("is case-insensitive", () => {
    expect(statusColor("dark", "TRAINED")).toBe(colorTokens.dark.success);
    expect(statusColor("dark", "Failed")).toBe(colorTokens.dark.error);
  });

  it("falls back to the secondary text color for unknown statuses", () => {
    expect(statusColor("dark", "something_unexpected")).toBe(
      colorTokens.dark.textSecondary,
    );
    expect(statusColor("dark", null)).toBe(colorTokens.dark.textSecondary);
    expect(statusColor("dark", undefined)).toBe(colorTokens.dark.textSecondary);
  });
});
