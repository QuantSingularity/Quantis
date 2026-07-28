/**
 * Design tokens shared conceptually with web-frontend/src/theme/tokens.js.
 * See /DESIGN_SYSTEM.md at the repo root for the source of truth.
 */
export type ThemeMode = "dark" | "light";

export const colorTokens = {
  dark: {
    primary: "#6366F1",
    primaryHover: "#818CF8",
    secondary: "#14B8A6",
    success: "#22C55E",
    warning: "#F59E0B",
    error: "#EF4444",
    info: "#38BDF8",
    background: "#0B0F19",
    surface: "#131826",
    surfaceElevated: "#1B2333",
    border: "#232B3D",
    textPrimary: "#F3F4F6",
    textSecondary: "#9CA3AF",
  },
  light: {
    primary: "#4F46E5",
    primaryHover: "#4338CA",
    secondary: "#0D9488",
    success: "#16A34A",
    warning: "#D97706",
    error: "#DC2626",
    info: "#0284C7",
    background: "#F8FAFC",
    surface: "#FFFFFF",
    surfaceElevated: "#FFFFFF",
    border: "#E2E8F0",
    textPrimary: "#0F172A",
    textSecondary: "#64748B",
  },
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  xxxl: 48,
};

export const statusColor = (
  mode: ThemeMode,
  status?: string | null,
): string => {
  const palette = colorTokens[mode];
  const key = (status || "").toLowerCase();
  const successStates = [
    "trained",
    "deployed",
    "ready",
    "active",
    "compliant",
    "approved",
    "completed",
    "low",
  ];
  const warningStates = [
    "training",
    "processing",
    "uploading",
    "pending",
    "under_review",
    "medium",
  ];
  const errorStates = [
    "failed",
    "error",
    "rejected",
    "non_compliant",
    "locked",
    "suspended",
    "high",
    "critical",
  ];
  if (successStates.includes(key)) return palette.success;
  if (warningStates.includes(key)) return palette.warning;
  if (errorStates.includes(key)) return palette.error;
  return palette.textSecondary;
};
