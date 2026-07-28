import { createTheme } from "@mui/material/styles";
import { tokens } from "./tokens";

/**
 * Builds an MUI theme for the given mode ('dark' | 'light') from the shared
 * design tokens. Keeping this a pure factory function (rather than two
 * static theme objects) means both modes always stay derived from the same
 * source of truth.
 */
export const createAppTheme = (mode = "dark") => {
  const palette = tokens[mode];

  return createTheme({
    palette: {
      mode,
      primary: {
        main: palette.primary,
        light: palette.primaryHover,
        contrastText: "#FFFFFF",
      },
      secondary: {
        main: palette.secondary,
        contrastText: "#FFFFFF",
      },
      success: { main: palette.success },
      warning: { main: palette.warning },
      error: { main: palette.error },
      info: { main: palette.info },
      background: {
        default: palette.background,
        paper: palette.surface,
      },
      text: {
        primary: palette.textPrimary,
        secondary: palette.textSecondary,
      },
      divider: palette.border,
    },
    shape: {
      borderRadius: tokens.radius.sm,
    },
    typography: {
      fontFamily: tokens.fontFamily,
      h1: { fontWeight: 800, letterSpacing: "-0.02em" },
      h2: { fontWeight: 800, letterSpacing: "-0.02em" },
      h3: { fontWeight: 700, letterSpacing: "-0.01em" },
      h4: { fontWeight: 700 },
      h5: { fontWeight: 600 },
      h6: { fontWeight: 600 },
      subtitle1: { fontWeight: 500 },
      button: { fontWeight: 600, textTransform: "none" },
    },
    shadows: Array(25).fill(
      mode === "dark"
        ? "0px 4px 24px rgba(0, 0, 0, 0.35)"
        : "0px 2px 12px rgba(15, 23, 42, 0.08)",
    ),
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: palette.background,
          },
          "::selection": {
            backgroundColor: palette.primary,
            color: "#fff",
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
            border: `1px solid ${palette.border}`,
          },
          rounded: {
            borderRadius: tokens.radius.md,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: tokens.radius.md,
            border: `1px solid ${palette.border}`,
            backgroundColor: palette.surface,
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: tokens.radius.sm,
            paddingInline: 18,
            paddingBlock: 9,
          },
          containedPrimary: {
            boxShadow: "none",
            "&:hover": { boxShadow: "none" },
          },
        },
      },
      MuiTextField: {
        defaultProps: { size: "small" },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: tokens.radius.sm,
            fontWeight: 600,
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderColor: palette.border,
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: palette.surface,
            borderRight: `1px solid ${palette.border}`,
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: palette.surface,
            backgroundImage: "none",
            borderBottom: `1px solid ${palette.border}`,
          },
        },
      },
    },
  });
};

export default createAppTheme;
