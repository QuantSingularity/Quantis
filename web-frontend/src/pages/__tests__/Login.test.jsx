import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { AuthProvider } from "../../context/AuthContext";
import { ThemeModeProvider } from "../../context/ThemeModeContext";
import Login from "../Login";

vi.mock("../../api", async () => {
  const actual = await vi.importActual("../../api");
  return {
    ...actual,
    authAPI: {
      ...actual.authAPI,
      getCurrentUser: vi.fn().mockRejectedValue(new Error("not authed")),
      login: vi.fn(),
    },
    tokenStorage: {
      ...actual.tokenStorage,
      isAuthenticated: () => false,
    },
  };
});

const renderLogin = () =>
  render(
    <ThemeModeProvider>
      <MemoryRouter initialEntries={["/login"]}>
        <AuthProvider>
          <Login />
        </AuthProvider>
      </MemoryRouter>
    </ThemeModeProvider>,
  );

describe("Login page", () => {
  it("renders the sign-in form", async () => {
    renderLogin();
    expect(
      await screen.findByRole("heading", { name: /welcome back/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/username or email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^password/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /sign in/i }),
    ).toBeInTheDocument();
  });

  it("lets the user type into the username and password fields", async () => {
    const user = userEvent.setup();
    renderLogin();
    const usernameField = await screen.findByLabelText(/username or email/i);
    const passwordField = screen.getByLabelText(/^password/i);

    await user.type(usernameField, "testuser");
    await user.type(passwordField, "SecurePass123!");

    expect(usernameField).toHaveValue("testuser");
    expect(passwordField).toHaveValue("SecurePass123!");
  });

  it("links to the register and forgot-password pages", async () => {
    renderLogin();
    expect(
      await screen.findByRole("link", { name: /create one/i }),
    ).toHaveAttribute("href", "/register");
    expect(
      screen.getByRole("link", { name: /forgot password/i }),
    ).toHaveAttribute("href", "/forgot-password");
  });
});
