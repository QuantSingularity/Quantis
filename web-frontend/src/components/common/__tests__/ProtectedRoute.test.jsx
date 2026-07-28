import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import ProtectedRoute from "../ProtectedRoute";

const mockUseAuth = vi.fn();
vi.mock("../../../context/AuthContext", () => ({
  useAuth: () => mockUseAuth(),
}));

const renderWithAuth = (authState) => {
  mockUseAuth.mockReturnValue(authState);
  return render(
    <MemoryRouter initialEntries={["/app/dashboard"]}>
      <Routes>
        <Route path="/login" element={<div>Login page</div>} />
        <Route element={<ProtectedRoute />}>
          <Route path="/app/dashboard" element={<div>Dashboard content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
};

describe("ProtectedRoute", () => {
  it("redirects unauthenticated users to /login", () => {
    renderWithAuth({ isAuthenticated: false, isLoading: false });
    expect(screen.getByText("Login page")).toBeInTheDocument();
  });

  it("renders the protected content for authenticated users", () => {
    renderWithAuth({ isAuthenticated: true, isLoading: false });
    expect(screen.getByText("Dashboard content")).toBeInTheDocument();
  });

  it("shows a loading state while auth status is being determined", () => {
    renderWithAuth({ isAuthenticated: false, isLoading: true });
    expect(screen.getByText(/checking your session/i)).toBeInTheDocument();
  });
});
