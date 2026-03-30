import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import App from "../App";

// Mock the AuthContext
jest.mock("../context/AuthContext", () => ({
  AuthProvider: ({ children }) => <div>{children}</div>,
  useAuth: () => ({
    isAuthenticated: false,
    isLoading: false,
    user: null,
    login: jest.fn(),
    logout: jest.fn(),
    register: jest.fn(),
  }),
}));

describe("App Component", () => {
  test("renders without crashing", () => {
    render(<App />);
  });

  test("shows login page for unauthenticated users", () => {
    render(<App />);
    expect(screen.getByText(/sign in to your account/i)).toBeInTheDocument();
  });
});
