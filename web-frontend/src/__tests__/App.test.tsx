import { configureStore } from "@reduxjs/toolkit";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { ErrorBoundary } from "react-error-boundary";
import { Provider } from "react-redux";
import App from "../App";
import rootReducer from "../store/rootReducer";

const createTestStore = () => {
  return configureStore({
    reducer: rootReducer,
    preloadedState: {
      auth: {
        isAuthenticated: false,
        user: null,
        loading: false,
        error: null,
      },
      predictions: {
        data: [],
        loading: false,
        error: null,
      },
    },
  });
};

describe("App Component", () => {
  let store;

  beforeEach(() => {
    store = createTestStore();
  });

  it("renders login page when not authenticated", () => {
    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );
    expect(screen.getByText(/Sign In/i)).toBeInTheDocument();
  });

  it("renders dashboard when authenticated", () => {
    store = configureStore({
      reducer: rootReducer,
      preloadedState: {
        auth: {
          isAuthenticated: true,
          user: { id: 1, email: "test@example.com" },
          loading: false,
          error: null,
        },
        predictions: {
          data: [],
          loading: false,
          error: null,
        },
      },
    });

    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );
    expect(screen.getByText(/Dashboard/i)).toBeInTheDocument();
  });

  it("handles login form submission", async () => {
    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );

    fireEvent.change(screen.getByLabelText(/Email/i), {
      target: { value: "test@example.com" },
    });
    fireEvent.change(screen.getByLabelText(/Password/i), {
      target: { value: "password123" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    await waitFor(() => {
      expect(store.getState().auth.loading).toBe(true);
    });
  });

  it("displays error message on failed login", async () => {
    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );

    fireEvent.change(screen.getByLabelText(/Email/i), {
      target: { value: "invalid@example.com" },
    });
    fireEvent.change(screen.getByLabelText(/Password/i), {
      target: { value: "wrongpassword" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    await waitFor(() => {
      expect(screen.getByText(/Invalid credentials/i)).toBeInTheDocument();
    });
  });

  it("validates form inputs", async () => {
    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );

    // Try to submit without filling required fields
    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    await waitFor(() => {
      expect(screen.getByText(/Email is required/i)).toBeInTheDocument();
      expect(screen.getByText(/Password is required/i)).toBeInTheDocument();
    });

    // Try invalid email format
    fireEvent.change(screen.getByLabelText(/Email/i), {
      target: { value: "invalid-email" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    await waitFor(() => {
      expect(screen.getByText(/Invalid email format/i)).toBeInTheDocument();
    });
  });

  it("shows loading state during authentication", async () => {
    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );

    fireEvent.change(screen.getByLabelText(/Email/i), {
      target: { value: "test@example.com" },
    });
    fireEvent.change(screen.getByLabelText(/Password/i), {
      target: { value: "password123" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    expect(screen.getByText(/Loading/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Sign In/i })).toBeDisabled();
  });

  it("handles error boundary", () => {
    const ThrowError = () => {
      throw new Error("Test error");
    };

    render(
      <Provider store={store}>
        <ErrorBoundary fallback={<div>Error occurred</div>}>
          <ThrowError />
        </ErrorBoundary>
      </Provider>,
    );

    expect(screen.getByText(/Error occurred/i)).toBeInTheDocument();
  });

  it("cleans up on unmount", () => {
    const { unmount } = render(
      <Provider store={store}>
        <App />
      </Provider>,
    );

    // Simulate some state changes
    act(() => {
      store.dispatch({ type: "SET_LOADING", payload: true });
    });

    // Unmount component
    unmount();

    // Verify store is cleaned up
    expect(store.getState().auth.loading).toBe(false);
  });

  it("handles network errors gracefully", async () => {
    // Mock network error
    global.fetch = jest.fn().mockRejectedValue(new Error("Network error"));

    render(
      <Provider store={store}>
        <App />
      </Provider>,
    );

    fireEvent.change(screen.getByLabelText(/Email/i), {
      target: { value: "test@example.com" },
    });
    fireEvent.change(screen.getByLabelText(/Password/i), {
      target: { value: "password123" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    await waitFor(() => {
      expect(screen.getByText(/Network error/i)).toBeInTheDocument();
    });
  });
});
