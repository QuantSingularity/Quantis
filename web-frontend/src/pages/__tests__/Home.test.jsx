import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { ThemeModeProvider } from "../../context/ThemeModeContext";
import Home from "../Home";

const renderHome = () =>
  render(
    <ThemeModeProvider>
      <MemoryRouter>
        <Home />
      </MemoryRouter>
    </ThemeModeProvider>,
  );

describe("Home (landing) page", () => {
  it("renders the hero heading and primary CTAs", () => {
    renderHome();
    expect(
      screen.getByRole("heading", {
        name: /financial forecasting, from raw data/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /get started free/i }),
    ).toHaveAttribute("href", "/register");
    expect(screen.getByRole("link", { name: /^sign in$/i })).toHaveAttribute(
      "href",
      "/login",
    );
  });

  it("lists the platform features", () => {
    renderHome();
    expect(screen.getByText(/unified data pipeline/i)).toBeInTheDocument();
    expect(screen.getByText(/multiple model families/i)).toBeInTheDocument();
    expect(screen.getByText(/real-time predictions/i)).toBeInTheDocument();
  });
});
