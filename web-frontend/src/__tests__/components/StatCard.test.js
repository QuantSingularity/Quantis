import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import StatCard from "../../components/StatCard";

const theme = createTheme();

const renderWithTheme = (component) => {
  return render(<ThemeProvider theme={theme}>{component}</ThemeProvider>);
};

describe("StatCard Component", () => {
  test("renders with correct title and value", () => {
    renderWithTheme(
      <StatCard
        title="Accuracy"
        value="92.7%"
        change="+2.1%"
        positive={true}
        icon={<TrendingUpIcon />}
        color="#1a73e8"
      />,
    );

    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByText("92.7%")).toBeInTheDocument();
    expect(screen.getByText("+2.1%")).toBeInTheDocument();
  });

  test("displays positive change correctly", () => {
    renderWithTheme(
      <StatCard
        title="Test Metric"
        value="100"
        change="+5%"
        positive={true}
        icon={<TrendingUpIcon />}
        color="#34a853"
      />,
    );

    expect(screen.getByText("+5%")).toBeInTheDocument();
  });

  test("displays negative change correctly", () => {
    renderWithTheme(
      <StatCard
        title="Error Rate"
        value="2.3%"
        change="-1.2%"
        positive={false}
        icon={<TrendingUpIcon />}
        color="#ea4335"
      />,
    );

    expect(screen.getByText("-1.2%")).toBeInTheDocument();
  });
});
