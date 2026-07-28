import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import StatusChip from "../StatusChip";

describe("StatusChip", () => {
  it("renders a formatted label for a snake_case status", () => {
    render(<StatusChip status="in_progress" />);
    expect(screen.getByText("In Progress")).toBeInTheDocument();
  });

  it("falls back to 'Unknown' when status is missing", () => {
    render(<StatusChip status={undefined} />);
    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });

  it("renders trained/deployed statuses distinctly from failed", () => {
    const { rerender } = render(<StatusChip status="trained" />);
    expect(screen.getByText("Trained")).toBeInTheDocument();
    rerender(<StatusChip status="failed" />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});
