import { render, screen } from "@testing-library/react-native";
import React from "react";
import { PaperProvider } from "react-native-paper";
import StatusChip from "../StatusChip";

const renderChip = (status?: string | null) =>
  render(
    <PaperProvider>
      <StatusChip status={status} />
    </PaperProvider>,
  );

describe("StatusChip", () => {
  it("renders a formatted label for a snake_case status", () => {
    renderChip("in_progress");
    expect(screen.getByText("In Progress")).toBeTruthy();
  });

  it("falls back to 'Unknown' when status is missing", () => {
    renderChip(undefined);
    expect(screen.getByText("Unknown")).toBeTruthy();
  });

  it("renders trained and failed statuses with their own labels", () => {
    const { rerender } = render(
      <PaperProvider>
        <StatusChip status="trained" />
      </PaperProvider>,
    );
    expect(screen.getByText("Trained")).toBeTruthy();

    rerender(
      <PaperProvider>
        <StatusChip status="failed" />
      </PaperProvider>,
    );
    expect(screen.getByText("Failed")).toBeTruthy();
  });
});
