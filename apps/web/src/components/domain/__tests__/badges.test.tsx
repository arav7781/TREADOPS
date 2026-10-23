import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { GateStatus, SourceBadge, TriageBadge } from "@/components/domain/badges";

describe("SourceBadge", () => {
  it("labels simulated metadata prominently", () => {
    render(<SourceBadge origin="simulated" />);
    expect(screen.getByText("SIMULATED METADATA")).toBeInTheDocument();
  });
  it("distinguishes uploads and the public dataset", () => {
    const { rerender } = render(<SourceBadge origin="user_provided" />);
    expect(screen.getByText("User upload")).toBeInTheDocument();
    rerender(<SourceBadge origin="observed" />);
    expect(screen.getByText("Public dataset")).toBeInTheDocument();
  });
  it("renders nothing for unknown origins", () => {
    const { container } = render(<SourceBadge origin={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("GateStatus", () => {
  it.each([
    ["passed", "Passed"],
    ["failed", "Failed"],
    ["insufficient_evidence", "Insufficient evidence"],
    ["not_applicable", "Not applicable"],
  ])("renders %s as %s", (status, label) => {
    render(<GateStatus status={status} label="Recall" />);
    expect(screen.getByText(new RegExp(`Recall: ${label}`))).toBeInTheDocument();
  });
});

describe("TriageBadge", () => {
  it("uses inspector-facing wording", () => {
    render(<TriageBadge triage="good_candidate" />);
    expect(screen.getByText("Retread candidate")).toBeInTheDocument();
  });
  it("shows pending when no prediction exists", () => {
    render(<TriageBadge triage={null} />);
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });
});
