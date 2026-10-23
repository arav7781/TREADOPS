import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ProbabilityPanel } from "@/components/domain/ProbabilityPanel";

describe("ProbabilityPanel", () => {
  it("shows calibrated and raw probabilities from the prediction", () => {
    render(<ProbabilityPanel raw={{ defective: 0.76, good: 0.24 }} calibrated={{ defective: 0.71, good: 0.29 }}
      pDefective={0.71} thresholds={{ low: 0.2, high: 0.8 }} temperature={1.37} />);
    expect(screen.getByText("71.00%")).toBeInTheDocument();
    expect(screen.getByText("76.00%")).toBeInTheDocument();
    expect(screen.getByText("24.00%")).toBeInTheDocument();
    expect(screen.getByText(/T=1.370/)).toBeInTheDocument();
    expect(screen.getByText(/Markers show triage thresholds \(0.2 \/ 0.8\)/)).toBeInTheDocument();
  });
  it("never invents a number when there is no prediction", () => {
    render(<ProbabilityPanel raw={null} calibrated={null} pDefective={null} />);
    expect(screen.getByText("No prediction yet.")).toBeInTheDocument();
    expect(screen.queryByText(/%/)).toBeNull();
  });
});
