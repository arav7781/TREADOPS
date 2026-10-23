import { act, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PipelineConsole, StageGallery } from "@/components/domain/StageGallery";
import type { Stage } from "@/lib/types";

const stage = (index: number, name: string, ms: number, image = true): Stage => ({
  index, name, title: `${name} title`, duration_ms: ms, params: {}, log: [`${name} log line`], image_url: image ? `/v1/media/${name}` : null,
});

afterEach(() => vi.useRealTimers());

describe("StageGallery", () => {
  it("orders stages by index regardless of arrival order and shows recorded timings", () => {
    render(<StageGallery stages={[stage(3, "letterbox", 2.5), stage(1, "ingest", 40), stage(2, "canonical", 120)]} running={false} animate={false} />);
    const items = within(screen.getByRole("list", { name: /stages/i })).getAllByRole("listitem");
    expect(items).toHaveLength(8); // the full plan is always shown
    expect(items[0]).toHaveAttribute("data-stage", "ingest");
    expect(within(items[0]).getByRole("img")).toHaveAttribute("src", "/api/v1/media/ingest");
    expect(within(items[1]).getByText("120 ms")).toBeInTheDocument();
    expect(within(items[2]).getByText("2.5 ms")).toBeInTheDocument();
    expect(within(items[3]).queryByRole("img")).toBeNull(); // tensor stage not produced yet
  });

  it("reveals newly arrived stages one at a time", () => {
    vi.useFakeTimers();
    render(<StageGallery stages={[stage(1, "ingest", 1), stage(2, "canonical", 1), stage(3, "letterbox", 1)]} running />);
    act(() => { vi.advanceTimersByTime(0); });
    expect(screen.getAllByRole("img")).toHaveLength(1);
    act(() => { vi.advanceTimersByTime(350); });
    expect(screen.getAllByRole("img")).toHaveLength(2);
    act(() => { vi.advanceTimersByTime(350); });
    expect(screen.getAllByRole("img")).toHaveLength(3);
  });
});

describe("PipelineConsole", () => {
  it("prints job transitions and each stage's log with cumulative time", () => {
    render(<PipelineConsole status="succeeded" stages={[stage(2, "canonical", 100), stage(1, "ingest", 50)]}
      timeline={[{ at: "2026-10-22T10:00:00Z", from: "queued", to: "running", attempt: 1, reason: "claimed" }]} />);
    expect(screen.getByText(/job queued → running/)).toBeInTheDocument();
    expect(screen.getByText(/\[stage 1\] ingest title 50 ms \(Σ 50 ms\)/)).toBeInTheDocument();
    expect(screen.getByText(/\[stage 2\] canonical title 100 ms \(Σ 150 ms\)/)).toBeInTheDocument();
  });
});
