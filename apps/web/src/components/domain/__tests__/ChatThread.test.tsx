import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, renderHook, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChatThread, Composer, eventText, usePhotoPicker } from "@/components/domain/ChatThread";
import type { ChatEvent, ChatMessage } from "@/lib/types";
import { renderWithQuery } from "@/test/render";

const at = (s: number) => new Date(Date.UTC(2026, 9, 23, 10, 0, s)).toISOString();
const ev = (s: number, e: Omit<ChatEvent, "at">): ChatEvent => ({ at: at(s), ...e });

const user: ChatMessage = {
  id: "u1", role: "user", content: "Can this casing be retreaded?", suggestions: [], trace_id: null, created_at: at(0),
  attachments: { photo: { url: "/v1/media/abc?sig=x", width: 1024, height: 768, bytes: 204800, sha256: "deadbeefcafe", metadata_removed: true, gps_removed: true } },
};
const events: ChatEvent[] = [
  ev(0, { kind: "start", message: "Bib received your message and tyre photo" }),
  ev(0, { kind: "guard_input", status: "passed", rules: [], message: "input guardrails: passed" }),
  ev(1, { kind: "tool_call", tool: "classify_tyre_image", args: "{}" }),
  ev(1, { kind: "stage", tool: "classify_tyre_image", stage: 1, name: "ingest", title: "Ingest", duration_ms: 41, message: "stage 1: Ingest (41 ms)" }),
  ev(2, { kind: "tool_result", tool: "classify_tyre_image", ok: true, summary: "Defect flagged, p(defective) 0.97" }),
];

describe("ChatThread", () => {
  it("shows the sent photo with its sanitisation facts and the agent's live steps", () => {
    const pending: ChatMessage = { id: "a1", role: "assistant", content: "", suggestions: [], trace_id: null, created_at: at(0),
      attachments: { status: "pending", events, has_photo: true, inspection_id: null } };
    renderWithQuery(<ChatThread messages={[user, pending]} onSuggest={() => {}} autoScroll={false} />);
    expect(screen.getByAltText("Tyre photo you sent")).toHaveAttribute("src", "/api/v1/media/abc?sig=x");
    expect(screen.getByText(/1024×768 · 200 KB · sha deadbeef · EXIF stripped · GPS removed/)).toBeInTheDocument();
    const steps = within(screen.getByRole("list", { name: "Agent steps" })).getAllByRole("listitem");
    expect(steps.map((s) => s.getAttribute("data-kind"))).toEqual(["start", "guard_input", "tool_call", "stage", "tool_result"]);
    expect(within(steps[4]).getByText("classify_tyre_image → Defect flagged, p(defective) 0.97")).toBeInTheDocument();
    expect(within(steps[4]).getByText("+2.0s")).toBeInTheDocument();
    expect(screen.getByText("Finished the model inspection")).toBeInTheDocument(); // live status from the latest step
  });

  it("folds the steps away once an answer is ready; a click opens them; follow-ups and copy are offered", async () => {
    const onSuggest = vi.fn();
    const done: ChatMessage = { id: "a2", role: "assistant", content: "**Triage: Defect flagged**", suggestions: ["Show the preprocessing stages"],
      trace_id: "t1", created_at: at(0), attachments: { status: "complete", events, mode: "deterministic_fallback", grounding: { checked: 2, ungrounded: [], pass: true } } };
    renderWithQuery(<ChatThread messages={[done]} onSuggest={onSuggest} autoScroll={false} />);
    expect(screen.queryByRole("list", { name: "Agent steps" })).toBeNull(); // collapsed (hidden from assistive tech)
    await userEvent.click(screen.getByRole("button", { name: /5 steps · 2.0 s/ }));
    expect(screen.getByRole("list", { name: "Agent steps" })).toBeInTheDocument();
    await userEvent.click(within(screen.getByRole("group", { name: "Suggested follow-ups" })).getByText("Show the preprocessing stages"));
    expect(onSuggest).toHaveBeenCalledWith("Show the preprocessing stages");
    expect(screen.getByText("grounded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy answer" })).toBeInTheDocument();
  });

  it("collapses the live steps by itself when the answer completes", () => {
    const base = { id: "a4", role: "assistant" as const, content: "", suggestions: [], trace_id: null, created_at: at(0) };
    const client = new QueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
    const { rerender } = render(<ChatThread messages={[{ ...base, attachments: { status: "pending", events } }]} onSuggest={() => {}} autoScroll={false} />, { wrapper });
    expect(screen.getByRole("list", { name: "Agent steps" })).toBeInTheDocument();
    rerender(<ChatThread messages={[{ ...base, content: "Done.", attachments: { status: "complete", events } }]} onSuggest={() => {}} autoScroll={false} />);
    expect(screen.queryByRole("list", { name: "Agent steps" })).toBeNull();
  });

  it("offers to ask again when an answer failed", async () => {
    const onSuggest = vi.fn();
    const asked: ChatMessage = { ...user, id: "u9", attachments: {} };
    const failed: ChatMessage = { id: "a9", role: "assistant", content: "Something went wrong.", suggestions: [], trace_id: null, created_at: at(1),
      attachments: { status: "error", events: [] } };
    renderWithQuery(<ChatThread messages={[asked, failed]} onSuggest={onSuggest} autoScroll={false} />);
    await userEvent.click(screen.getByRole("button", { name: "Ask again" }));
    expect(onSuggest).toHaveBeenCalledWith("Can this casing be retreaded?");
  });

  it("describes the inspection event with its pinned deployment", () => {
    expect(eventText({ at: at(0), kind: "inspection", inspection_id: "c65b0e94-c703", model: "yolo26s-cls v2", deployment_generation: 3 }))
      .toBe("inspection c65b0e94 created · yolo26s-cls v2 · generation 3");
  });
});

describe("Composer photo attachment", () => {
  const png = new File(["x"], "tyre.png", { type: "image/png" });

  it("attaches a photo and sends it with the message", async () => {
    const onSend = vi.fn();
    const { result } = renderHook(() => usePhotoPicker(null));
    const { rerender } = renderWithQuery(<Composer onSend={onSend} busy={false} photo={result.current} />);
    await userEvent.upload(screen.getByLabelText("Attach a tyre photo"), png);
    rerender(<Composer onSend={onSend} busy={false} photo={result.current} />);
    expect(screen.getByAltText("Selected tyre photo preview")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Message"), "is it retreadable?{Enter}");
    expect(onSend).toHaveBeenCalledWith("is it retreadable?", png, null);
  });

  it("keeps the attach button visible but disabled with the reason for non-inspectors", () => {
    const { result } = renderHook(() => usePhotoPicker("Attaching tyre photos requires the inspector role."));
    renderWithQuery(<Composer onSend={() => {}} busy={false} photo={result.current} />);
    expect(screen.getByLabelText("Attach a tyre photo")).toBeDisabled();
    expect(screen.getByTitle("Attaching tyre photos requires the inspector role.")).toBeInTheDocument();
  });

  it("rejects non-image files with a readable error", async () => {
    const { result } = renderHook(() => usePhotoPicker(null));
    renderWithQuery(<Composer onSend={() => {}} busy={false} photo={result.current} />);
    await userEvent.upload(screen.getByLabelText("Attach a tyre photo"), new File(["x"], "notes.pdf", { type: "application/pdf" }), { applyAccept: false });
    expect(result.current.file).toBeNull();
    expect(result.current.error).toMatch(/Unsupported file type application\/pdf/);
  });
});
