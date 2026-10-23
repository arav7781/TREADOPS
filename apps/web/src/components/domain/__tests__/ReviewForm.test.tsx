import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ReviewForm } from "@/components/domain/ReviewForm";
import { renderWithQuery } from "@/test/render";

function respond(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

describe("ReviewForm", () => {
  it("sends expected_revision and reports the saved revision", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(respond(201, { inspection_id: "i1", revision: 1, label: "defective", disposition: "material_recovery" }));
    const onSaved = vi.fn();
    renderWithQuery(<ReviewForm inspectionId="i1" expectedRevision={0} onSaved={onSaved} suggestedDisposition="material_recovery" />);
    await userEvent.click(screen.getByText("Defective"));
    await userEvent.type(screen.getByLabelText("Reason (required)"), "bead damage visible");
    await userEvent.click(screen.getByRole("button", { name: "Save review" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalled());
    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body).toEqual({ expected_revision: 0, label: "defective", reason: "bead damage visible", disposition: "material_recovery" });
    expect(screen.getByText(/Saved revision r1/)).toBeInTheDocument();
  });

  it("on 409 shows the other reviewer's label and resubmits on top of the current revision", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(respond(409, { error: { code: "stale_revision", message: "Another reviewer changed this label; reload and reconcile.", request_id: "r", retryable: false,
        details: { expected_revision: 0, current_revision: 1, latest: { label: "good", disposition: "retread", reason: "looks fine" } } } }))
      .mockResolvedValueOnce(respond(201, { inspection_id: "i1", revision: 2, label: "defective", disposition: null }));
    renderWithQuery(<ReviewForm inspectionId="i1" expectedRevision={0} />);
    await userEvent.click(screen.getByText("Defective"));
    await userEvent.type(screen.getByLabelText("Reason (required)"), "crack at shoulder");
    await userEvent.click(screen.getByRole("button", { name: "Save review" }));

    const alert = await screen.findByText(/Another reviewer saved revision r1 first/);
    expect(alert).toBeInTheDocument();
    expect(screen.getByText(/looks fine/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save review" })).toBeDisabled(); // no silent overwrite

    await userEvent.click(screen.getByRole("button", { name: /Keep mine on top of r1/ }));
    await userEvent.click(screen.getByRole("button", { name: "Save review" }));
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(2));
    expect(JSON.parse((fetchSpy.mock.calls[1][1] as RequestInit).body as string).expected_revision).toBe(1);
    expect(await screen.findByText(/Saved revision r2/)).toBeInTheDocument();
  });

  it("requires a label and a reason", async () => {
    renderWithQuery(<ReviewForm inspectionId="i1" expectedRevision={0} />);
    expect(screen.getByRole("button", { name: "Save review" })).toBeDisabled();
  });
});
