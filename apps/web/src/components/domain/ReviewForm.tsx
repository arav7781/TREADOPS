"use client";

import { useMutation } from "@tanstack/react-query";
import { useId, useState } from "react";

import { Badge, Button, inputClass } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { humanize } from "@/lib/format";
import type { ReviewCreated } from "@/lib/types";

export const LABELS = ["good", "defective", "unjudgeable"] as const;
export const DISPOSITIONS = ["retread", "repair", "material_recovery", "reinspect"] as const;

interface Conflict {
  current_revision: number;
  latest?: { label?: string; disposition?: string | null; reason?: string } | null;
}

/**
 * Append-only review. Sends expected_revision; on 409 the operator sees the newer label and explicitly
 * chooses to keep theirs (resubmit on top of it) or accept the other reviewer's - never a silent overwrite.
 */
export function ReviewForm({ inspectionId, expectedRevision, current, onSaved, suggestedDisposition }: {
  inspectionId: string;
  expectedRevision: number;
  current?: { label: string | null; disposition: string | null };
  onSaved?: (r: ReviewCreated) => void;
  suggestedDisposition?: string | null;
}) {
  const id = useId();
  const [label, setLabel] = useState<string>(current?.label ?? "");
  const [disposition, setDisposition] = useState<string>(current?.disposition ?? suggestedDisposition ?? "");
  const [reason, setReason] = useState("");
  const [expected, setExpected] = useState(expectedRevision);
  const [conflict, setConflict] = useState<Conflict | null>(null);
  const [saved, setSaved] = useState<ReviewCreated | null>(null);

  const m = useMutation({
    mutationFn: () =>
      api<ReviewCreated>(`/v1/inspections/${inspectionId}/reviews`, {
        method: "POST",
        json: { expected_revision: expected, label, reason, disposition: disposition || null },
      }),
    onSuccess: (r) => {
      setSaved(r);
      setConflict(null);
      setExpected(r.revision);
      setReason("");
      onSaved?.(r);
    },
    onError: (e) => {
      if (e instanceof ApiError && e.status === 409 && e.code === "stale_revision") {
        setConflict({ current_revision: Number(e.details.current_revision ?? expected), latest: e.details.latest as Conflict["latest"] });
      }
    },
  });

  const valid = Boolean(label) && reason.trim().length > 0;
  const error = m.error instanceof ApiError && !(m.error.status === 409 && m.error.code === "stale_revision") ? m.error : null;

  return (
    <form
      aria-label="Record a verified review"
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (valid) m.mutate();
      }}
    >
      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-ink-soft">Verified label</legend>
        <div className="flex flex-wrap gap-2">
          {LABELS.map((l) => (
            <label key={l} className={`cursor-pointer rounded-xl border px-3 py-2 font-semibold transition ${label === l
              ? "border-primary bg-primary-soft text-primary-ink" : "border-line-strong bg-surface text-ink-soft hover:bg-surface-muted"}`}>
              <input type="radio" name={`${id}-label`} value={l} checked={label === l} onChange={() => setLabel(l)} className="sr-only" />
              {humanize(l)}
            </label>
          ))}
        </div>
      </fieldset>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="flex flex-col gap-1">
          <label htmlFor={`${id}-disp`} className="text-sm font-semibold text-ink-soft">Casing disposition</label>
          <select id={`${id}-disp`} className={inputClass} value={disposition} onChange={(e) => setDisposition(e.target.value)}>
            <option value="">Not decided</option>
            {DISPOSITIONS.map((d) => <option key={d} value={d}>{humanize(d)}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-sm font-semibold text-ink-soft">Revision</span>
          <span className="flex min-h-10 items-center gap-2 text-ink">
            expecting <Badge tone="primary">r{expected}</Badge> → will write r{expected + 1}
          </span>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <label htmlFor={`${id}-reason`} className="text-sm font-semibold text-ink-soft">Reason (required)</label>
        <textarea id={`${id}-reason`} required rows={2} className={inputClass} value={reason}
          onChange={(e) => setReason(e.target.value)} placeholder="What you saw on the casing" maxLength={2000} />
      </div>

      {conflict && (
        <div role="alert" className="rounded-2xl border border-warn/40 bg-warn-soft p-4">
          <p className="font-bold text-ink">Another reviewer saved revision r{conflict.current_revision} first.</p>
          {conflict.latest && (
            <p className="text-ink-soft">
              Their label: <strong>{humanize(conflict.latest.label)}</strong>
              {conflict.latest.disposition ? ` · ${humanize(conflict.latest.disposition)}` : ""}
              {conflict.latest.reason ? ` — “${conflict.latest.reason}”` : ""}
            </p>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            <Button type="button" variant="secondary" onClick={() => {
              setLabel(conflict.latest?.label ?? label);
              setDisposition(conflict.latest?.disposition ?? "");
              setExpected(conflict.current_revision);
              setConflict(null);
            }}>Accept theirs</Button>
            <Button type="button" onClick={() => {
              setExpected(conflict.current_revision);
              setConflict(null);
            }}>Keep mine on top of r{conflict.current_revision}</Button>
          </div>
        </div>
      )}
      {error && (
        <p role="alert" className="text-bad">
          {error.message} <span className="mono text-sm">({error.code}{error.requestId ? ` · ${error.requestId}` : ""})</span>
        </p>
      )}
      {saved && !conflict && (
        <p role="status" className="text-good">Saved revision r{saved.revision}: {humanize(saved.label)}
          {saved.disposition ? ` · ${humanize(saved.disposition)}` : ""}.</p>
      )}
      <Button type="submit" busy={m.isPending} disabled={!valid || Boolean(conflict)}>Save review</Button>
    </form>
  );
}
