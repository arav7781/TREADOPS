"use client";

import clsx from "clsx";
import { ImageUp, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Button, ErrorState, inputClass, ProgressBar } from "@/components/ui";
import { newId, uploadWithProgress } from "@/lib/api";
import type { InspectionAccepted } from "@/lib/types";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];
const CATEGORIES = [
  ["truck_bus", "Truck / bus"], ["light_truck", "Light truck"], ["passenger", "Passenger"],
  ["two_wheeler", "Two-wheeler"], ["unknown", "Unknown"],
] as const;

/**
 * One photo -> POST /v1/inspections (multipart). A fresh Idempotency-Key is minted per selected file and
 * reused for retries of that same file, so a double click or a network retry can never create two jobs.
 */
export function InspectionUploader({ disabled, disabledReason, onAccepted }: {
  disabled?: boolean;
  disabledReason?: string;
  onAccepted: (a: InspectionAccepted) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dims, setDims] = useState<string | null>(null);
  const [category, setCategory] = useState("truck_bus");
  const [notes, setNotes] = useState("");
  const [key, setKey] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const pick = (f: File | undefined | null) => {
    setError(null);
    if (!f) return;
    if (!ACCEPT.includes(f.type)) {
      setError(new Error(`Unsupported file type ${f.type || "(unknown)"}; use JPEG, PNG or WebP.`));
      return;
    }
    setDims(null);
    setFile(f);
    setKey(newId());
  };

  const upload = async () => {
    if (!file || !key) return;
    setError(null);
    setProgress(0);
    const form = new FormData();
    form.set("file", file);
    form.set("source", "upload");
    form.set("metadata_origin", "user_provided");
    form.set("tyre_category", category);
    if (notes.trim()) form.set("notes", notes.trim());
    try {
      const accepted = await uploadWithProgress<InspectionAccepted>("/api/v1/inspections", form, {
        idempotencyKey: key, onProgress: setProgress,
      });
      onAccepted(accepted);
      setFile(null);
      setKey(null);
      setNotes("");
      if (input.current) input.current.value = "";
    } catch (e) {
      setError(e);
    } finally {
      setProgress(null);
    }
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); if (!disabled) pick(e.dataTransfer.files?.[0]); }}
        className={clsx("relative flex min-h-56 flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-6 text-center transition",
          dragging ? "border-primary bg-primary-soft" : "border-line-strong bg-surface-muted", disabled && "opacity-60")}>
        {preview ? (
          <div className="flex w-full flex-col items-center gap-3 sm:flex-row sm:text-left">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="Decoded preview of the selected tyre photo" className="max-h-56 rounded-xl border border-line object-contain"
              onLoad={(e) => setDims(`${e.currentTarget.naturalWidth}×${e.currentTarget.naturalHeight}`)} />
            <div className="min-w-0 flex-1 space-y-1">
              <p className="truncate font-bold text-ink">{file?.name}</p>
              <p className="text-ink-soft">{dims ?? "decoding…"} · {((file?.size ?? 0) / 1024 / 1024).toFixed(2)} MB · {file?.type}</p>
              <p className="mono text-sm text-muted">Idempotency-Key {key?.slice(0, 8)}…</p>
              <button type="button" className="inline-flex items-center gap-1 text-sm font-semibold text-bad hover:underline"
                onClick={() => { setFile(null); setKey(null); if (input.current) input.current.value = ""; }}>
                <X aria-hidden className="h-4 w-4" /> Remove
              </button>
            </div>
          </div>
        ) : (
          <>
            <ImageUp aria-hidden className="h-10 w-10 text-primary" />
            <p className="text-lg font-bold text-ink">Drop a tyre photo here</p>
            <p className="text-ink-soft">JPEG, PNG or WebP · full frame is analysed (no centre crop) · metadata stripped at ingest</p>
          </>
        )}
        <label className={clsx("mt-1", disabled && "pointer-events-none")}>
          <span className="sr-only">Choose a tyre photo</span>
          <input ref={input} type="file" accept={ACCEPT.join(",")} disabled={disabled}
            onChange={(e) => pick(e.target.files?.[0])}
            className="block text-sm text-ink-soft file:mr-3 file:rounded-xl file:border-0 file:bg-primary file:px-4 file:py-2 file:font-semibold file:text-white hover:file:bg-primary-strong" />
        </label>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="flex flex-col gap-1">
          <label htmlFor="tyre-cat" className="text-sm font-semibold text-ink-soft">Tyre category</label>
          <select id="tyre-cat" className={inputClass} value={category} onChange={(e) => setCategory(e.target.value)}>
            {CATEGORIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="tyre-notes" className="text-sm font-semibold text-ink-soft">Notes (optional)</label>
          <input id="tyre-notes" className={inputClass} value={notes} maxLength={1000} onChange={(e) => setNotes(e.target.value)} placeholder="e.g. casing from fleet intake bay 2" />
        </div>
      </div>
      {progress !== null && (
        <div aria-live="polite">
          <ProgressBar value={progress} label="Upload progress" />
          <p className="mt-1 text-sm text-muted">Uploading {(progress * 100).toFixed(0)}% — then validation, sanitising and queueing…</p>
        </div>
      )}
      {error ? <ErrorState error={error} title="Upload rejected" onRetry={file ? upload : undefined} /> : null}
      {disabled && disabledReason && <p className="font-semibold text-warn">{disabledReason}</p>}
      <Button onClick={upload} disabled={!file || disabled} busy={progress !== null} className="w-full sm:w-auto">
        Inspect tyre
      </Button>
    </div>
  );
}
