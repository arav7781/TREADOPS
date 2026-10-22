"use client";

import { pct } from "@/lib/format";

export interface ProbabilityPanelProps {
  raw: Record<string, number> | null | undefined;
  calibrated: Record<string, number> | null | undefined;
  pDefective: number | null | undefined;
  thresholds?: { low?: number; high?: number; binary?: number };
  temperature?: number | null;
}

function Bar({ label, value, tone, markers }: { label: string; value: number; tone: string;
  markers?: { at: number; label: string }[] }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
        <span className="font-semibold text-ink-soft">{label}</span>
        <span className="mono font-bold text-ink">{(value * 100).toFixed(2)}%</span>
      </div>
      <div className="relative h-3 overflow-visible rounded-full bg-line">
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
        {markers?.map((m) => (
          <span key={m.label} title={m.label} aria-hidden
            className="absolute -top-1 h-5 w-0.5 bg-ink/60" style={{ left: `${m.at * 100}%` }} />
        ))}
      </div>
    </div>
  );
}

/** Raw softmax vs temperature-calibrated probability, with the versioned triage thresholds marked. */
export function ProbabilityPanel({ raw, calibrated, pDefective, thresholds, temperature }: ProbabilityPanelProps) {
  if (pDefective === null || pDefective === undefined || !raw || !calibrated) {
    return <p className="text-muted">No prediction yet.</p>;
  }
  const markers = [
    thresholds?.low !== undefined ? { at: thresholds.low, label: `triage low ${thresholds.low}` } : null,
    thresholds?.high !== undefined ? { at: thresholds.high, label: `triage high ${thresholds.high}` } : null,
  ].filter(Boolean) as { at: number; label: string }[];
  return (
    <div className="space-y-4" aria-label="Defect probability">
      <Bar label="Calibrated p(defective)" value={calibrated.defective ?? pDefective} tone="bg-gradient-to-r from-primary to-bad" markers={markers} />
      <Bar label="Raw softmax: defective" value={raw.defective ?? 0} tone="bg-primary/50" />
      <Bar label="Raw softmax: good" value={raw.good ?? 0} tone="bg-accent/60" />
      <p className="text-sm text-muted">
        Calibrated with temperature {temperature ? `T=${temperature.toFixed(3)}` : "scaling"} on a dedicated calibration
        cohort.{markers.length ? ` Markers show triage thresholds (${markers.map((m) => m.at).join(" / ")}).` : ""}
        {" "}Summary: p(defective) = {pct(pDefective, 2)}.
      </p>
    </div>
  );
}
