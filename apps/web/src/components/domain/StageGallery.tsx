"use client";

import clsx from "clsx";
import { CheckCircle2, Loader2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { mediaUrl } from "@/lib/api";
import { humanize, ms, timeOnly } from "@/lib/format";
import type { Stage, TimelineEvent } from "@/lib/types";

export const STAGE_PLAN: { index: number; name: string; title: string; caption: string }[] = [
  { index: 1, name: "ingest", title: "Ingest", caption: "Validate, apply EXIF orientation, strip metadata" },
  { index: 2, name: "canonical", title: "Canonical", caption: "Long side ≤ 1024 px, JPEG q95 — same path as training" },
  { index: 3, name: "letterbox", title: "Letterbox 224", caption: "Aspect preserved; lavender = padding, no crop" },
  { index: 4, name: "tensor", title: "Tensor", caption: "float32 NCHW in [0,1], R | G | B planes" },
  { index: 5, name: "quality", title: "Quality screen", caption: "Brightness, contrast, blur — low quality escalates" },
  { index: 6, name: "inference", title: "YOLO inference", caption: "Raw softmax → temperature-calibrated p(defective)" },
  { index: 7, name: "attribution", title: "Attribution", caption: "Occlusion heatmap — not a defect location" },
  { index: 8, name: "classified", title: "Classified", caption: "Final triage decision, human clearance required" },
];

/** Reveal newly arrived stages one by one (~350 ms apart) so the progression is legible. */
export function useStagedReveal(count: number, stepMs = 350): number {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (shown >= count) return;
    const t = setTimeout(() => setShown((s) => s + 1), shown === 0 ? 0 : stepMs);
    return () => clearTimeout(t);
  }, [count, shown, stepMs]);
  return Math.min(shown, count);
}

function DataPanel({ stage }: { stage: Stage }) {
  const p = stage.params as Record<string, unknown>;
  if (stage.name === "quality") {
    const d = (p.descriptors ?? {}) as Record<string, number>;
    const rows: [string, number | undefined, number][] = [
      ["Brightness", d.brightness, 1], ["Contrast", d.contrast, 0.5], ["Saturation", d.saturation, 1],
    ];
    return (
      <div className="flex h-full flex-col justify-center gap-2 p-3">
        {rows.map(([k, v, max]) => (
          <div key={k}>
            <div className="flex justify-between text-sm"><span className="text-muted">{k}</span>
              <span className="mono">{v === undefined ? "n/a" : v.toFixed(3)}</span></div>
            <div className="h-2 rounded-full bg-line">
              <div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(1, (v ?? 0) / max) * 100}%` }} />
            </div>
          </div>
        ))}
        <p className="text-sm"><span className="text-muted">Blur variance </span>
          <span className="mono">{d.blur_variance === undefined ? "n/a" : d.blur_variance.toFixed(1)}</span></p>
        <p className={clsx("text-sm font-bold", p.status === "accepted" ? "text-good" : "text-warn")}>
          {humanize(String(p.status ?? ""))}</p>
      </div>
    );
  }
  if (stage.name === "inference") {
    const raw = (p.raw_probs ?? {}) as Record<string, number>;
    const cal = (p.calibrated_probs ?? {}) as Record<string, number>;
    return (
      <div className="flex h-full flex-col justify-center gap-2 p-3">
        {[["raw defective", raw.defective, "bg-primary/50"], ["calibrated defective", cal.defective, "bg-bad"],
          ["calibrated good", cal.good, "bg-good"]].map(([k, v, c]) => (
          <div key={String(k)}>
            <div className="flex justify-between text-sm"><span className="text-muted">{String(k)}</span>
              <span className="mono">{typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "n/a"}</span></div>
            <div className="h-2 rounded-full bg-line">
              <div className={clsx("h-full rounded-full", String(c))} style={{ width: `${(typeof v === "number" ? v : 0) * 100}%` }} />
            </div>
          </div>
        ))}
        <p className="mono text-sm text-muted">T = {typeof p.temperature === "number" ? p.temperature.toFixed(3) : "n/a"}</p>
      </div>
    );
  }
  return <pre className="mono overflow-auto p-3 text-xs text-ink-soft">{JSON.stringify(p, null, 1)}</pre>;
}

export function StageGallery({ stages, running, animate = true }: { stages: Stage[]; running: boolean; animate?: boolean }) {
  const ordered = useMemo(() => [...stages].sort((a, b) => a.index - b.index), [stages]);
  const shown = useStagedReveal(ordered.length, animate ? 350 : 0);
  const visible = animate ? ordered.slice(0, shown) : ordered;
  const byIndex = new Map(visible.map((s) => [s.index, s]));
  return (
    <ol className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Preprocessing and classification stages">
      {STAGE_PLAN.map((plan) => {
        const s = byIndex.get(plan.index);
        const pending = !s;
        const url = s ? mediaUrl(s.image_url) : null;
        return (
          <li key={plan.index} data-stage={plan.name}
            className={clsx("card flex flex-col overflow-hidden", s && "stage-in", pending && "opacity-60")}>
            <div className="relative aspect-square bg-surface-muted">
              {s && url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={url} alt={`Stage ${plan.index}: ${s.title}`} className="h-full w-full object-contain" />
              ) : s ? (
                <DataPanel stage={s} />
              ) : (
                <div className="flex h-full items-center justify-center text-muted">
                  {running ? <Loader2 aria-hidden className="h-6 w-6 animate-spin" /> : <span className="text-sm">not run</span>}
                </div>
              )}
              <span className="absolute left-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-ink/80 text-sm font-bold text-white">
                {plan.index}
              </span>
            </div>
            <div className="flex flex-1 flex-col gap-0.5 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="font-bold text-ink">{plan.title}</span>
                {s ? <span className="mono text-sm text-accent-ink">{ms(s.duration_ms)}</span>
                  : <span className="text-sm text-muted">{running ? "waiting" : ""}</span>}
              </div>
              <span className="text-sm text-muted">{s?.title && s.title !== plan.title ? s.title : plan.caption}</span>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

/** Monospace log of the real pipeline: job transitions and every stage's own log lines with recorded timings. */
export function PipelineConsole({ stages, timeline, status, error }: {
  stages: Stage[];
  timeline?: TimelineEvent[];
  status: string;
  error?: string | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const lines = useMemo(() => {
    const out: { key: string; text: string; tone: string }[] = [];
    for (const e of timeline ?? []) {
      out.push({ key: `t-${e.at}-${e.to}`, text: `${timeOnly(e.at)}  job ${e.from ?? "∅"} → ${e.to}${e.reason ? `  (${e.reason})` : ""}`, tone: "text-[#7fb2ff]" });
    }
    let cumulative = 0;
    for (const s of [...stages].sort((a, b) => a.index - b.index)) {
      cumulative += s.duration_ms;
      out.push({ key: `s-${s.index}`, text: `[stage ${s.index}] ${s.title}  ${ms(s.duration_ms)}  (Σ ${ms(cumulative)})`, tone: "font-bold text-[#b9b0ff]" });
      for (const [i, l] of s.log.entries()) out.push({ key: `s-${s.index}-${i}`, text: `    ${l}`, tone: "text-white/75" });
    }
    if (error) out.push({ key: "err", text: `error: ${error}`, tone: "text-[#ff8a9a]" });
    return out;
  }, [stages, timeline, error]);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [lines.length]);
  const active = status === "queued" || status === "running";
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between border-b border-line bg-ink px-4 py-2 text-white">
        <span className="mono text-sm">pipeline.log</span>
        <span className="flex items-center gap-2 text-sm" aria-live="polite">
          {active ? <Loader2 aria-hidden className="h-4 w-4 animate-spin" /> : <CheckCircle2 aria-hidden className="h-4 w-4 text-accent" />}
          {humanize(status)}
        </span>
      </div>
      <div ref={ref} className="mono max-h-80 overflow-auto bg-[#151827] p-4 text-[13px] leading-relaxed">
        {lines.length === 0 ? (
          <p className="text-white/60">Waiting for the worker to pick up the job…</p>
        ) : (
          lines.map((l) => (
            <p key={l.key} className={clsx("whitespace-pre-wrap", l.tone)}>{l.text}</p>
          ))
        )}
      </div>
    </div>
  );
}
