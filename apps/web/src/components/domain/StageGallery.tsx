"use client";

import clsx from "clsx";
import { CheckCircle2, Disc3, Loader2, Radio } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { StageLens, stageChanges, type Thresholds } from "@/components/domain/StageLens";
import { mediaUrl } from "@/lib/api";
import { humanize, ms, timeOnly } from "@/lib/format";
import type { Stage, TimelineEvent } from "@/lib/types";

export const STAGE_PLAN: { index: number; name: string; title: string; short?: string; caption: string }[] = [
  { index: 1, name: "ingest", title: "Ingest", caption: "Validate, apply EXIF orientation, strip metadata" },
  { index: 2, name: "canonical", title: "Canonical", caption: "Long side ≤ 1024 px, JPEG q95 — same path as training" },
  { index: 3, name: "letterbox", title: "Letterbox 224", short: "Letterbox", caption: "Aspect preserved; lavender = padding, no crop" },
  { index: 4, name: "tensor", title: "Tensor", caption: "float32 NCHW in [0,1], R | G | B planes" },
  { index: 5, name: "quality", title: "Quality screen", short: "Quality", caption: "Brightness, contrast, blur — low quality escalates" },
  { index: 6, name: "inference", title: "YOLO inference", short: "YOLO", caption: "Raw softmax → temperature-calibrated p(defective)" },
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
      <div className="flex h-full flex-col justify-center gap-2 p-3 pt-11">
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
      <div className="flex h-full flex-col justify-center gap-2 p-3 pt-11">
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
  return <pre className="mono overflow-auto p-3 pt-11 text-xs text-ink-soft">{JSON.stringify(p, null, 1)}</pre>;
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
              <span className="font-bold leading-tight text-ink">{plan.title}</span>
              {s ? <span className="mono text-sm text-accent-ink">{ms(s.duration_ms)}</span>
                : <span className="text-sm text-muted">{running ? "waiting…" : "—"}</span>}
              <span className="text-sm text-muted">{plan.caption}</span>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

/**
 * The pipeline as a short film: each recorded stage plays in order in a large viewer that shows what that stage
 * changed (before → after, pixel detail, quality chart, probability dial), with its recorded facts and log lines.
 * The filmstrip fills up as stages arrive; any finished stage can be revisited.
 */
export function StageTheatre({ stages, running, stepMs = 1600, onRevealed, thresholds = null, focus = null }: {
  stages: Stage[];
  running: boolean;
  stepMs?: number;
  /** Stage name to open at (e.g. "attribution") instead of the latest one. */
  focus?: string | null;
  /** Called once every recorded stage has been shown and the job is finished. */
  onRevealed?: () => void;
  thresholds?: Thresholds | null;
}) {
  const ordered = useMemo(() => [...stages].sort((a, b) => a.index - b.index), [stages]);
  const allByIndex = useMemo(() => new Map(ordered.map((s) => [s.index, s])), [ordered]);
  const shown = useStagedReveal(ordered.length, stepMs);
  useEffect(() => {
    if (!running && shown >= ordered.length) onRevealed?.();
  }, [running, shown, ordered.length, onRevealed]);
  const visible = ordered.slice(0, shown);
  const [picked, setPicked] = useState<number | null>(() => STAGE_PLAN.find((p) => p.name === focus)?.index ?? null);
  const byIndex = new Map(visible.map((s) => [s.index, s]));
  const latest = visible[visible.length - 1];
  const current = (picked !== null && byIndex.get(picked)) || latest;
  const plan = current ? STAGE_PLAN.find((p) => p.index === current.index) : undefined;
  const live = running || shown < ordered.length;
  const next = live ? STAGE_PLAN.find((p) => !byIndex.has(p.index)) : undefined;
  const lb = allByIndex.get(3);
  return (
    <div className="overflow-hidden rounded-2xl border border-line bg-surface" role="region" aria-label="Preprocessing and classification stages">
      <div className="flex items-center gap-3 px-4 py-2.5 text-sm">
        <span className="font-semibold text-ink">Image pipeline</span>
        <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-line">
          <div className="h-full rounded-full bg-primary transition-[width] duration-700 ease-out" style={{ width: `${(visible.length / STAGE_PLAN.length) * 100}%` }} />
        </div>
        <span className="mono text-muted" aria-live="polite">{visible.length}/{STAGE_PLAN.length}</span>
      </div>
      <div className="relative min-h-[300px] overflow-hidden bg-navy px-4 py-5">
        {current ? (
          <div key={current.index} className="wipe-in">
            <StageLens stage={current} byIndex={allByIndex} thresholds={thresholds} />
          </div>
        ) : (
          <div className="flex h-[260px] flex-col items-center justify-center gap-3 text-white/80">
            <Disc3 aria-hidden className="h-10 w-10 animate-spin text-brand-yellow [animation-duration:2.4s]" />
            <span className="text-sm font-semibold">{running ? "Waiting for the inference worker…" : "No stages recorded"}</span>
          </div>
        )}
        {current && <span key={`scan-${current.index}`} aria-hidden className="scan-sweep pointer-events-none absolute inset-0 border-b-2 border-brand-yellow bg-gradient-to-b from-transparent via-transparent to-brand-yellow/25" />}
        {!current && running && <span aria-hidden className="scan-loop pointer-events-none absolute inset-0 border-b-2 border-brand-yellow/70 bg-gradient-to-b from-transparent to-brand-yellow/15" />}
        {next && (
          <span className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1 text-xs font-semibold text-white">
            <Loader2 aria-hidden className="h-3.5 w-3.5 animate-spin" /> next: {next.title}
          </span>
        )}
      </div>
      {current && plan && (
        <div key={`d-${current.index}`} className="space-y-2 border-t border-line px-4 py-3">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="text-xs font-bold uppercase tracking-[0.14em] text-primary">Stage {current.index} / {STAGE_PLAN.length}</span>
            <h4 className="log-in text-lg font-bold text-ink">{plan.title}</h4>
            <span className="mono text-sm text-primary-ink">{ms(current.duration_ms)}</span>
            {picked !== null && live && (
              <button type="button" onClick={() => setPicked(null)} className="ml-auto flex items-center gap-1.5 rounded-full bg-primary-soft px-3 py-1 text-xs font-semibold text-primary-ink">
                <Radio aria-hidden className="h-3.5 w-3.5" /> Follow live
              </button>
            )}
          </div>
          <ul className="flex flex-wrap gap-1.5" aria-label="What changed">
            {stageChanges(current, allByIndex).map((c, i) => (
              <li key={c} className="log-in rounded-full bg-accent-soft px-2.5 py-0.5 text-sm text-ink" style={{ animationDelay: `${100 + i * 90}ms` }}>{c}</li>
            ))}
          </ul>
          <ol className="mono space-y-0.5 text-xs leading-relaxed text-ink-soft">
            {current.log.slice(0, 4).map((l, i) => (
              <li key={i} className="log-in" style={{ animationDelay: `${250 + i * 120}ms` }}>› {l}</li>
            ))}
          </ol>
        </div>
      )}
      <ol className="grid grid-cols-4 gap-2 border-t border-line p-3 sm:grid-cols-8">
        {STAGE_PLAN.map((p) => {
          const s = byIndex.get(p.index);
          // Quality and inference record no picture of their own: show what they looked at.
          const img = s ? mediaUrl(s.image_url ?? (p.name === "quality" ? allByIndex.get(2)?.image_url : lb?.image_url)) : null;
          const active = current?.index === p.index;
          const params = (s?.params ?? {}) as Record<string, unknown>;
          return (
            <li key={p.index} data-stage={p.name} className="min-w-0">
              <button type="button" disabled={!s} onClick={() => setPicked(p.index)} aria-label={`Stage ${p.index}: ${p.title}`} aria-pressed={active}
                className={clsx("relative block aspect-square w-full overflow-hidden rounded-lg ring-2 transition",
                  active ? "ring-brand-yellow" : "ring-transparent hover:ring-line-strong",
                  s ? "pop-in bg-navy" : next?.index === p.index ? "shimmer" : "bg-surface-muted")}>
                {img && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={img} alt="" className={clsx("h-full w-full object-cover", p.name === "quality" && "grayscale")}
                    style={p.name === "letterbox" || p.name === "inference" ? { imageRendering: "pixelated" } : undefined} />
                )}
                {s && p.name === "quality" && (
                  <span className={clsx("absolute bottom-1 right-1 rounded-full px-1.5 text-[10px] font-bold text-white", params.status === "accepted" ? "bg-good" : "bg-warn")}>
                    {params.status === "accepted" ? "OK" : "LOW"}
                  </span>
                )}
                {s && p.name === "inference" && typeof params.p_defective === "number" && (
                  <span className="mono absolute inset-x-1 bottom-1 rounded bg-black/70 text-center text-[10px] font-bold text-brand-yellow">p {params.p_defective.toFixed(2)}</span>
                )}
                <span className={clsx("absolute left-1 top-1 flex h-5 min-w-5 items-center justify-center rounded-full px-1 text-[11px] font-bold",
                  s ? "bg-navy/85 text-white" : "bg-line text-muted")}>{p.index}</span>
              </button>
              <span className={clsx("mt-1 block truncate text-center text-xs font-semibold", s ? "text-ink-soft" : "text-muted")} title={p.title}>{p.short ?? p.title}</span>
            </li>
          );
        })}
      </ol>
    </div>
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
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
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
