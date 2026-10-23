"use client";

import clsx from "clsx";
import { Brain, CheckCircle2, Disc3, Gauge, Loader2, Radio } from "lucide-react";
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

function ViewerFrame({ stage }: { stage: Stage }) {
  const url = mediaUrl(stage.image_url);
  return (
    <div className="absolute inset-0 overflow-hidden">
      {url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={url} alt={`Stage ${stage.index}: ${stage.title}`} className="wipe-in h-full w-full object-contain" />
      ) : (
        <div className="wipe-in h-full overflow-auto bg-surface"><DataPanel stage={stage} /></div>
      )}
      <span aria-hidden className="scan-sweep pointer-events-none absolute inset-0 border-b-2 border-brand-yellow bg-gradient-to-b from-transparent via-transparent to-brand-yellow/35" />
    </div>
  );
}

/**
 * The pipeline as a short film: each recorded stage wipes into a large viewer (with a scan line) in order, its own
 * log lines type in beside it, and the filmstrip fills up. Click any finished stage to look at it again.
 */
export function StageTheatre({ stages, running, stepMs = 850, onRevealed }: {
  stages: Stage[];
  running: boolean;
  stepMs?: number;
  /** Called once every recorded stage has been shown and the job is finished. */
  onRevealed?: () => void;
}) {
  const ordered = useMemo(() => [...stages].sort((a, b) => a.index - b.index), [stages]);
  const shown = useStagedReveal(ordered.length, stepMs);
  useEffect(() => {
    if (!running && shown >= ordered.length) onRevealed?.();
  }, [running, shown, ordered.length, onRevealed]);
  const visible = ordered.slice(0, shown);
  const [picked, setPicked] = useState<number | null>(null);
  const byIndex = new Map(visible.map((s) => [s.index, s]));
  const latest = visible[visible.length - 1];
  const current = (picked !== null && byIndex.get(picked)) || latest;
  const plan = current ? STAGE_PLAN.find((p) => p.index === current.index) : undefined;
  const live = running || shown < ordered.length;
  const next = live ? STAGE_PLAN.find((p) => !byIndex.has(p.index)) : undefined;
  return (
    <div className="overflow-hidden rounded-2xl border border-line bg-surface" role="region" aria-label="Preprocessing and classification stages">
      <div className="flex items-center gap-3 border-b border-line px-4 py-2.5 text-sm">
        <span className="font-display font-bold text-ink">Image pipeline</span>
        <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-line">
          <div className="h-full rounded-full bg-primary transition-[width] duration-700 ease-out" style={{ width: `${(visible.length / STAGE_PLAN.length) * 100}%` }} />
        </div>
        <span className="mono text-muted" aria-live="polite">{visible.length}/{STAGE_PLAN.length}</span>
      </div>
      <div className="grid md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div className="relative aspect-[4/3] bg-navy md:aspect-auto md:min-h-[300px]">
          {current ? <ViewerFrame key={current.index} stage={current} /> : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 overflow-hidden text-white/80">
              <Disc3 aria-hidden className="h-10 w-10 animate-spin text-brand-yellow [animation-duration:2.4s]" />
              <span className="text-sm font-semibold">{running ? "Waiting for the inference worker…" : "No stages recorded"}</span>
              {running && <span aria-hidden className="scan-loop pointer-events-none absolute inset-0 border-b-2 border-brand-yellow/70 bg-gradient-to-b from-transparent to-brand-yellow/15" />}
            </div>
          )}
          {next && (
            <span className="absolute bottom-3 left-3 flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1 text-xs font-semibold text-white">
              <Loader2 aria-hidden className="h-3.5 w-3.5 animate-spin" /> {next.title}…
            </span>
          )}
        </div>
        <div className="flex min-w-0 flex-col gap-1.5 p-4">
          {current && plan ? (
            <>
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-primary">Stage {current.index} of {STAGE_PLAN.length}</p>
              <h4 key={`t-${current.index}`} className="log-in text-xl font-extrabold text-ink">{plan.title}</h4>
              <p className="text-sm text-ink-soft">{plan.caption}</p>
              <p className="mono text-sm font-semibold text-primary-ink">{ms(current.duration_ms)}</p>
              <ol key={`l-${current.index}`} className="mono mt-1 space-y-1 text-xs leading-relaxed text-ink-soft">
                {current.log.slice(0, 6).map((l, i) => (
                  <li key={i} className="log-in" style={{ animationDelay: `${150 + i * 140}ms` }}>› {l}</li>
                ))}
              </ol>
              {picked !== null && live && (
                <button type="button" onClick={() => setPicked(null)} className="mt-auto flex w-fit items-center gap-1.5 rounded-full bg-primary-soft px-3 py-1 text-xs font-semibold text-primary-ink">
                  <Radio aria-hidden className="h-3.5 w-3.5" /> Follow live
                </button>
              )}
            </>
          ) : (
            <p className="text-sm text-muted">Stages appear here as the worker records them: ingest → canonical → letterbox → tensor → quality → YOLO → attribution → classified.</p>
          )}
        </div>
      </div>
      <ol className="grid grid-cols-4 gap-2 border-t border-line p-3 sm:grid-cols-8">
        {STAGE_PLAN.map((p) => {
          const s = byIndex.get(p.index);
          const url = s ? mediaUrl(s.image_url) : null;
          const active = current?.index === p.index;
          const Glyph = p.name === "quality" ? Gauge : Brain;
          return (
            <li key={p.index} data-stage={p.name} className="min-w-0">
              <button type="button" disabled={!s} onClick={() => setPicked(p.index)} aria-label={`Stage ${p.index}: ${p.title}`} aria-pressed={active}
                className={clsx("relative block aspect-square w-full overflow-hidden rounded-lg ring-2 transition",
                  active ? "ring-brand-yellow" : "ring-transparent hover:ring-line-strong",
                  s ? "pop-in bg-navy" : next?.index === p.index ? "shimmer" : "bg-surface-muted")}>
                {s && url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={url} alt="" className="h-full w-full object-cover" />
                ) : s ? (
                  <span className="flex h-full items-center justify-center bg-primary-soft text-primary"><Glyph aria-hidden className="h-6 w-6" /></span>
                ) : null}
                <span className={clsx("absolute left-1 top-1 flex h-5 min-w-5 items-center justify-center rounded-full px-1 text-[11px] font-bold",
                  s ? "bg-navy/85 text-white" : "bg-line text-muted")}>{p.index}</span>
              </button>
              <span className={clsx("mt-1 block truncate text-center text-xs font-semibold", s ? "text-ink-soft" : "text-muted")}>{p.title}</span>
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
