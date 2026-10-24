"use client";

import clsx from "clsx";
import { AlertTriangle, ArrowRight, CheckCircle2 } from "lucide-react";
import { useEffect, useState, type CSSProperties, type ReactNode } from "react";

import { mediaUrl } from "@/lib/api";
import { humanize } from "@/lib/format";
import type { Stage } from "@/lib/types";

/** Policy thresholds of the pinned model version (triage zones on the inference dial). */
export interface Thresholds { low: number; high: number; binary?: number }

type Params = Record<string, unknown>;
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const fixed = (v: unknown, d = 3) => (num(v) === null ? null : (v as number).toFixed(d));
const pct = (v: unknown, d = 1) => (num(v) === null ? null : `${((v as number) * 100).toFixed(d)}%`);
const kb = (v: unknown) => (num(v) === null ? null : `${Math.round((v as number) / 1024)} KB`);
const params = (s: Stage | undefined): Params => (s?.params ?? {}) as Params;
const PIXELATED: CSSProperties = { imageRendering: "pixelated" };

/** What a stage changed, stated only from its recorded parameters (missing values are left out, never guessed). */
export function stageChanges(stage: Stage, byIndex: Map<number, Stage>): string[] {
  const p = params(stage);
  const out: unknown[] = [];
  switch (stage.name) {
    case "ingest":
      out.push(num(p.width) && `${p.width}×${p.height} ${p.format ?? ""} ${p.mode ?? ""}`.trim(), kb(p.bytes),
        p.had_exif ? "EXIF metadata removed" : "no EXIF", p.had_gps ? "GPS removed" : "no GPS",
        p.exif_orientation ? `rotated upright (EXIF ${p.exif_orientation})` : "no rotation needed");
      break;
    case "canonical": {
      const i = params(byIndex.get(1));
      out.push(num(i.width) && num(p.canonical_width) && `${i.width}×${i.height} → ${p.canonical_width}×${p.canonical_height}`,
        num(p.scale) !== null && (p.scale === 1 ? "scale 1.000 (already ≤ 1024 px)" : `scale ${fixed(p.scale)}`),
        `${i.format ?? "image"} → JPEG q${p.quality ?? "?"}`, kb(i.bytes) && kb(p.bytes) && `${kb(i.bytes)} → ${kb(p.bytes)}`);
      break;
    }
    case "letterbox": {
      const c = params(byIndex.get(2));
      out.push(num(c.canonical_width) && `${c.canonical_width}×${c.canonical_height} → ${p.resized_w}×${p.resized_h}`,
        num(p.size) && `on a ${p.size}×${p.size} canvas`, num(p.ratio) && `ratio ${fixed(p.ratio, 4)}`,
        pct(p.pad_fraction) && `${pct(p.pad_fraction)} padding (value ${p.pad_value})`, "aspect kept, nothing cropped");
      break;
    }
    case "depth":
      out.push(pct(p.near_surface_fraction, 0) && `nearest surface covers ${pct(p.near_surface_fraction, 0)} of the frame`,
        p.framing ? String(p.framing) : null, num(p.input_px) && `${p.model ?? "depth model"} at ${p.input_px} px`,
        "relative depth, no units", "advisory: not used for the verdict");
      break;
    case "tensor":
      out.push(Array.isArray(p.shape) && `shape [${(p.shape as number[]).join(", ")}]`, String(p.dtype ?? ""),
        num(p.min) !== null && `range [${fixed(p.min)}, ${fixed(p.max)}]`);
      break;
    case "quality": {
      const d = (p.descriptors ?? {}) as Params;
      const sj = (p.subject ?? null) as Params | null;
      out.push(`${humanize(String(p.status ?? ""))}`, fixed(d.brightness) && `brightness ${fixed(d.brightness)}`,
        fixed(d.contrast) && `contrast ${fixed(d.contrast)}`, fixed(d.blur_variance, 1) && `blur variance ${fixed(d.blur_variance, 1)}`,
        sj && fixed(sj.score) && `unusual-image score ${fixed(sj.score)} (${sj.unusual ? "unusual" : "typical"})`,
        sj?.vision_tyre_visible === false ? "vision model: no tyre" : sj?.vision_tyre_visible === true ? "vision model: tyre" : null,
        ...((p.reasons as string[] | undefined) ?? []).map(humanize));
      break;
    }
    case "inference": {
      const raw = (p.raw_probs ?? {}) as Params;
      out.push(pct(raw.defective) && `raw p(defective) ${pct(raw.defective)}`, fixed(p.temperature) && `temperature T ${fixed(p.temperature)}`,
        pct(p.p_defective) && `calibrated ${pct(p.p_defective)}`, p.triage ? humanize(String(p.triage)) : null);
      break;
    }
    case "attribution": {
      const g = num(p.grid);
      const cell = p.strongest_cell as number[] | undefined;
      out.push(g && `${g}×${g} occlusion grid · ${g * g} occluded copies`,
        cell && `strongest cell: row ${cell[0] + 1}, column ${cell[1] + 1}`,
        fixed(p.max_delta, 2) && `largest margin drop +${fixed(p.max_delta, 2)}`);
      break;
    }
    case "classified": {
      const so = p.second_opinion as { decision?: string; model?: string } | undefined;
      const dm = p.damage as { boxes?: { class: string; confidence: number }[]; escalate?: boolean } | undefined;
      out.push(p.triage ? humanize(String(p.triage)) : null, fixed(p.p_defective) && `p(defective) ${fixed(p.p_defective)}`,
        so?.decision && `vision second opinion ${so.decision}${so.model ? ` (${so.model})` : ""}`,
        p.policy_triage && p.policy_triage !== p.triage ? `classifier alone: ${humanize(String(p.policy_triage))}` : null,
        dm ? (dm.boxes?.length ? `damage locator: ${dm.boxes.slice(0, 2).map((b) => `${b.class.replace("_", " ")} ${b.confidence.toFixed(2)}`).join(", ")}${dm.escalate ? " (escalated)" : ""}` : "damage locator: no boxes") : null,
        "human clearance required");
      break;
    }
  }
  return out.filter((x): x is string => typeof x === "string" && x.length > 0);
}

function Shot({ src, label, pixelated, children, className }: {
  src: string | null;
  label: ReactNode;
  pixelated?: boolean;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <figure className="flex min-w-0 flex-col items-center gap-2">
      <div className="relative max-w-full">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={src} alt="" className={clsx("block max-h-[230px] max-w-full rounded-md object-contain md:max-h-[260px]", className)}
            style={pixelated ? PIXELATED : undefined} />
        ) : <div className="h-40 w-40 rounded-md bg-white/10" />}
        {children}
      </div>
      <figcaption className="mono text-center text-xs text-white/85">{label}</figcaption>
    </figure>
  );
}

function Compare({ before, after }: { before: ReactNode; after: ReactNode }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3">
      {before}
      <ArrowRight aria-hidden className="h-6 w-6 text-brand-yellow" />
      {after}
    </div>
  );
}

/** The same central detail at the same zoom from two images: makes a resolution change visible. */
function Detail({ src, label, zoomPct, pixelated }: { src: string | null; label: string; zoomPct: number; pixelated?: boolean }) {
  return (
    <figure className="flex flex-col items-center gap-1">
      <div className="h-28 w-28 rounded-md bg-no-repeat ring-2 ring-white/20 sm:h-32 sm:w-32"
        style={{ backgroundImage: src ? `url(${src})` : undefined, backgroundSize: `${zoomPct}% auto`, backgroundPosition: "50% 50%",
          ...(pixelated ? PIXELATED : {}) }} />
      <figcaption className="text-xs text-white/75">{label}</figcaption>
    </figure>
  );
}

function LetterboxLens({ stage, byIndex }: { stage: Stage; byIndex: Map<number, Stage> }) {
  const p = params(stage);
  const c = params(byIndex.get(2));
  const size = num(p.size) ?? 224;
  const [left, top, rw, rh] = [num(p.left) ?? 0, num(p.top) ?? 0, num(p.resized_w) ?? size, num(p.resized_h) ?? size];
  const canonical = mediaUrl(byIndex.get(2)?.image_url);
  const box = { left: `${(left / size) * 100}%`, top: `${(top / size) * 100}%`, width: `${(rw / size) * 100}%`, height: `${(rh / size) * 100}%` };
  // Show 30% of the content width, magnified, from both images.
  const frac = 0.3;
  return (
    <div className="space-y-4">
      <Compare
        before={<Shot src={canonical} label={`canonical ${c.canonical_width ?? "?"}×${c.canonical_height ?? "?"} px`} />}
        after={
          <Shot src={mediaUrl(stage.image_url)} pixelated label={`model input ${size}×${size} px`}>
            <span aria-hidden className="absolute rounded-sm border-2 border-dashed border-brand-yellow" style={box} />
          </Shot>
        } />
      <div className="flex flex-wrap items-center justify-center gap-4">
        <Detail src={canonical} label="detail at full resolution" zoomPct={100 / frac} />
        <ArrowRight aria-hidden className="h-5 w-5 text-brand-yellow" />
        <Detail src={mediaUrl(stage.image_url)} pixelated label={`same detail at ${size} px`} zoomPct={100 / (frac * (rw / size))} />
        <p className="max-w-56 text-xs text-white/75">Dashed box: the photo scaled into the canvas. Outside it is padding
          (shown lavender); the model sees only {size}×{size} pixels.</p>
      </div>
    </div>
  );
}

function TensorLens({ stage, byIndex }: { stage: Stage; byIndex: Map<number, Stage> }) {
  const p = params(stage);
  const means = (p.channel_mean as number[] | undefined) ?? [];
  return (
    <div className="space-y-4">
      <Compare
        before={<Shot src={mediaUrl(byIndex.get(3)?.image_url)} pixelated label="RGB image (uint8)" />}
        after={<Shot src={mediaUrl(stage.image_url)} label="3 planes: R | G | B (float32)" />} />
      {means.length === 3 && (
        <div className="mx-auto grid max-w-md grid-cols-3 gap-3">
          {means.map((m, i) => (
            <div key={i}>
              <div className="flex justify-between text-xs text-white/80"><span>{"RGB"[i]} mean</span><span className="mono">{m.toFixed(3)}</span></div>
              <div className="h-2 rounded-full bg-white/15">
                <div className={clsx("h-full rounded-full", ["bg-[#e5484d]", "bg-[#30a46c]", "bg-[#3e8eed]"][i])} style={{ width: `${m * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Edge map and luminance histogram of the stored canonical image, computed in the browser for display. */
function useImageAnalysis(src: string | null) {
  const [res, setRes] = useState<{ edges: string; hist: number[] } | null>(null);
  useEffect(() => {
    if (!src) return;
    let alive = true;
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, 240 / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.max(3, Math.round(img.naturalWidth * scale));
      const h = Math.max(3, Math.round(img.naturalHeight * scale));
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const g = canvas.getContext("2d", { willReadFrequently: true });
      if (!g) return;
      g.drawImage(img, 0, 0, w, h);
      const { data } = g.getImageData(0, 0, w, h);
      const lum = new Float32Array(w * h);
      const hist = new Array<number>(32).fill(0);
      for (let i = 0; i < w * h; i++) {
        const y = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
        lum[i] = y;
        hist[Math.min(31, Math.floor(y * 32))] += 1;
      }
      const out = g.createImageData(w, h);
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          const lap = 4 * lum[i] - lum[i - 1] - lum[i + 1] - lum[i - w] - lum[i + w];
          const v = Math.min(255, Math.abs(lap) * 255 * 3);
          out.data.set([v, v, v, 255], i * 4);
        }
      }
      g.putImageData(out, 0, 0);
      if (alive) setRes({ edges: canvas.toDataURL("image/png"), hist });
    };
    img.src = src;
    return () => { alive = false; };
  }, [src]);
  return res;
}

function QualityLens({ stage, byIndex }: { stage: Stage; byIndex: Map<number, Stage> }) {
  const p = params(stage);
  const d = (p.descriptors ?? {}) as Params;
  const t = (p.thresholds ?? {}) as Params;
  const canonical = mediaUrl(byIndex.get(2)?.image_url);
  const analysis = useImageAnalysis(canonical);
  const ok = p.status === "accepted";
  const rejected = p.status === "rejected";
  const bright = num(d.brightness);
  const [minB, maxB] = [num(t.min_brightness), num(t.max_brightness)];
  const blur = num(d.blur_variance);
  const minBlur = num(t.min_blur_variance);
  const logPos = (v: number) => Math.min(100, Math.max(0, (Math.log10(Math.max(v, 1)) / 4) * 100)); // 1..10^4
  const peak = Math.max(1, ...(analysis?.hist ?? [1]));
  return (
    <div className="grid items-center gap-5 md:grid-cols-[auto_1fr]">
      <Shot src={analysis?.edges ?? canonical} pixelated={false}
        label={analysis ? "edge map (Laplacian): what the blur check measures" : "computing edge map…"} />
      <div className="space-y-4 text-white">
        <p className={clsx("inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-bold", ok ? "bg-good text-white" : rejected ? "bg-bad text-white" : "bg-warn text-white")}>
          {ok ? <CheckCircle2 aria-hidden className="h-4 w-4" /> : <AlertTriangle aria-hidden className="h-4 w-4" />}
          {ok ? "Quality accepted" : rejected ? "Not a tyre: rejected, no verdict" : "Low quality: escalated to a human"}
        </p>
        <div>
          <div className="mb-1 flex justify-between text-xs text-white/80"><span>Brightness histogram</span>
            <span className="mono">mean {bright?.toFixed(3) ?? "n/a"} · allowed {minB ?? "?"}–{maxB ?? "?"}</span></div>
          <svg viewBox="0 0 320 70" className="h-20 w-full" role="img" aria-label="Brightness histogram with the allowed range">
            {minB !== null && maxB !== null && <rect x={minB * 320} y={0} width={(maxB - minB) * 320} height={62} fill="rgb(255 255 255 / 0.08)" />}
            {(analysis?.hist ?? []).map((n, i) => (
              <rect key={i} x={i * 10 + 1} y={62 - (n / peak) * 58} width={8} height={(n / peak) * 58} fill="rgb(255 255 255 / 0.7)" />
            ))}
            {bright !== null && <line x1={bright * 320} x2={bright * 320} y1={0} y2={66} stroke="var(--brand-yellow)" strokeWidth={2.5} />}
            {[minB, maxB].map((v, i) => v !== null && <line key={i} x1={v * 320} x2={v * 320} y1={0} y2={66} stroke="rgb(255 255 255 / 0.6)" strokeDasharray="3 3" />)}
          </svg>
        </div>
        <div>
          <div className="mb-1 flex justify-between text-xs text-white/80"><span>Sharpness (blur variance, log scale)</span>
            <span className="mono">{blur?.toFixed(1) ?? "n/a"} · min {minBlur ?? "?"}</span></div>
          <div className="relative h-2.5 rounded-full bg-white/15">
            {blur !== null && <div className={clsx("h-full rounded-full", minBlur !== null && blur < minBlur ? "bg-warn" : "bg-good")} style={{ width: `${logPos(blur)}%` }} />}
            {minBlur !== null && <span aria-hidden className="absolute -top-1 h-4.5 w-0.5 bg-white" style={{ left: `${logPos(minBlur)}%` }} />}
          </div>
        </div>
        <p className="text-xs text-white/60">The edge map and histogram are drawn in your browser from the stored image; the
          decision uses the worker&apos;s recorded values.</p>
      </div>
    </div>
  );
}

function point(cx: number, cy: number, r: number, p: number) {
  return [cx - r * Math.cos(p * Math.PI), cy - r * Math.sin(p * Math.PI)] as const;
}

/** Half-dial of calibrated p(defective) with the policy's triage zones; the needle sweeps in on first view. */
export function ProbabilityDial({ p, raw, thresholds }: { p: number; raw: number | null; thresholds: Thresholds | null }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(p));
    return () => cancelAnimationFrame(id);
  }, [p]);
  const [cx, cy, r] = [120, 118, 96];
  const arc = (a: number, b: number) => {
    const [x1, y1] = point(cx, cy, r, a);
    const [x2, y2] = point(cx, cy, r, b);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  const zones = thresholds
    ? [[0, thresholds.low, "#30a46c"], [thresholds.low, thresholds.high, "#f5a524"], [thresholds.high, 1, "#e5484d"]] as const
    : [[0, 1, "rgb(255 255 255 / 0.35)"]] as const;
  const rawPt = raw !== null ? point(cx, cy, r, raw) : null;
  return (
    <svg viewBox="0 0 240 140" className="w-full max-w-xs" role="img" aria-label={`Calibrated p(defective) ${(p * 100).toFixed(1)}%`}>
      {zones.map(([a, b, c]) => <path key={a} d={arc(a, b)} stroke={c} strokeWidth={16} fill="none" />)}
      {rawPt && <circle cx={rawPt[0]} cy={rawPt[1]} r={6} fill="none" stroke="#fff" strokeWidth={2} strokeDasharray="2 2" />}
      <g style={{ transform: `rotate(${shown * 180}deg)`, transformOrigin: `${cx}px ${cy}px`, transition: "transform 1100ms cubic-bezier(0.2,0.7,0.2,1)" }}>
        <line x1={cx} y1={cy} x2={cx - r + 18} y2={cy} stroke="var(--brand-yellow)" strokeWidth={5} strokeLinecap="round" />
      </g>
      <circle cx={cx} cy={cy} r={8} fill="var(--brand-yellow)" />
      <text x={cx - r} y={cy + 18} fontSize={10} fill="rgb(255 255 255 / 0.7)" textAnchor="middle">0</text>
      <text x={cx + r} y={cy + 18} fontSize={10} fill="rgb(255 255 255 / 0.7)" textAnchor="middle">1</text>
    </svg>
  );
}

function InferenceLens({ stage, byIndex, thresholds }: { stage: Stage; byIndex: Map<number, Stage>; thresholds: Thresholds | null }) {
  const p = params(stage);
  const pd = num(p.p_defective) ?? 0;
  const raw = num(((p.raw_probs ?? {}) as Params).defective);
  return (
    <div className="grid items-center gap-5 md:grid-cols-[auto_1fr]">
      <Shot src={mediaUrl(byIndex.get(3)?.image_url)} pixelated label="what the classifier sees: 224×224×3" />
      <div className="flex flex-col items-center gap-2 text-white">
        <ProbabilityDial p={pd} raw={raw} thresholds={thresholds} />
        <p className="headline text-3xl">{(pd * 100).toFixed(1)}%</p>
        <p className="text-sm text-white/80">calibrated p(defective) · dashed ring = raw softmax {raw !== null ? `${(raw * 100).toFixed(1)}%` : ""}</p>
        {thresholds && (
          <p className="flex flex-wrap justify-center gap-3 text-xs text-white/75">
            <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#30a46c]" />retread &lt; {thresholds.low}</span>
            <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#f5a524]" />expert review</span>
            <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#e5484d]" />defect ≥ {thresholds.high}</span>
          </p>
        )}
      </div>
    </div>
  );
}

function AttributionLens({ stage, byIndex }: { stage: Stage; byIndex: Map<number, Stage> }) {
  const p = params(stage);
  const lb = params(byIndex.get(3));
  const grid = num(p.grid) ?? 8;
  const size = num(lb.size) ?? 224;
  const [left, top, rw, rh] = [num(lb.left) ?? 0, num(lb.top) ?? 0, num(lb.resized_w) ?? size, num(lb.resized_h) ?? size];
  const cell = size / grid;
  // Grid lines live in letterbox space; the heatmap image is the canonical photo, so map through the padding.
  const xs = Array.from({ length: grid + 1 }, (_, k) => ((k * cell - left) / rw) * 100).filter((v) => v > 0 && v < 100);
  const ys = Array.from({ length: grid + 1 }, (_, k) => ((k * cell - top) / rh) * 100).filter((v) => v > 0 && v < 100);
  const sc = p.strongest_cell as number[] | undefined;
  const clampPct = (v: number) => Math.max(0, Math.min(100, v));
  const hot = sc && {
    left: `${clampPct(((sc[1] * cell - left) / rw) * 100)}%`, top: `${clampPct(((sc[0] * cell - top) / rh) * 100)}%`,
    width: `${(cell / rw) * 100}%`, height: `${(cell / rh) * 100}%`,
  };
  return (
    <Compare
      before={<Shot src={mediaUrl(byIndex.get(2)?.image_url)} label="canonical photo" />}
      after={
        <Shot src={mediaUrl(stage.image_url)} label="red: pushes toward defective · blue: toward good">
          <svg aria-hidden className="absolute inset-0 h-full w-full" preserveAspectRatio="none" viewBox="0 0 100 100">
            {xs.map((x) => <line key={`x${x}`} x1={x} x2={x} y1={0} y2={100} stroke="rgb(255 255 255 / 0.45)" strokeWidth={0.4} />)}
            {ys.map((y) => <line key={`y${y}`} y1={y} y2={y} x1={0} x2={100} stroke="rgb(255 255 255 / 0.45)" strokeWidth={0.4} />)}
          </svg>
          {hot && <span aria-hidden className="absolute animate-pulse rounded-sm border-[3px] border-brand-yellow" style={hot} />}
        </Shot>
      } />
  );
}

function DepthLens({ stage, byIndex }: { stage: Stage; byIndex: Map<number, Stage> }) {
  const p = params(stage);
  return (
    <div className="space-y-3">
      <Compare before={<Shot src={mediaUrl(byIndex.get(2)?.image_url)} label="canonical photo" />}
        after={<Shot src={mediaUrl(stage.image_url)} label="relative depth: bright = near · yellow outline = nearest surface" />} />
      <p className="text-center text-xs text-white/75">
        {String(p.limits ?? "Relative depth: advisory only, not used for the verdict.")}
      </p>
    </div>
  );
}

/** The large visual for one stage: before → after, pixel-level detail, or a chart, depending on what changed. */
export function StageLens({ stage, byIndex, thresholds }: { stage: Stage; byIndex: Map<number, Stage>; thresholds: Thresholds | null }) {
  switch (stage.name) {
    case "canonical":
      return <Compare before={<Shot src={mediaUrl(byIndex.get(1)?.image_url)} label="uploaded (oriented, metadata removed)" />}
        after={<Shot src={mediaUrl(stage.image_url)} label="canonical JPEG (training path)" />} />;
    case "depth": return <DepthLens stage={stage} byIndex={byIndex} />;
    case "letterbox": return <LetterboxLens stage={stage} byIndex={byIndex} />;
    case "tensor": return <TensorLens stage={stage} byIndex={byIndex} />;
    case "quality": return <QualityLens stage={stage} byIndex={byIndex} />;
    case "inference": return <InferenceLens stage={stage} byIndex={byIndex} thresholds={thresholds} />;
    case "attribution": return <AttributionLens stage={stage} byIndex={byIndex} />;
    default:
      return <div className="flex justify-center"><Shot src={mediaUrl(stage.image_url)} label={stage.name === "ingest" ? "your photo, upright, metadata stripped" : "final decision"} /></div>;
  }
}
