"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Cctv, ScanSearch } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { api, mediaUrl } from "@/lib/api";
import { ago } from "@/lib/format";
import { qk } from "@/lib/query";
import type { Camera } from "@/lib/types";

const VERDICT = {
  damaged: { label: "Damaged", cls: "bg-bad text-white" },
  not_damaged: { label: "Not damaged", cls: "bg-good text-white" },
  uncertain: { label: "Expert check", cls: "bg-warn text-ink" },
} as const;

/**
 * The camera as the line sees it: annotated frames from the connector (~4 per second), each preloaded before it
 * is swapped in so the picture never flickers. The connector draws the inspection zone in the colour of the
 * latest verdict and the damage locator's boxes on every frame. Without live frames (connector stopped, or an
 * older connector) it falls back to the plain snapshot refreshed every 2 s.
 */
/** What the connector drew on the newest frame (the X-Overlay header): the verdict for the tyre in the zone. */
interface Overlay {
  verdict: keyof typeof VERDICT | null;
  p_defective: number | null;
  inspection_id: string | null;
  inspecting: boolean;
  boxes: { class: string; confidence: number }[];
  experimental: boolean;
}

/**
 * The camera as the line sees it: annotated frames from the connector (~4 per second), each fetched before it is
 * swapped in so the picture never flickers. The connector draws the inspection zone in the colour of the verdict
 * for the tyre in it and the damage locator's boxes on every frame; the chip below repeats what that frame shows.
 * Without live frames (connector stopped, or an older connector) it falls back to the snapshot every 2 s.
 */
export function LiveCamera({ cam, compact = false }: { cam: Camera; compact?: boolean }) {
  const [frame, setFrame] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [liveFailed, setLiveFailed] = useState(false);
  const busy = useRef(false);
  useEffect(() => {
    if (!cam.live) return;
    let stop = false;
    let current: string | null = null;
    const tick = async () => {
      if (stop || busy.current) return;
      busy.current = true;
      try {
        const res = await fetch(`/api/v1/cameras/${cam.id}/live`, { cache: "no-store" });
        if (!res.ok) throw new Error(String(res.status));
        const url = URL.createObjectURL(await res.blob());
        const meta = res.headers.get("x-overlay");
        if (stop) { URL.revokeObjectURL(url); return; }
        setFrame(url);
        setOverlay(meta ? (JSON.parse(meta) as Overlay) : null);
        setLiveFailed(false);
        if (current) URL.revokeObjectURL(current);
        current = url;
      } catch {
        if (!stop) setLiveFailed(true);
      } finally {
        busy.current = false;
      }
    };
    void tick();
    const t = setInterval(() => void tick(), 250);
    return () => { stop = true; clearInterval(t); if (current) URL.revokeObjectURL(current); };
  }, [cam.id, cam.live]);
  const snap = useQuery({
    queryKey: qk.cameraSnapshot(cam.id),
    queryFn: () => api<{ url: string; at: string }>(`/v1/cameras/${cam.id}/snapshot`),
    enabled: Boolean(cam.last_frame_at) && (liveFailed || !frame),
    refetchInterval: cam.live && (liveFailed || !frame) ? 2000 : false,
  });
  const live = Boolean(frame && !liveFailed);
  const src = (live && frame) || mediaUrl(snap.data?.url);
  const annotated = live && cam.overlay;
  const r = cam.last_result;
  return (
    <div className="space-y-2">
      <div className="relative aspect-video overflow-hidden rounded-xl bg-navy">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={src} alt={`Live view of ${cam.name}`} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-white/70">
            <Cctv aria-hidden className="h-8 w-8" />
            <span className="text-sm">{cam.enabled ? "No frame yet" : "Camera disabled"}</span>
          </div>
        )}
        {cam.live && !annotated && ( // an annotated frame carries its own LIVE heads-up line
          <span className="absolute right-2 top-2 flex items-center gap-1.5 rounded-full bg-black/65 px-2.5 py-0.5 text-xs font-bold text-white">
            <span aria-hidden className="pulse-dot h-2 w-2 rounded-full bg-bad" /> LIVE
          </span>
        )}
        {annotated && <span className="sr-only">Live, with the AI overlay</span>}
        {!compact && cam.resolution && (
          <span className="mono absolute bottom-2 right-2 rounded bg-black/65 px-2 py-0.5 text-xs text-white">
            {cam.resolution} · {cam.fps ?? "?"} fps
          </span>
        )}
      </div>
      {annotated && overlay ? (
        <p className="flex min-h-7 flex-wrap items-center gap-2 text-sm" aria-live="polite">
          <span className="text-muted">In the zone:</span>
          {overlay.verdict ? (
            <span className={clsx("rounded-full px-2.5 py-0.5 text-xs font-bold", VERDICT[overlay.verdict].cls)}>{VERDICT[overlay.verdict].label}</span>
          ) : <span className="rounded-full bg-surface-muted px-2.5 py-0.5 text-xs font-bold text-ink-soft">{overlay.inspecting ? "Inspecting…" : "Waiting for a tyre"}</span>}
          {overlay.p_defective !== null && <span className="mono text-ink-soft">p {overlay.p_defective.toFixed(2)}</span>}
          {overlay.boxes.length > 0 && (
            <span className="flex items-center gap-1 text-ink-soft"><ScanSearch aria-hidden className="h-3.5 w-3.5" />
              {overlay.boxes.slice(0, 2).map((b) => `${b.class.replace("_", " ")} ${b.confidence.toFixed(2)}`).join(", ")}
              {overlay.experimental ? " (experimental)" : ""}</span>
          )}
          {overlay.inspection_id && <Link href={`/inspections/${overlay.inspection_id}`} className="ml-auto font-semibold text-primary-ink hover:underline">Open →</Link>}
        </p>
      ) : r ? (
        <p className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-muted">Last result:</span>
          <span className={clsx("rounded-full px-2.5 py-0.5 text-xs font-bold", VERDICT[r.verdict].cls)}>{VERDICT[r.verdict].label}</span>
          <span className="mono text-ink-soft">p {r.p_defective.toFixed(2)}{r.model_family ? ` · ${r.model_family}` : ""}</span>
          <span className="text-muted">{ago(r.at)}</span>
          <Link href={`/inspections/${r.inspection_id}`} className="ml-auto font-semibold text-primary-ink hover:underline">Open →</Link>
        </p>
      ) : null}
    </div>
  );
}
