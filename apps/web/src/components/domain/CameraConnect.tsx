"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Plug, X } from "lucide-react";
import { useState } from "react";

import { Button, ErrorState, Field, inputClass } from "@/components/ui";
import { api, mediaUrl } from "@/lib/api";
import { qk } from "@/lib/query";
import type { Camera, CameraProbe } from "@/lib/types";

const ZONES: Record<string, { label: string; roi: number[] | null }> = {
  whole: { label: "Whole frame", roi: null },
  centre: { label: "Centre (half the frame)", roi: [0.25, 0.25, 0.75, 0.75] },
  belt: { label: "Conveyor centre (tall band)", roi: [0.256, 0.067, 0.744, 0.933] },
  custom: { label: "Custom…", roi: null },
};

/**
 * Connect a line camera: RTSP (or HTTP) URL, a test frame, the inspection zone drawn on that frame, and capture
 * settings. Used on the Cameras page and inside Bib's chat; inspectors and admins can connect cameras.
 */
export function CameraConnect({ onClose, onSaved, compact = false }: { onClose?: () => void; onSaved?: (c: Camera) => void; compact?: boolean }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({ name: "", line_id: "", stream_url: "rtsp://", simulated: false, capture_mode: "on_change", interval_s: 3, overlay: true });
  const [zone, setZone] = useState<keyof typeof ZONES>("whole");
  const [custom, setCustom] = useState([0.25, 0.1, 0.75, 0.9]);
  const set = (k: keyof typeof form, v: string | number | boolean) => setForm((f) => ({ ...f, [k]: v }));
  const roi = zone === "custom" ? custom : ZONES[zone].roi;
  const probe = useMutation({
    mutationFn: () => api<CameraProbe>("/v1/cameras/probe", { method: "POST", json: { stream_url: form.stream_url } }),
  });
  const save = useMutation({
    mutationFn: () => api<Camera>("/v1/cameras", { method: "POST", json: { ...form, line_id: form.line_id || null, roi } }),
    onSuccess: (c) => { qc.invalidateQueries({ queryKey: qk.cameras }); onSaved?.(c); onClose?.(); },
  });
  return (
    <form className={clsx("grid gap-4", !compact && "lg:grid-cols-[1fr_minmax(0,340px)]")} onSubmit={(e) => { e.preventDefault(); save.mutate(); }}
      aria-label="Connect a camera">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <Field label="Stream URL" htmlFor="cam-url" hint="rtsp://user:password@host:554/stream (rtsps:// and http(s):// work too). Stored encrypted, shown masked.">
            <input id="cam-url" required className={clsx(inputClass, "mono")} value={form.stream_url} onChange={(e) => set("stream_url", e.target.value)}
              autoComplete="off" spellCheck={false} placeholder="rtsp://192.168.1.20:554/stream1" />
          </Field>
        </div>
        <Field label="Name" htmlFor="cam-name"><input id="cam-name" required minLength={2} maxLength={80} className={inputClass} value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="Line 1 intake" /></Field>
        <Field label="Production line" htmlFor="cam-line"><input id="cam-line" maxLength={64} className={inputClass} value={form.line_id} onChange={(e) => set("line_id", e.target.value)} placeholder="L1" /></Field>
        <Field label="Inspection zone" htmlFor="cam-zone" hint="Only this part of the frame is inspected and boxed">
          <select id="cam-zone" className={inputClass} value={zone} onChange={(e) => setZone(e.target.value as keyof typeof ZONES)}>
            {Object.entries(ZONES).map(([k, z]) => <option key={k} value={k}>{z.label}</option>)}
          </select>
        </Field>
        <Field label="Capture" htmlFor="cam-mode">
          <select id="cam-mode" className={inputClass} value={form.capture_mode} onChange={(e) => set("capture_mode", e.target.value)}>
            <option value="on_change">When a tyre stops in the zone</option>
            <option value="interval">Every interval (moving belt)</option>
          </select>
        </Field>
        {zone === "custom" && (
          <div className="grid grid-cols-4 gap-2 sm:col-span-2">
            {["left", "top", "right", "bottom"].map((k, i) => (
              <Field key={k} label={k} htmlFor={`roi-${k}`}>
                <input id={`roi-${k}`} type="number" min={0} max={1} step={0.01} className={inputClass} value={custom[i]}
                  onChange={(e) => setCustom((c) => c.map((v, j) => (j === i ? Number(e.target.value) : v)))} />
              </Field>
            ))}
          </div>
        )}
        <Field label="At most every (s)" htmlFor="cam-int"><input id="cam-int" type="number" min={1} max={3600} className={inputClass} value={form.interval_s} onChange={(e) => set("interval_s", Number(e.target.value))} /></Field>
        <div className="flex flex-col justify-end gap-1.5 text-sm text-ink">
          <label className="flex items-center gap-2"><input type="checkbox" checked={form.overlay} onChange={(e) => set("overlay", e.target.checked)} className="h-4 w-4" /> AI overlay on the live view</label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={form.simulated} onChange={(e) => set("simulated", e.target.checked)} className="h-4 w-4" /> Simulated test stream</label>
        </div>
        <div className="flex flex-wrap gap-2 sm:col-span-2">
          <Button type="button" variant="secondary" busy={probe.isPending} disabled={form.stream_url.length < 10} onClick={() => probe.mutate()}>
            <Plug aria-hidden className="h-4 w-4" /> Test connection
          </Button>
          <Button type="submit" busy={save.isPending}>Save camera</Button>
          {onClose && <Button type="button" variant="ghost" onClick={onClose}><X aria-hidden className="h-4 w-4" /> Cancel</Button>}
        </div>
        {(probe.error || save.error) ? <div className="sm:col-span-2"><ErrorState error={probe.error ?? save.error} title={probe.error ? "Connection failed" : "Not saved"} /></div> : null}
      </div>
      <div className="flex flex-col gap-2">
        <div className="relative aspect-video overflow-hidden rounded-xl bg-navy">
          {probe.data ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={mediaUrl(probe.data.snapshot_url) ?? ""} alt="Frame from the tested stream" className="h-full w-full object-contain" />
          ) : <div className="flex h-full items-center justify-center px-4 text-center text-sm text-white/60">Test the connection to see a frame and the inspection zone</div>}
          {probe.data && roi && (
            <span aria-hidden className="absolute rounded-sm border-2 border-brand-yellow shadow-[0_0_0_9999px_rgb(0_0_0/0.35)]"
              style={{ left: `${roi[0] * 100}%`, top: `${roi[1] * 100}%`, width: `${(roi[2] - roi[0]) * 100}%`, height: `${(roi[3] - roi[1]) * 100}%` }} />
          )}
        </div>
        {probe.data && <p className="text-sm text-good">Connected: {probe.data.width}×{probe.data.height}{probe.data.fps ? ` · ${probe.data.fps} fps` : ""}</p>}
      </div>
    </form>
  );
}
