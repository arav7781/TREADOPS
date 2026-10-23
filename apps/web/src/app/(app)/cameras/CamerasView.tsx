"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Cctv, MessageSquare, Plus, Power, ScanLine, Trash2, X } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { useMe } from "@/components/AppShell";
import { Badge, Button, Card, EmptyState, ErrorState, Field, inputClass, PageHeader, SkeletonRows } from "@/components/ui";
import { api, mediaUrl } from "@/lib/api";
import { ago } from "@/lib/format";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { Camera, CameraProbe, InspectionAccepted } from "@/lib/types";

const STATE: Record<Camera["state"], { label: string; tone: "good" | "warn" | "bad" | "neutral" | "info" }> = {
  streaming: { label: "Live", tone: "good" }, connecting: { label: "Connecting", tone: "info" },
  error: { label: "Error", tone: "bad" }, idle: { label: "Waiting for connector", tone: "warn" },
  disabled: { label: "Disabled", tone: "neutral" },
};

function LiveView({ cam }: { cam: Camera }) {
  // A fresh signed snapshot URL every 2 s while the camera is live; the connector refreshes the frame itself.
  const snap = useQuery({
    queryKey: qk.cameraSnapshot(cam.id),
    queryFn: () => api<{ url: string; at: string }>(`/v1/cameras/${cam.id}/snapshot`),
    enabled: Boolean(cam.last_frame_at),
    refetchInterval: cam.live ? 2000 : false,
  });
  const src = mediaUrl(snap.data?.url);
  return (
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
      {cam.live && (
        <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded-full bg-black/65 px-2.5 py-0.5 text-xs font-bold text-white">
          <span aria-hidden className="pulse-dot h-2 w-2 rounded-full bg-bad" /> LIVE
        </span>
      )}
      {cam.resolution && (
        <span className="mono absolute bottom-2 right-2 rounded bg-black/65 px-2 py-0.5 text-xs text-white">
          {cam.resolution} · {cam.fps ?? "?"} fps
        </span>
      )}
    </div>
  );
}

function CameraCard({ cam, canManage, canInspect }: { cam: Camera; canManage: boolean; canInspect: boolean }) {
  const qc = useQueryClient();
  const refresh = () => qc.invalidateQueries({ queryKey: qk.cameras });
  const inspect = useMutation({
    mutationFn: () => api<InspectionAccepted>(`/v1/cameras/${cam.id}/inspect`, { method: "POST" }),
  });
  const toggle = useMutation({
    mutationFn: () => api<Camera>(`/v1/cameras/${cam.id}`, { method: "PATCH", json: { enabled: !cam.enabled } }),
    onSuccess: refresh,
  });
  const remove = useMutation({
    mutationFn: () => api(`/v1/cameras/${cam.id}`, { method: "DELETE" }),
    onSuccess: refresh,
  });
  const h = cam.last_hour;
  const rate = h.inspected ? h.damaged / h.inspected : null;
  const st = STATE[cam.state];
  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-lg font-bold text-ink">{cam.name}</h2>
          <p className="truncate text-sm text-muted">{cam.line_id ? `Line ${cam.line_id} · ` : ""}<span className="mono">{cam.stream_url}</span></p>
        </div>
        <Badge tone={st.tone}>{st.label}</Badge>
        {cam.simulated && <Badge tone="sim" title="Test stream: its inspections are labelled simulated">Simulated</Badge>}
      </div>
      <LiveView cam={cam} />
      {cam.state === "error" && cam.last_error && <p className="rounded-lg bg-bad-soft px-3 py-2 text-sm text-bad">{cam.last_error}</p>}
      <div>
        <div className="mb-1 flex items-baseline justify-between text-sm">
          <span className="font-semibold text-ink">Last hour</span>
          <span className="text-muted">{h.inspected} inspected · last capture {cam.last_capture_at ? ago(cam.last_capture_at) : "never"}</span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg bg-bad-soft py-1.5"><p className="text-xl font-bold text-bad">{h.damaged}</p><p className="text-xs text-ink-soft">Damaged</p></div>
          <div className="rounded-lg bg-good-soft py-1.5"><p className="text-xl font-bold text-good">{h.not_damaged}</p><p className="text-xs text-ink-soft">Not damaged</p></div>
          <div className="rounded-lg bg-warn-soft py-1.5"><p className="text-xl font-bold text-warn">{h.expert_check}</p><p className="text-xs text-ink-soft">Expert check</p></div>
        </div>
        {rate !== null && (
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-good-soft" title={`Damage rate ${(rate * 100).toFixed(0)}%`}>
            <div className="h-full bg-bad" style={{ width: `${rate * 100}%` }} />
          </div>
        )}
      </div>
      <p className="text-xs text-muted">
        {cam.capture_mode === "on_change" ? "Captures when the scene changes" : "Captures on a fixed interval"}, at most every {cam.interval_s} s ·
        {" "}{cam.captures} captures total
      </p>
      <div className="mt-auto flex flex-wrap items-center gap-2">
        {canInspect && (
          <Button variant="primary" busy={inspect.isPending} disabled={!cam.live} onClick={() => inspect.mutate()}
            title={cam.live ? "Inspect the tyre in front of the camera now" : "Camera is not live"}>
            <ScanLine aria-hidden className="h-4 w-4" /> Inspect now
          </Button>
        )}
        <Link href={`/inspect`} className="inline-flex min-h-11 items-center gap-1.5 rounded-lg px-3 text-sm font-semibold text-ink ring-1 ring-inset ring-ink/70 hover:bg-surface-muted"
          title={`Ask Bib, e.g. "What does ${cam.name} see right now?"`}>
          <MessageSquare aria-hidden className="h-4 w-4" /> Ask Bib
        </Link>
        <Link href={`/inspections?camera_id=${encodeURIComponent(cam.code)}&source=camera`} className="text-sm font-semibold text-primary-ink hover:underline">Inspections →</Link>
        {canManage && (
          <span className="ml-auto flex items-center gap-1">
            <button type="button" onClick={() => toggle.mutate()} disabled={toggle.isPending}
              aria-label={cam.enabled ? `Disable ${cam.name}` : `Enable ${cam.name}`} title={cam.enabled ? "Disable" : "Enable"}
              className={clsx("rounded-lg p-2 hover:bg-surface-muted", cam.enabled ? "text-good" : "text-muted")}>
              <Power className="h-4 w-4" />
            </button>
            <button type="button" aria-label={`Remove ${cam.name}`} title="Remove (its inspections are kept)"
              onClick={() => { if (window.confirm(`Remove ${cam.name}? Its inspections are kept.`)) remove.mutate(); }}
              className="rounded-lg p-2 text-muted hover:bg-bad-soft hover:text-bad">
              <Trash2 className="h-4 w-4" />
            </button>
          </span>
        )}
      </div>
      {inspect.data && (
        <p role="status" className="rounded-lg bg-primary-soft px-3 py-2 text-sm">
          Inspection queued: <Link href={`/inspections/${inspect.data.inspection_id}`} className="font-semibold text-primary-ink hover:underline">open {inspect.data.inspection_id.slice(0, 8)} →</Link>
        </p>
      )}
      {(inspect.error || toggle.error || remove.error) ? <ErrorState error={inspect.error ?? toggle.error ?? remove.error} title="Action failed" /> : null}
    </Card>
  );
}

function AddCamera({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({ name: "", line_id: "", stream_url: "", simulated: false, capture_mode: "on_change", interval_s: 5 });
  const set = (k: keyof typeof form, v: string | number | boolean) => setForm((f) => ({ ...f, [k]: v }));
  const probe = useMutation({
    mutationFn: () => api<CameraProbe>("/v1/cameras/probe", { method: "POST", json: { stream_url: form.stream_url } }),
  });
  const save = useMutation({
    mutationFn: () => api<Camera>("/v1/cameras", { method: "POST", json: { ...form, line_id: form.line_id || null } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: qk.cameras }); onClose(); },
  });
  return (
    <Card className="mb-6" title="Add a camera" subtitle="Any RTSP or HTTP stream the server can reach. The URL is encrypted at rest and shown masked."
      actions={<button type="button" aria-label="Close" onClick={onClose} className="rounded-lg p-2 text-muted hover:bg-surface-muted"><X className="h-5 w-5" /></button>}>
      <form className="grid gap-4 lg:grid-cols-[1fr_minmax(0,320px)]" onSubmit={(e) => { e.preventDefault(); save.mutate(); }}>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Name" htmlFor="cam-name"><input id="cam-name" required minLength={2} maxLength={80} className={inputClass} value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="Line 1 intake" /></Field>
          <Field label="Production line" htmlFor="cam-line"><input id="cam-line" maxLength={64} className={inputClass} value={form.line_id} onChange={(e) => set("line_id", e.target.value)} placeholder="L1" /></Field>
          <div className="sm:col-span-2">
            <Field label="Stream URL" htmlFor="cam-url" hint="rtsp://user:password@host:554/stream, rtsps:// or http(s)://">
              <input id="cam-url" required className={clsx(inputClass, "mono")} value={form.stream_url} onChange={(e) => set("stream_url", e.target.value)} autoComplete="off" spellCheck={false} />
            </Field>
          </div>
          <Field label="Capture" htmlFor="cam-mode">
            <select id="cam-mode" className={inputClass} value={form.capture_mode} onChange={(e) => set("capture_mode", e.target.value)}>
              <option value="on_change">When the scene changes</option>
              <option value="interval">Every interval</option>
            </select>
          </Field>
          <Field label="At most every (s)" htmlFor="cam-int"><input id="cam-int" type="number" min={1} max={3600} className={inputClass} value={form.interval_s} onChange={(e) => set("interval_s", Number(e.target.value))} /></Field>
          <label className="flex items-center gap-2 text-sm text-ink sm:col-span-2">
            <input type="checkbox" checked={form.simulated} onChange={(e) => set("simulated", e.target.checked)} className="h-4 w-4" />
            Simulated test stream (its inspections are labelled simulated)
          </label>
          <div className="flex flex-wrap gap-2 sm:col-span-2">
            <Button type="button" variant="secondary" busy={probe.isPending} disabled={form.stream_url.length < 8} onClick={() => probe.mutate()}>Test connection</Button>
            <Button type="submit" busy={save.isPending}>Save camera</Button>
          </div>
          {(probe.error || save.error) ? <div className="sm:col-span-2"><ErrorState error={probe.error ?? save.error} title={probe.error ? "Connection failed" : "Not saved"} /></div> : null}
        </div>
        <div className="flex flex-col gap-2">
          <div className="aspect-video overflow-hidden rounded-xl bg-navy">
            {probe.data ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={mediaUrl(probe.data.snapshot_url) ?? ""} alt="Frame from the tested stream" className="h-full w-full object-contain" />
            ) : <div className="flex h-full items-center justify-center text-sm text-white/60">Test the connection to see a frame</div>}
          </div>
          {probe.data && <p className="text-sm text-good">Connected: {probe.data.width}×{probe.data.height}{probe.data.fps ? ` · ${probe.data.fps} fps` : ""}</p>}
        </div>
      </form>
    </Card>
  );
}

export function CamerasView() {
  const me = useMe();
  const [adding, setAdding] = useState(false);
  const cams = useQuery({ queryKey: qk.cameras, queryFn: () => api<{ items: Camera[] }>("/v1/cameras"), refetchInterval: 4000 });
  const canManage = hasRole(me.data?.roles, "ml_engineer");
  const canInspect = hasRole(me.data?.roles, "inspector");
  return (
    <>
      <PageHeader eyebrow="Quality checks" title="Line cameras"
        subtitle="RTSP cameras on the production line. The connector inspects the sharpest frame whenever a new tyre is in view; every capture is a normal inspection with the model, lineage and review."
        actions={canManage && !adding ? <Button onClick={() => setAdding(true)}><Plus aria-hidden className="h-4 w-4" /> Add camera</Button> : undefined} />
      {adding && <AddCamera onClose={() => setAdding(false)} />}
      {cams.isError ? <ErrorState error={cams.error} onRetry={() => cams.refetch()} /> : !cams.data ? <SkeletonRows rows={6} />
        : cams.data.items.length === 0 ? (
          <EmptyState title="No cameras yet" icon={<Cctv className="h-7 w-7" />}>
            {canManage ? "Add an RTSP camera above. " : "An ML engineer can add line cameras. "}
            For a demo without hardware, <code className="mono">python scripts/sim_cameras.py up</code> serves two simulated RTSP
            streams built from dataset photos, and <code className="mono">scripts/dev.sh up cameras</code> starts the connector.
          </EmptyState>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 2xl:grid-cols-3">
            {cams.data.items.map((c) => <CameraCard key={c.id} cam={c} canManage={canManage} canInspect={canInspect} />)}
          </div>
        )}
    </>
  );
}
