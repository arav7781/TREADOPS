"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Cctv, MessageSquare, Plus, Power, ScanLine, Trash2, X } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { useMe } from "@/components/AppShell";
import { CameraConnect } from "@/components/domain/CameraConnect";
import { LiveCamera } from "@/components/domain/LiveCamera";
import { Badge, Button, Card, EmptyState, ErrorState, PageHeader, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { ago } from "@/lib/format";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { Camera, InspectionAccepted } from "@/lib/types";

const STATE: Record<Camera["state"], { label: string; tone: "good" | "warn" | "bad" | "neutral" | "info" }> = {
  streaming: { label: "Live", tone: "good" }, connecting: { label: "Connecting", tone: "info" },
  error: { label: "Error", tone: "bad" }, idle: { label: "Waiting for connector", tone: "warn" },
  disabled: { label: "Disabled", tone: "neutral" },
};

function CameraCard({ cam, canManage, canRemove, canInspect }: { cam: Camera; canManage: boolean; canRemove: boolean; canInspect: boolean }) {
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
      <LiveCamera cam={cam} />
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
        {cam.capture_mode === "on_change" ? "Captures each tyre once it stops in the zone" : "Captures the sharpest frame of each interval"}, at most every {cam.interval_s} s ·
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
            {canRemove && (
              <button type="button" aria-label={`Remove ${cam.name}`} title="Remove (its inspections are kept)"
                onClick={() => { if (window.confirm(`Remove ${cam.name}? Its inspections are kept.`)) remove.mutate(); }}
                className="rounded-lg p-2 text-muted hover:bg-bad-soft hover:text-bad">
                <Trash2 className="h-4 w-4" />
              </button>
            )}
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
  return (
    <Card className="mb-6" title="Connect a camera" subtitle="Any RTSP or HTTP stream the server can reach. Draw the inspection zone on a test frame; the URL is encrypted at rest and shown masked."
      actions={<button type="button" aria-label="Close" onClick={onClose} className="rounded-lg p-2 text-muted hover:bg-surface-muted"><X className="h-5 w-5" /></button>}>
      <CameraConnect onClose={onClose} />
    </Card>
  );
}

export function CamerasView() {
  const me = useMe();
  const [adding, setAdding] = useState(false);
  const cams = useQuery({ queryKey: qk.cameras, queryFn: () => api<{ items: Camera[] }>("/v1/cameras"), refetchInterval: 4000 });
  const canManage = hasRole(me.data?.roles, "inspector");    // connect, edit, enable/disable
  const canRemove = hasRole(me.data?.roles, "ml_engineer");
  const canInspect = hasRole(me.data?.roles, "inspector");
  return (
    <>
      <PageHeader eyebrow="Quality checks" title="Line cameras"
        subtitle="RTSP cameras on the production line, live with the AI overlay: the inspection zone in the colour of the latest verdict and the damage locator's boxes on every frame. The connector inspects the sharpest frame whenever a new tyre is in view; every capture is a normal inspection with lineage and review."
        actions={canManage && !adding ? <Button onClick={() => setAdding(true)}><Plus aria-hidden className="h-4 w-4" /> Connect camera</Button> : undefined} />
      {adding && <AddCamera onClose={() => setAdding(false)} />}
      {cams.isError ? <ErrorState error={cams.error} onRetry={() => cams.refetch()} /> : !cams.data ? <SkeletonRows rows={6} />
        : cams.data.items.length === 0 ? (
          <EmptyState title="No cameras yet" icon={<Cctv className="h-7 w-7" />}>
            {canManage ? "Connect an RTSP camera above (or from Bib's chat). " : "An inspector or admin can connect line cameras. "}
            For a demo without hardware, <code className="mono">python scripts/sim_cameras.py up</code> serves two simulated RTSP
            streams built from dataset photos plus a stop-and-go conveyor (line3-conveyor), and <code className="mono">scripts/dev.sh up cameras</code> starts the connector.
          </EmptyState>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 2xl:grid-cols-3">
            {cams.data.items.map((c) => <CameraCard key={c.id} cam={c} canManage={canManage} canRemove={canRemove} canInspect={canInspect} />)}
          </div>
        )}
    </>
  );
}
