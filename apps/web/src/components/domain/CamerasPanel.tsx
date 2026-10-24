"use client";

import { useQuery } from "@tanstack/react-query";
import { Cctv, MessageSquare, Plus, X } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { CameraConnect } from "@/components/domain/CameraConnect";
import { LiveCamera } from "@/components/domain/LiveCamera";
import { Badge, Button, EmptyState, ErrorState, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { qk } from "@/lib/query";
import type { Camera } from "@/lib/types";

/** Line cameras inside Bib's chat: live views with the AI overlay, "ask Bib" shortcuts and an RTSP connect form. */
export function CamerasPanel({ canConnect, onAsk, onClose }: { canConnect: boolean; onAsk: (text: string) => void; onClose: () => void }) {
  const [connecting, setConnecting] = useState(false);
  const cams = useQuery({ queryKey: qk.cameras, queryFn: () => api<{ items: Camera[] }>("/v1/cameras"), refetchInterval: 3000 });
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-4">
        <Cctv aria-hidden className="h-5 w-5 text-primary" />
        <h2 className="flex-1 font-bold text-ink">Live cameras</h2>
        {canConnect && !connecting && (
          <Button variant="secondary" className="min-h-9 px-3 py-1 text-sm" onClick={() => setConnecting(true)}>
            <Plus aria-hidden className="h-4 w-4" /> Connect
          </Button>
        )}
        <button type="button" aria-label="Close live cameras" onClick={onClose} className="rounded-full p-2 text-muted hover:bg-surface-muted"><X className="h-5 w-5" /></button>
      </div>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {connecting && (
          <div className="rounded-2xl border border-line bg-surface p-4">
            <h3 className="mb-3 font-bold text-ink">Connect a camera (RTSP)</h3>
            <CameraConnect compact onClose={() => setConnecting(false)} />
          </div>
        )}
        {cams.isError ? <ErrorState error={cams.error} onRetry={() => cams.refetch()} />
          : !cams.data ? <SkeletonRows rows={3} />
            : cams.data.items.length === 0 ? (
              <EmptyState title="No cameras connected" icon={<Cctv className="h-6 w-6" />}>
                {canConnect ? "Connect an RTSP camera with the button above." : "An inspector or admin can connect line cameras."}
              </EmptyState>
            ) : [...cams.data.items].sort((a, b) => Number(b.live) - Number(a.live)).map((c) => (
              <section key={c.id} aria-label={c.name} className="space-y-2 rounded-2xl border border-line bg-surface p-3">
                <div className="flex items-center gap-2">
                  <h3 className="min-w-0 flex-1 truncate font-semibold text-ink">{c.name}{c.line_id ? <span className="font-normal text-muted"> · {c.line_id}</span> : null}</h3>
                  {c.simulated && <Badge tone="sim">Simulated</Badge>}
                  {!c.live && <Badge tone={c.state === "error" ? "bad" : "neutral"}>{c.state === "error" ? "Error" : "Offline"}</Badge>}
                </div>
                <LiveCamera cam={c} compact />
                <div className="flex flex-wrap items-center gap-3 text-sm">
                  <button type="button" onClick={() => onAsk(`What does ${c.name} see right now?`)}
                    className="inline-flex items-center gap-1.5 font-semibold text-primary-ink hover:underline">
                    <MessageSquare aria-hidden className="h-4 w-4" /> Ask Bib about it
                  </button>
                  <Link href="/cameras" className="text-muted hover:underline">Settings →</Link>
                </div>
              </section>
            ))}
      </div>
    </div>
  );
}
