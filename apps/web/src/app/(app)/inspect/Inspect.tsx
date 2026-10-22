"use client";

import { useQuery } from "@tanstack/react-query";
import { Bot, Sparkles } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { ChatThread, Composer, CopilotModeBadge, useCopilotSession, useSendTurn } from "@/components/domain/ChatThread";
import { InspectionUploader } from "@/components/domain/InspectionUploader";
import { ResultPanel } from "@/components/domain/ResultPanel";
import { PipelineConsole, StageGallery } from "@/components/domain/StageGallery";
import { SourceBadge, StatusBadge } from "@/components/domain/badges";
import { Card, EmptyState, ErrorState, PageHeader, SkeletonRows } from "@/components/ui";
import { useMe } from "@/components/AppShell";
import { api } from "@/lib/api";
import { useInspection } from "@/lib/hooks";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { ModelsResponse } from "@/lib/types";
import { useUrlState } from "@/lib/useUrlState";

function CopilotSuggestions({ inspectionId }: { inspectionId: string }) {
  const storageKey = `treadops:copilot:${inspectionId}`;
  const [sessionId, setSessionId] = useState<string | null>(() => {
    try { return typeof window === "undefined" ? null : sessionStorage.getItem(storageKey); } catch { return null; }
  });
  const started = useRef(false);
  const send = useSendTurn((r) => {
    setSessionId(r.session_id);
    try { sessionStorage.setItem(storageKey, r.session_id); } catch { /* storage unavailable */ }
  });
  const session = useCopilotSession(sessionId);

  // One automatic turn per inspection: explain the result and suggest next steps for the inspector.
  useEffect(() => {
    if (started.current || sessionId) return;
    started.current = true;
    send.mutate({ text: `Explain inspection ${inspectionId} and recommend next steps for the inspector.` });
  }, [inspectionId, send, sessionId]);

  return (
    <Card title={<span className="flex items-center gap-2"><Sparkles aria-hidden className="h-5 w-5 text-primary" /> Copilot suggestions</span>}
      subtitle="Guarded agent: grounded in the stored inspection and knowledge base; it never certifies a tyre."
      actions={<><CopilotModeBadge /><Link href={sessionId ? `/copilot?session=${sessionId}` : "/copilot"} className="text-sm font-semibold text-primary-ink hover:underline">Open in Copilot →</Link></>}>
      {send.isError && !sessionId ? <ErrorState error={send.error} title="Copilot unavailable" onRetry={() => send.mutate({ text: `Explain inspection ${inspectionId} and recommend next steps for the inspector.` })} />
        : !session.data ? <SkeletonRows rows={3} />
          : <ChatThread messages={session.data.messages.filter((m) => m.role === "assistant" || m.content)} showStages={false}
            onSuggest={(s) => send.mutate({ text: s, sessionId })} />}
      {sessionId && (
        <div className="mt-4">
          <Composer busy={send.isPending} allowImage={false} error={send.error}
            placeholder="Ask a follow-up about this tyre…" onSend={(t) => send.mutate({ text: t, sessionId })} />
        </div>
      )}
    </Card>
  );
}

export function Inspect() {
  const [{ id }, setUrl] = useUrlState(["id"] as const);
  const me = useMe();
  const models = useQuery({ queryKey: qk.models, queryFn: () => api<ModelsResponse>("/v1/models") });
  const insp = useInspection(id || null);
  const canUpload = hasRole(me.data?.roles, "inspector");
  const untrained = models.data?.state === "untrained";
  const d = insp.data;
  const running = d ? d.status === "queued" || d.status === "running" : Boolean(id);

  return (
    <>
      <PageHeader eyebrow="Inspector workflow" title="Inspect a tyre"
        subtitle="Upload one photo. Watch every preprocessing stage the YOLO bundle actually ran, with real timings, then the calibrated triage, the circular pathway and copilot guidance." />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,420px)_1fr]">
        <Card title="1 · Upload" subtitle="Accepted asynchronously: 202 + job, then the worker classifies">
          {models.isError ? <ErrorState error={models.error} onRetry={() => models.refetch()} /> : (
            <InspectionUploader
              disabled={!canUpload || untrained}
              disabledReason={!canUpload ? "Uploading requires the inspector role." : untrained ? "No trained model available — promote a validated bundle first." : undefined}
              onAccepted={(a) => setUrl({ id: a.inspection_id })} />
          )}
        </Card>
        <div className="flex min-w-0 flex-col gap-6">
          {!id ? (
            <Card title="2 · Pipeline">
              <EmptyState title="No inspection yet" icon={<Bot className="h-7 w-7" />}>
                After upload you will see: ingest → canonical → letterbox → tensor → quality → YOLO inference →
                attribution → classified image.
              </EmptyState>
            </Card>
          ) : insp.isError ? (
            <ErrorState error={insp.error} onRetry={() => insp.refetch()} />
          ) : (
            <>
              <Card title="2 · Pipeline" subtitle={d ? `inspection ${d.inspection_id.slice(0, 8)} · model ${d.deployment.model_family ?? ""} v${d.deployment.registry_version ?? "?"} (generation ${d.deployment.generation ?? "?"})` : "loading…"}
                actions={d && <><StatusBadge status={d.status} /><SourceBadge origin={d.metadata_origin} />
                  <Link href={`/inspections/${d.inspection_id}`} className="text-sm font-semibold text-primary-ink hover:underline">Details →</Link></>}>
                <PipelineConsole stages={d?.stages ?? []} timeline={d?.timeline} status={d?.status ?? "queued"}
                  error={d?.job?.error_message ?? null} />
                <div className="mt-4">
                  <StageGallery stages={d?.stages ?? []} running={running} />
                </div>
                {d?.status === "failed" && <div className="mt-4"><ErrorState error={new Error(d.job?.error_message ?? "The inspection job failed.")} title={`Inspection failed (${d.job?.error_code ?? "error"})`} /></div>}
              </Card>
            </>
          )}
        </div>
      </div>
      {d?.status === "succeeded" && d.prediction && (
        <div className="mt-6 space-y-6">
          <ResultPanel d={d} />
          <CopilotSuggestions key={d.inspection_id} inspectionId={d.inspection_id} />
        </div>
      )}
    </>
  );
}
