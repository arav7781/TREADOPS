"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { MessageSquarePlus, ShieldCheck } from "lucide-react";

import { ChatThread, Composer, CopilotModeBadge, useCopilotSession, useSendTurn } from "@/components/domain/ChatThread";
import { useMe } from "@/components/AppShell";
import { Button, Card, EmptyState, ErrorState, PageHeader, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { ago } from "@/lib/format";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import { useUrlState } from "@/lib/useUrlState";

const STARTERS = [
  "How reliable is the current model?",
  "How much CO2 does retreading 25 truck casings save?",
  "Is there any drift right now?",
  "What visual cues mean a casing should not be retreaded?",
  "How should I photograph a tyre for inspection?",
];

export function CopilotView() {
  const me = useMe();
  const [{ session }, setUrl] = useUrlState(["session"] as const);
  const sessions = useQuery({ queryKey: qk.copilotSessions, queryFn: () => api<{ items: { id: string; title: string; updated_at: string; inspection_id: string | null }[] }>("/v1/copilot/sessions"), refetchInterval: 30_000 });
  const thread = useCopilotSession(session || null);
  const send = useSendTurn((r) => setUrl({ session: r.session_id }));
  const canUpload = hasRole(me.data?.roles, "inspector");
  const ask = (text: string, file: File | null = null) => send.mutate({ text, file, sessionId: session || null });

  return (
    <>
      <PageHeader eyebrow="Agentic assistant" title="TreadOps Copilot"
        subtitle="A LangGraph agent that classifies attached photos with the deployed YOLO bundle (as a tool), explains the evidence and recommends a circular pathway — behind input and output guardrails, with every turn traced."
        actions={<CopilotModeBadge />} />
      <div className="grid gap-6 xl:grid-cols-[260px_1fr]">
        <Card title="Conversations" actions={<Button variant="ghost" onClick={() => setUrl({ session: "" })} aria-label="New conversation"><MessageSquarePlus className="h-5 w-5" /></Button>}>
          {sessions.isError ? <ErrorState error={sessions.error} onRetry={() => sessions.refetch()} /> : !sessions.data ? <SkeletonRows rows={4} />
            : sessions.data.items.length === 0 ? <p className="text-muted">No conversations yet.</p> : (
              <ul className="space-y-1">
                {sessions.data.items.map((s) => (
                  <li key={s.id}>
                    <button type="button" onClick={() => setUrl({ session: s.id })} aria-current={session === s.id ? "true" : undefined}
                      className={clsx("w-full rounded-xl px-3 py-2 text-left", session === s.id ? "bg-primary-soft text-primary-ink" : "hover:bg-surface-muted")}>
                      <span className="block truncate font-semibold">{s.title}</span>
                      <span className="text-sm text-muted">{ago(s.updated_at)}{s.inspection_id ? " · 📷" : ""}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          <div className="mt-6 rounded-2xl bg-surface-muted p-3 text-sm text-ink-soft">
            <p className="mb-1 flex items-center gap-1.5 font-bold text-ink"><ShieldCheck aria-hidden className="h-4 w-4 text-accent" /> Guardrails</p>
            Blocks prompt injection and off-topic requests, redacts PII, rewrites any safety/certification claim, checks
            every figure against tool outputs and never contradicts the YOLO triage.
          </div>
        </Card>
        <div className="flex min-w-0 flex-col gap-4">
          <Card className="min-h-[420px]">
            {!session ? (
              <EmptyState title="Ask about a tyre, the model, drift or circularity">
                <p className="mb-3">{canUpload ? "Attach a tyre photo to have the deployed YOLO model classify it, then ask follow-ups." : "Viewers can ask questions; attaching photos requires the inspector role."}</p>
                <div className="flex flex-wrap justify-center gap-2">
                  {STARTERS.map((s) => (
                    <button key={s} type="button" onClick={() => ask(s)} className="rounded-full border border-primary/30 bg-primary-soft px-3 py-1 text-sm font-semibold text-primary-ink hover:bg-primary/15">{s}</button>
                  ))}
                </div>
              </EmptyState>
            ) : thread.isError ? <ErrorState error={thread.error} onRetry={() => thread.refetch()} />
              : !thread.data ? <SkeletonRows rows={4} />
                : <ChatThread messages={thread.data.messages} onSuggest={(s) => ask(s)} />}
          </Card>
          <Composer busy={send.isPending} allowImage={canUpload} error={send.error} onSend={(t, f) => ask(t, f)} />
          <p className="text-sm text-muted">Triage aid only — final clearance requires a qualified inspector and the retreader&apos;s full casing inspection.</p>
        </div>
      </div>
    </>
  );
}
