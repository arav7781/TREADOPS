"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Bot, ImagePlus, Loader2, Send, ShieldCheck, ThumbsDown, ThumbsUp, User, Wrench, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { Markdown } from "@/components/domain/Markdown";
import { StageGallery } from "@/components/domain/StageGallery";
import { TriageBadge } from "@/components/domain/badges";
import { Badge, Button, ErrorState, inputClass } from "@/components/ui";
import { api, ApiError, uploadWithProgress } from "@/lib/api";
import { humanize, timeOnly } from "@/lib/format";
import { useInspection } from "@/lib/hooks";
import { pollWhile, qk } from "@/lib/query";
import type { ChatMessage, ChatSessionDetail, ChatTurnStarted, CopilotStatus } from "@/lib/types";

export function useCopilotSession(sessionId: string | null) {
  return useQuery({
    queryKey: qk.copilotSession(sessionId ?? "none"),
    queryFn: () => api<ChatSessionDetail>(`/v1/copilot/sessions/${sessionId}`),
    enabled: Boolean(sessionId),
    refetchInterval: pollWhile<ChatSessionDetail>(
      (d) => !d || d.messages.some((m) => m.role === "assistant" && m.attachments?.status === "pending"), 1200),
  });
}

export function useSendTurn(onStarted: (r: ChatTurnStarted) => void) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ text, file, sessionId }: { text: string; file?: File | null; sessionId?: string | null }) => {
      const form = new FormData();
      form.set("message", text);
      if (sessionId) form.set("session_id", sessionId);
      if (file) form.set("file", file);
      return uploadWithProgress<ChatTurnStarted>("/api/v1/copilot/messages", form);
    },
    onSuccess: (r) => {
      onStarted(r);
      qc.invalidateQueries({ queryKey: qk.copilotSessions });
      qc.invalidateQueries({ queryKey: qk.copilotSession(r.session_id) });
    },
  });
}

function LiveInspection({ id }: { id: string }) {
  const q = useInspection(id);
  if (!q.data) return <p className="text-sm text-muted">Loading inspection…</p>;
  const running = q.data.status === "queued" || q.data.status === "running";
  return (
    <div className="mt-3 space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold text-ink">YOLO pipeline</span>
        <Badge tone={running ? "primary" : "good"}>{humanize(q.data.status)}</Badge>
        {q.data.prediction && <TriageBadge triage={q.data.prediction.triage} />}
        <Link href={`/inspections/${id}`} className="font-semibold text-primary-ink hover:underline">Open inspection →</Link>
      </div>
      <StageGallery stages={q.data.stages} running={running} />
    </div>
  );
}

function Feedback({ message }: { message: ChatMessage }) {
  const [score, setScore] = useState<number | null>(null);
  const m = useMutation({
    mutationFn: (s: number) => api(`/v1/copilot/messages/${message.id}/feedback`, { method: "POST", json: { score: s } }),
    onSuccess: (_d, s) => setScore(s),
  });
  return (
    <span className="flex items-center gap-1">
      <button type="button" aria-label="Helpful" aria-pressed={score === 1} disabled={m.isPending}
        onClick={() => m.mutate(1)} className={clsx("rounded-lg p-1.5 hover:bg-surface-muted", score === 1 ? "text-good" : "text-muted")}>
        <ThumbsUp className="h-4 w-4" />
      </button>
      <button type="button" aria-label="Not helpful" aria-pressed={score === -1} disabled={m.isPending}
        onClick={() => m.mutate(-1)} className={clsx("rounded-lg p-1.5 hover:bg-surface-muted", score === -1 ? "text-bad" : "text-muted")}>
        <ThumbsDown className="h-4 w-4" />
      </button>
    </span>
  );
}

function AssistantMessage({ m, onSuggest, showStages }: { m: ChatMessage; onSuggest: (s: string) => void; showStages: boolean }) {
  const a = m.attachments ?? {};
  const pending = a.status === "pending";
  return (
    <div className="flex gap-3">
      <span aria-hidden className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent text-white"><Bot className="h-4 w-4" /></span>
      <div className="min-w-0 flex-1 rounded-2xl rounded-tl-sm border border-line bg-surface px-4 py-3">
        {pending ? (
          <div aria-live="polite">
            <p className="flex items-center gap-2 font-semibold text-ink"><Loader2 aria-hidden className="h-4 w-4 animate-spin text-primary" /> Copilot is working…</p>
            <ol className="mono mt-2 space-y-0.5 text-[13px] text-ink-soft">
              {(a.events ?? []).map((e, i) => (
                <li key={i}><span className="text-muted">{timeOnly(e.at)}</span>{" "}
                  {e.kind === "tool_call" ? <><Wrench aria-hidden className="inline h-3 w-3" /> calling <strong>{e.tool}</strong></> : e.message ?? e.kind}</li>
              ))}
            </ol>
          </div>
        ) : a.status === "error" ? (
          <p className="text-bad">{m.content}</p>
        ) : (
          <Markdown text={m.content} />
        )}
        {showStages && a.inspection_id && <LiveInspection id={a.inspection_id} />}
        {!pending && (
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-2 text-sm">
            {a.mode && <Badge tone={a.mode === "llm" ? "primary" : "neutral"} title="How this answer was produced">{a.mode === "llm" ? "LLM + tools" : "Deterministic fallback"}</Badge>}
            {a.grounding && <Badge tone={a.grounding.pass ? "good" : "warn"}><ShieldCheck aria-hidden className="h-3.5 w-3.5" />
              {a.grounding.pass ? "grounded" : `${a.grounding.ungrounded.length} unverified figure(s)`}</Badge>}
            {(a.guardrails ?? []).filter((g) => g !== "disclaimer").map((g) => <Badge key={g} tone="warn">{humanize(g)}</Badge>)}
            {(a.tools ?? []).length > 0 && <span className="text-muted">tools: {a.tools?.join(", ")}</span>}
            <span className="ml-auto flex items-center gap-2">
              {m.trace_id && <Link href={`/llmops?trace=${m.trace_id}`} className="text-primary-ink hover:underline">trace</Link>}
              <Feedback message={m} />
            </span>
          </div>
        )}
        {!pending && m.suggestions?.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2" aria-label="Suggested follow-ups">
            {m.suggestions.map((s) => (
              <button key={s} type="button" onClick={() => onSuggest(s)}
                className="rounded-full border border-primary/30 bg-primary-soft px-3 py-1 text-sm font-semibold text-primary-ink hover:bg-primary/15">
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function ChatThread({ messages, onSuggest, showStages = true, autoScroll = true }: {
  messages: ChatMessage[];
  onSuggest: (s: string) => void;
  showStages?: boolean;
  /** Embedded threads must not drag the whole page away from what the user is looking at. */
  autoScroll?: boolean;
}) {
  const end = useRef<HTMLDivElement>(null);
  const last = messages[messages.length - 1];
  useEffect(() => {
    if (autoScroll) end.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [autoScroll, messages.length, last?.content, last?.attachments?.events?.length]);
  return (
    <div className="space-y-4">
      {messages.map((m) => m.role === "user" ? (
        <div key={m.id} className="flex justify-end gap-3">
          <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-white">
            {m.content || <span className="italic opacity-80">Photo attached</span>}
            {m.attachments?.inspection_id && <span className="mt-1 block text-sm opacity-80">📷 tyre photo · inspection {m.attachments.inspection_id.slice(0, 8)}</span>}
          </div>
          <span aria-hidden className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink text-white"><User className="h-4 w-4" /></span>
        </div>
      ) : (
        <AssistantMessage key={m.id} m={m} onSuggest={onSuggest} showStages={showStages} />
      ))}
      <div ref={end} />
    </div>
  );
}

export function Composer({ onSend, busy, allowImage, placeholder, error }: {
  onSend: (text: string, file: File | null) => void;
  busy: boolean;
  allowImage: boolean;
  placeholder?: string;
  error?: unknown;
}) {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  const submit = () => {
    if (busy || (!text.trim() && !file)) return;
    onSend(text.trim(), file);
    setText("");
    setFile(null);
    if (fileRef.current) fileRef.current.value = "";
  };
  return (
    <form className="card p-3" onSubmit={(e) => { e.preventDefault(); submit(); }} aria-label="Ask the copilot">
      {preview && (
        <div className="mb-2 flex items-center gap-3 rounded-xl bg-surface-muted p-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={preview} alt="Selected tyre photo preview" className="h-16 w-16 rounded-lg object-cover" />
          <span className="min-w-0 flex-1 truncate text-sm text-ink-soft">{file?.name} · {(file!.size / 1024).toFixed(0)} KB</span>
          <button type="button" aria-label="Remove photo" onClick={() => setFile(null)} className="rounded-lg p-1.5 text-muted hover:bg-line"><X className="h-4 w-4" /></button>
        </div>
      )}
      <div className="flex items-end gap-2">
        {allowImage && (
          <>
            <input ref={fileRef} id="copilot-file" type="file" accept="image/jpeg,image/png,image/webp" className="sr-only"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <label htmlFor="copilot-file" className="flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center rounded-xl border border-line-strong text-ink-soft hover:bg-surface-muted" title="Attach a tyre photo">
              <ImagePlus aria-hidden className="h-5 w-5" /><span className="sr-only">Attach a tyre photo</span>
            </label>
          </>
        )}
        <label htmlFor="copilot-text" className="sr-only">Message</label>
        <textarea id="copilot-text" rows={1} value={text} onChange={(e) => setText(e.target.value)} maxLength={4000}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          placeholder={placeholder ?? "Ask about a tyre, the model, drift or circularity…"} className={clsx(inputClass, "max-h-40 min-h-11 resize-y")} />
        <Button type="submit" busy={busy} aria-label="Send" className="h-11 w-11 shrink-0 px-0"><Send aria-hidden className="h-5 w-5" /></Button>
      </div>
      {error ? <div className="mt-2"><ErrorState error={error} title={error instanceof ApiError && error.status === 403 ? undefined : "Message not sent"} /></div> : null}
    </form>
  );
}

export function CopilotModeBadge() {
  const s = useQuery({ queryKey: qk.copilotStatus, queryFn: () => api<CopilotStatus>("/v1/copilot/status"), staleTime: 60_000 });
  if (!s.data) return null;
  return (
    <Badge tone={s.data.mode === "llm" ? "primary" : "neutral"} title={`prompt ${s.data.prompt_version} · ${s.data.prompt_sha256.slice(0, 12)}`}>
      {s.data.mode === "llm" ? `LLM: ${s.data.model}` : "LLM not configured · deterministic tool-grounded mode"}
    </Badge>
  );
}
