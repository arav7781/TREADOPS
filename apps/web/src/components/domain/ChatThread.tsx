"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Brain,
  CheckCircle2,
  ChevronDown,
  ClipboardCheck,
  ImagePlus,
  Layers,
  Leaf,
  Loader2,
  PenLine,
  Recycle,
  ScanLine,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Terminal,
  ThumbsDown,
  ThumbsUp,
  User,
  Wrench,
  X,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Markdown } from "@/components/domain/Markdown";
import { ProbabilityPanel } from "@/components/domain/ProbabilityPanel";
import { ReviewForm } from "@/components/domain/ReviewForm";
import { PipelineConsole, StageTheatre } from "@/components/domain/StageGallery";
import { QualityBadge, StatusBadge, TriageBadge } from "@/components/domain/badges";
import { Badge, Button, ErrorState, ProgressBar } from "@/components/ui";
import { api, ApiError, mediaUrl, uploadWithProgress } from "@/lib/api";
import { ASSISTANT_NAME } from "@/lib/brand";
import { humanize, ms } from "@/lib/format";
import { useInspection } from "@/lib/hooks";
import { pollWhile, qk } from "@/lib/query";
import type { ChatEvent, ChatMessage, ChatSessionDetail, ChatTurnStarted, CopilotStatus, Evidence, InspectionDetail } from "@/lib/types";

export const PHOTO_TYPES = ["image/jpeg", "image/png", "image/webp"];
const SUGGESTED_DISPOSITION: Record<string, string> = {
  good_candidate: "retread", manual_review: "reinspect", defect_flagged: "material_recovery",
};
const PATHWAY_ICON = { retread_intake: Recycle, expert_inspection: Search, material_recovery: Leaf } as const;

export function useCopilotSession(sessionId: string | null) {
  return useQuery({
    queryKey: qk.copilotSession(sessionId ?? "none"),
    queryFn: () => api<ChatSessionDetail>(`/v1/copilot/sessions/${sessionId}`),
    enabled: Boolean(sessionId),
    refetchInterval: pollWhile<ChatSessionDetail>(
      (d) => !d || d.messages.some((m) => m.role === "assistant" && m.attachments?.status === "pending"), 1000),
  });
}

export function useSendTurn(onStarted: (r: ChatTurnStarted) => void) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ text, file, sessionId, onProgress }: {
      text: string; file?: File | null; sessionId?: string | null; onProgress?: (f: number) => void;
    }) => {
      const form = new FormData();
      form.set("message", text);
      if (sessionId) form.set("session_id", sessionId);
      if (file) form.set("file", file);
      return uploadWithProgress<ChatTurnStarted>("/api/v1/copilot/messages", form, { onProgress });
    },
    onSuccess: (r) => {
      onStarted(r);
      qc.invalidateQueries({ queryKey: qk.copilotSessions });
      qc.invalidateQueries({ queryKey: qk.copilotSession(r.session_id) });
    },
  });
}

/** The photo staged in the composer. Kept outside the composer so a drop anywhere on the page can attach it. */
export function usePhotoPicker(blockedReason: string | null) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pick = (f: File | null | undefined) => {
    setError(null);
    if (!f) return;
    if (blockedReason) return setError(blockedReason);
    if (!PHOTO_TYPES.includes(f.type)) return setError(`Unsupported file type ${f.type || "(unknown)"}; use JPEG, PNG or WebP.`);
    setFile(f);
  };
  return { file, error, blockedReason, pick, clear: () => { setFile(null); setError(null); } };
}
export type PhotoPicker = ReturnType<typeof usePhotoPicker>;

export function BibAvatar({ size = "md" }: { size?: "md" | "lg" }) {
  return (
    <span aria-hidden className={clsx("flex shrink-0 items-center justify-center rounded-full bg-navy font-display font-extrabold italic text-brand-yellow ring-brand-yellow/70",
      size === "lg" ? "h-16 w-16 text-2xl ring-[5px]" : "mt-1 h-9 w-9 text-sm ring-2")}>
      <span className={clsx("flex items-center justify-center rounded-full border-2 border-dashed border-white/35",
        size === "lg" ? "h-11 w-11" : "h-7 w-7")}>B</span>
    </span>
  );
}

const STEP: Record<string, { icon: typeof Brain; tone: string }> = {
  start: { icon: Sparkles, tone: "text-primary" },
  guard_input: { icon: ShieldCheck, tone: "text-accent-ink" },
  plan: { icon: Brain, tone: "text-primary" },
  tool_call: { icon: Wrench, tone: "text-ink" },
  tool_progress: { icon: Loader2, tone: "text-muted" },
  inspection: { icon: ScanLine, tone: "text-primary" },
  stage: { icon: Layers, tone: "text-accent-ink" },
  compose: { icon: PenLine, tone: "text-ink" },
  guard_output: { icon: ShieldCheck, tone: "text-accent-ink" },
};

export function eventText(e: ChatEvent): string {
  switch (e.kind) {
    case "tool_call":
      return `calling ${e.tool}(${e.args && e.args !== "{}" ? e.args : ""})`;
    case "tool_result":
      return `${e.tool} → ${e.summary ?? (e.ok ? "ok" : "failed")}`;
    case "inspection":
      return `inspection ${e.inspection_id?.slice(0, 8)} created · ${e.model ?? "model"} · generation ${e.deployment_generation ?? "?"}`;
    default:
      return e.message ?? humanize(e.kind);
  }
}

/** The LangGraph run as it happens: guardrails, planning, tool calls, pipeline stages, composition. */
export function AgentTimeline({ events, pending, defaultOpen = false }: { events: ChatEvent[]; pending: boolean; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  if (!events.length) {
    return pending ? <p className="flex items-center gap-2 text-sm text-muted"><Loader2 aria-hidden className="h-4 w-4 animate-spin" /> Starting the agent…</p> : null;
  }
  const t0 = Date.parse(events[0].at);
  const total = (Date.parse(events[events.length - 1].at) - t0) / 1000;
  const expanded = pending || open;
  // Consecutive pipeline stages read better as one row of chips; the theatre below shows each one in full.
  const rows: ChatEvent[][] = [];
  for (const e of events) {
    const prev = rows[rows.length - 1];
    if (e.kind === "stage" && prev?.[0].kind === "stage") prev.push(e);
    else rows.push([e]);
  }
  return (
    <div className="rounded-xl border border-line bg-surface-muted/60">
      <button type="button" onClick={() => setOpen((o) => !o)} disabled={pending} aria-expanded={expanded}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-semibold text-ink-soft">
        {pending ? <Loader2 aria-hidden className="h-4 w-4 animate-spin text-primary" /> : <CheckCircle2 aria-hidden className="h-4 w-4 text-good" />}
        {pending ? `${ASSISTANT_NAME} is working…` : `Agent steps · ${events.length} events · ${total.toFixed(1)} s`}
        {!pending && <ChevronDown aria-hidden className={clsx("ml-auto h-4 w-4 transition", open && "rotate-180")} />}
      </button>
      {expanded && (
        <ol className="space-y-1 border-t border-line px-3 py-2 text-[13px]" aria-live="polite" aria-label="Agent steps">
          {rows.map((group, i) => {
            const e = group[group.length - 1];
            const meta = STEP[e.kind];
            const Icon = e.kind === "tool_result" ? (e.ok ? CheckCircle2 : XCircle) : meta?.icon ?? Sparkles;
            const tone = e.kind === "tool_result" ? (e.ok ? "text-good" : "text-bad") : meta?.tone ?? "text-muted";
            const last = pending && i === rows.length - 1;
            return (
              <li key={`${group[0].at}-${i}`} data-kind={e.kind} className="log-in flex items-start gap-2">
                <Icon aria-hidden className={clsx("mt-0.5 h-3.5 w-3.5 shrink-0", tone, (last || e.kind === "tool_progress") && pending && "animate-pulse")} />
                {e.kind === "stage" ? (
                  <span className="flex min-w-0 flex-1 flex-wrap gap-1" title={group.map(eventText).join("\n")}>
                    {group.map((g) => (
                      <span key={g.stage} className="pop-in mono rounded-full bg-accent-soft px-2 py-px text-xs text-ink">{g.stage} {g.name} · {ms(g.duration_ms ?? 0)}</span>
                    ))}
                  </span>
                ) : (
                  <span className={clsx("min-w-0 flex-1 break-words", e.kind === "tool_call" ? "mono text-ink-soft" : "text-ink")}>
                    {eventText(e)}
                    {e.kind === "guard_input" && e.status === "blocked" && <Badge tone="bad" className="ml-2">blocked</Badge>}
                  </span>
                )}
                <span className="mono shrink-0 text-muted">+{((Date.parse(e.at) - t0) / 1000).toFixed(1)}s</span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

function ChatResult({ d, canReview }: { d: InspectionDetail; canReview: boolean }) {
  const qc = useQueryClient();
  const [reviewing, setReviewing] = useState(false);
  const p = d.prediction;
  const mvId = d.deployment.model_version_id;
  // Thresholds come from the pinned model version's policy, never from constants in the UI.
  const evidence = useQuery({
    queryKey: qk.evidence(mvId ?? "none"),
    queryFn: () => api<Evidence>(`/v1/models/${mvId}/evidence`),
    enabled: Boolean(mvId && p),
    staleTime: 300_000,
  });
  if (!p) return null;
  const policy = evidence.data?.policy;
  const temperature = d.stages.find((s) => s.name === "inference")?.params?.temperature;
  const circ = p.circularity;
  const Icon = circ ? PATHWAY_ICON[circ.pathway] ?? Recycle : Recycle;
  const classified = mediaUrl(p.classified_image_url);
  return (
    <div className="stage-in mt-3 rounded-2xl border border-line bg-surface p-4" data-testid="chat-result">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-bold text-ink">Result</span>
        <TriageBadge triage={p.triage} large />
        <QualityBadge status={p.quality_status} reasons={p.quality.reasons} />
      </div>
      <p className="mt-1 text-sm text-ink-soft">{p.triage_reason}</p>
      <div className="mt-3 grid gap-4 md:grid-cols-[minmax(0,220px)_1fr]">
        {classified && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={classified} alt="Classified tyre image with the triage overlay" className="glow-once w-full rounded-xl border border-line object-contain" />
        )}
        <div className="min-w-0 space-y-3">
          <ProbabilityPanel raw={p.raw_probs} calibrated={p.calibrated_probs} pDefective={p.p_defective}
            temperature={typeof temperature === "number" ? temperature : null}
            thresholds={policy ? { low: policy.triage_low, high: policy.triage_high, binary: policy.binary_threshold } : undefined} />
          <p className="mono text-sm text-muted">
            preprocess {ms(p.timing.preprocess_ms)} · YOLO {ms(p.timing.inference_ms)} · attribution {ms(p.timing.attribution_ms)} · worker {ms(p.timing.worker_total_ms)}
          </p>
        </div>
      </div>
      {circ && (
        <div className="mt-3 flex items-start gap-3 rounded-xl bg-accent-soft/60 p-3">
          <span aria-hidden className="rounded-lg bg-surface p-2 text-accent-ink"><Icon className="h-5 w-5" /></span>
          <div className="min-w-0">
            <p className="font-bold text-ink">{circ.label}</p>
            <p className="text-sm text-ink-soft">{circ.action}</p>
            {circ.estimate_if_verified && (
              <p className="mt-1 text-sm text-ink">
                <Badge tone="good">ESTIMATE</Badge>{" "}
                <strong>{circ.estimate_if_verified.raw_material_saved_kg} kg</strong> raw material ·{" "}
                <strong>{circ.estimate_if_verified.co2e_avoided_kg} kg</strong> CO₂e if an inspector verifies the retread
              </p>
            )}
          </div>
        </div>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-sm font-semibold">
        <Link href={`/inspections/${d.inspection_id}`} className="text-primary-ink hover:underline">Full inspection &amp; lineage →</Link>
        <Link href={`/circularity/passport/${d.inspection_id}`} className="text-primary-ink hover:underline">Casing passport →</Link>
        {d.review.revision > 0 && <Badge tone="good">Reviewed r{d.review.revision} · {humanize(d.review.label ?? "")}{d.review.disposition ? ` · ${humanize(d.review.disposition)}` : ""}</Badge>}
        {canReview && (
          <Button variant="ghost" className="ml-auto" onClick={() => setReviewing((r) => !r)} aria-expanded={reviewing}>
            <ClipboardCheck aria-hidden className="h-4 w-4" /> {reviewing ? "Close review" : "Record inspector review"}
          </Button>
        )}
      </div>
      {canReview && reviewing && (
        <div className="mt-3 border-t border-line pt-3">
          <ReviewForm key={d.review.revision} inspectionId={d.inspection_id} expectedRevision={d.review.revision}
            current={{ label: d.review.label, disposition: d.review.disposition }}
            suggestedDisposition={SUGGESTED_DISPOSITION[p.triage] ?? null}
            onSaved={() => qc.invalidateQueries({ queryKey: qk.inspection(d.inspection_id) })} />
        </div>
      )}
    </div>
  );
}

/** The YOLO run the agent's tool started: stage images, the worker's log and the final classified result. */
function InspectionInChat({ id, canReview, animate, revealed, onRevealed }: {
  id: string;
  canReview: boolean;
  animate: boolean;
  revealed: boolean;
  onRevealed: () => void;
}) {
  const q = useInspection(id);
  useEffect(() => { if (q.isError) onRevealed(); }, [q.isError, onRevealed]);
  const [showLog, setShowLog] = useState<boolean | null>(null); // null: open while running, closed after
  if (q.isError) return <div className="mt-3"><ErrorState error={q.error} onRetry={() => q.refetch()} /></div>;
  const d = q.data;
  const running = !d || d.status === "queued" || d.status === "running";
  const logOpen = showLog ?? running;
  return (
    <section className="mt-3 space-y-3" aria-label="YOLO inspection pipeline">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <ScanLine aria-hidden className="h-4 w-4 text-primary" />
        <span className="font-semibold text-ink">YOLO pipeline</span>
        <StatusBadge status={d?.status ?? "queued"} />
        {d && <span className="mono text-muted">{d.deployment.model_family} v{d.deployment.registry_version ?? "?"} · gen {d.deployment.generation ?? "?"} · {id.slice(0, 8)}</span>}
        <button type="button" onClick={() => setShowLog(!logOpen)} aria-expanded={logOpen}
          className="ml-auto flex items-center gap-1 rounded-full px-2.5 py-1 font-semibold text-ink-soft hover:bg-surface-muted">
          <Terminal aria-hidden className="h-4 w-4" /> {logOpen ? "Hide log" : "Show log"}
        </button>
      </div>
      <StageTheatre stages={d?.stages ?? []} running={running} stepMs={animate ? 850 : 0} onRevealed={onRevealed} />
      {logOpen && <PipelineConsole stages={d?.stages ?? []} timeline={d?.timeline} status={d?.status ?? "queued"} error={d?.job?.error_message ?? null} />}
      {d?.status === "failed" && <ErrorState error={new Error(d.job?.error_message ?? "The inspection job failed.")} title={`Inspection failed (${d.job?.error_code ?? "error"})`} />}
      {d?.status === "succeeded" && revealed && <ChatResult d={d} canReview={canReview} />}
    </section>
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

function AssistantMessage({ m, onSuggest, canReview, latest }: { m: ChatMessage; onSuggest: (s: string) => void; canReview: boolean; latest: boolean }) {
  const a = m.attachments ?? {};
  const pending = a.status === "pending";
  // A turn watched live plays the pipeline first and reveals the answer after it; history renders at once.
  const [live] = useState(pending);
  const [revealed, setRevealed] = useState(!pending);
  const markRevealed = useCallback(() => setRevealed(true), []);
  const showAnswer = !pending && (!a.inspection_id || revealed);
  return (
    <div className="flex gap-3" data-role="assistant">
      <BibAvatar />
      <div className="min-w-0 flex-1 space-y-3 rounded-2xl rounded-tl-sm border border-line bg-surface px-4 py-3">
        <p className="text-sm font-bold text-ink">{ASSISTANT_NAME}</p>
        <AgentTimeline events={a.events ?? []} pending={pending} defaultOpen={latest} />
        {a.inspection_id && <InspectionInChat id={a.inspection_id} canReview={canReview} animate={live} revealed={revealed} onRevealed={markRevealed} />}
        {!pending && !showAnswer && (
          <p className="flex items-center gap-2 text-sm text-muted"><Loader2 aria-hidden className="h-4 w-4 animate-spin" /> {ASSISTANT_NAME} is walking through the pipeline…</p>
        )}
        {!showAnswer ? null : a.status === "error" ? <p className="text-bad">{m.content}</p> : <div className="stage-in"><Markdown text={m.content} /></div>}
        {showAnswer && (
          <div className="flex flex-wrap items-center gap-2 border-t border-line pt-2 text-sm">
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
        {showAnswer && m.suggestions?.length > 0 && (
          <div className="flex flex-wrap gap-2" role="group" aria-label="Suggested follow-ups">
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

function UserMessage({ m }: { m: ChatMessage }) {
  const photo = m.attachments?.photo;
  const src = mediaUrl(photo?.url);
  return (
    <div className="flex justify-end gap-3" data-role="user">
      <div className="max-w-[85%] space-y-2">
        {photo && (
          <figure className="ml-auto w-fit overflow-hidden rounded-2xl rounded-tr-sm border border-line bg-surface">
            {src && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={src} alt="Tyre photo you sent" className="max-h-72 w-auto object-contain" />
            )}
            <figcaption className="mono px-3 py-1.5 text-xs text-muted">
              {photo.width}×{photo.height} · {(photo.bytes / 1024).toFixed(0)} KB · sha {photo.sha256.slice(0, 8)}
              {photo.metadata_removed ? " · EXIF stripped" : ""}{photo.gps_removed ? " · GPS removed" : ""}
            </figcaption>
          </figure>
        )}
        {(m.content || !photo) && (
          <div className="ml-auto w-fit rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-white">
            {m.content || <span className="italic opacity-80">Photo attached</span>}
          </div>
        )}
      </div>
      <span aria-hidden className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink text-white"><User className="h-4 w-4" /></span>
    </div>
  );
}

export function ChatThread({ messages, onSuggest, canReview = false, autoScroll = true }: {
  messages: ChatMessage[];
  onSuggest: (s: string) => void;
  canReview?: boolean;
  /** Embedded threads must not drag the whole page away from what the user is looking at. */
  autoScroll?: boolean;
}) {
  const end = useRef<HTMLDivElement>(null);
  const last = messages[messages.length - 1];
  const latestAssistant = messages.findLast((m) => m.role === "assistant")?.id;
  useEffect(() => {
    if (autoScroll) end.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [autoScroll, messages.length, last?.content, last?.attachments?.events?.length]);
  return (
    <div className="space-y-5">
      {messages.map((m) => m.role === "user"
        ? <UserMessage key={m.id} m={m} />
        : <AssistantMessage key={m.id} m={m} onSuggest={onSuggest} canReview={canReview} latest={m.id === latestAssistant} />)}
      <div ref={end} />
    </div>
  );
}

export function Composer({ onSend, busy, photo, progress, placeholder, error }: {
  onSend: (text: string, file: File | null) => void;
  busy: boolean;
  photo: PhotoPicker;
  progress?: number | null;
  placeholder?: string;
  error?: unknown;
}) {
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { // grow with the text up to ~6 lines, shrink back after sending
    const t = textRef.current;
    if (!t) return;
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 160)}px`;
  }, [text]);
  const { file } = photo;
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  const submit = () => {
    if (busy || (!text.trim() && !file)) return;
    onSend(text.trim(), file);
    setText("");
    photo.clear();
    if (fileRef.current) fileRef.current.value = "";
  };
  const blocked = photo.blockedReason;
  return (
    <form className="rounded-2xl border border-line-strong bg-surface p-2 shadow-[0_8px_30px_rgb(0_32_91/0.08)] focus-within:border-primary" onSubmit={(e) => { e.preventDefault(); submit(); }} aria-label={`Chat composer for ${ASSISTANT_NAME}`}
      onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); photo.pick(e.dataTransfer.files?.[0]); }}>
      {preview && (
        <div className="mb-2 flex items-center gap-3 rounded-xl bg-surface-muted p-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={preview} alt="Selected tyre photo preview" className="h-16 w-16 rounded-lg object-cover" />
          <span className="min-w-0 flex-1 text-sm text-ink-soft">
            <span className="block truncate font-semibold text-ink">{file?.name}</span>
            {((file?.size ?? 0) / 1024).toFixed(0)} KB · {ASSISTANT_NAME} will run the YOLO tool on it
          </span>
          <button type="button" aria-label="Remove photo" onClick={() => { photo.clear(); if (fileRef.current) fileRef.current.value = ""; }}
            className="rounded-lg p-1.5 text-muted hover:bg-line"><X className="h-4 w-4" /></button>
        </div>
      )}
      <div className="flex items-end gap-2">
        <input ref={fileRef} id="chat-photo" type="file" accept={PHOTO_TYPES.join(",")} className="sr-only" disabled={Boolean(blocked)}
          onChange={(e) => photo.pick(e.target.files?.[0])} />
        <label htmlFor="chat-photo" aria-disabled={Boolean(blocked)} title={blocked ?? "Attach a tyre photo (or drop / paste one)"}
          className={clsx("flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-primary",
            blocked ? "cursor-not-allowed opacity-40" : "cursor-pointer hover:bg-primary-soft")}>
          <ImagePlus aria-hidden className="h-5 w-5" /><span className="sr-only">Attach a tyre photo</span>
        </label>
        <label htmlFor="chat-text" className="sr-only">Message</label>
        <textarea ref={textRef} id="chat-text" rows={1} value={text} onChange={(e) => setText(e.target.value)} maxLength={4000}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          onPaste={(e) => { const f = e.clipboardData.files?.[0]; if (f) { e.preventDefault(); photo.pick(f); } }}
          placeholder={placeholder ?? (file ? `Ask ${ASSISTANT_NAME} about this tyre (optional)…` : `Ask ${ASSISTANT_NAME}, or attach a tyre photo…`)}
          className="max-h-40 min-h-11 w-full resize-none bg-transparent px-2 py-2.5 text-base text-ink outline-none placeholder:text-muted focus-visible:outline-none" />
        <Button type="submit" busy={busy} aria-label="Send" className="h-11 w-11 shrink-0 px-0!">{busy ? null : <Send aria-hidden className="h-5 w-5" />}</Button>
      </div>
      {progress != null && file === null && busy && (
        <div className="mt-2" aria-live="polite"><ProgressBar value={progress} label="Photo upload progress" /></div>
      )}
      {photo.error && <p className="mt-2 text-sm font-semibold text-bad" role="alert">{photo.error}</p>}
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
