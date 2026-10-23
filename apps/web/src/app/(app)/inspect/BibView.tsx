"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowDown, Camera, ChevronDown, History, ImagePlus, MessageSquare, Plus, Search, ShieldCheck, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ChatThread, Composer, CopilotModeBadge, useCopilotSession, usePhotoPicker, useSendTurn } from "@/components/domain/ChatThread";
import { useMe } from "@/components/AppShell";
import { Button, ErrorState, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { ASSISTANT_DISCLAIMER, ASSISTANT_NAME } from "@/lib/brand";
import { ago } from "@/lib/format";
import { useModels } from "@/lib/hooks";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { CopilotStatus } from "@/lib/types";
import { useUrlState } from "@/lib/useUrlState";

interface SessionRow { id: string; title: string; updated_at: string; inspection_id: string | null }

const STARTERS = [
  "How reliable is the current model?",
  "How much CO2 does retreading 25 truck casings save?",
  "Is there any drift right now?",
  "What visual cues mean a casing should not be retreaded?",
];

const GRAPH = [
  ["guard_input", "PII redaction, injection and off-topic checks"],
  ["plan", "the LLM picks tools (or the deterministic planner)"],
  ["tools", "classify_tyre_image runs the pinned YOLO bundle"],
  ["guard_output", "no certification claims; every figure grounded"],
  ["suggest", "next steps for the inspector"],
] as const;

/**
 * Follow new content while the reader is at the bottom. Only the reader can unpin (wheel, touch, keys, scrollbar):
 * smooth and programmatic scrolls pass through intermediate positions and must not stop the follow. New content
 * that arrives while unpinned lights the jump button instead.
 */
function useStickToBottom(resetKey: string) {
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const userInputAt = useRef(-Infinity);
  const [atBottom, setAtBottom] = useState(true);
  const [unread, setUnread] = useState(false);
  useEffect(() => {
    const s = scroller.current;
    const c = content.current;
    if (!s || !c) return;
    const markUser = () => { userInputAt.current = performance.now(); };
    const onScroll = () => {
      const near = s.scrollHeight - s.scrollTop - s.clientHeight < 80;
      if (near) {
        pinned.current = true;
        setUnread(false);
      } else if (performance.now() - userInputAt.current < 1000) {
        pinned.current = false;
      }
      setAtBottom(near || pinned.current);
    };
    const onGrow = () => {
      if (pinned.current) s.scrollTop = s.scrollHeight;
      else setUnread(true);
    };
    const opts = { passive: true } as const;
    s.addEventListener("scroll", onScroll, opts);
    const inputs = ["wheel", "touchmove", "pointerdown", "keydown"] as const;
    for (const ev of inputs) s.addEventListener(ev, markUser, opts);
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(onGrow);
    ro?.observe(c);
    return () => {
      s.removeEventListener("scroll", onScroll);
      for (const ev of inputs) s.removeEventListener(ev, markUser);
      ro?.disconnect();
    };
  }, []);
  useEffect(() => { // a conversation opens at its latest message; the welcome screen opens at the top
    pinned.current = Boolean(resetKey);
    scroller.current?.scrollTo({ top: resetKey ? scroller.current.scrollHeight : 0 });
  }, [resetKey]);
  const jump = () => {
    pinned.current = true;
    setUnread(false);
    setAtBottom(true);
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  };
  return { scroller, content, atBottom, unread, jump };
}

function groupByDay(rows: SessionRow[]): [string, SessionRow[]][] {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const day = 86_400_000;
  const groups = new Map<string, SessionRow[]>();
  for (const r of rows) {
    const t = Date.parse(r.updated_at);
    const label = t >= startOfToday.getTime() ? "Today" : t >= startOfToday.getTime() - day ? "Yesterday"
      : t >= startOfToday.getTime() - 7 * day ? "Previous 7 days" : "Older";
    groups.set(label, [...(groups.get(label) ?? []), r]);
  }
  return [...groups.entries()];
}

function HistorySidebar({ rows, error, onRetry, active, onPick, onNew }: {
  rows: SessionRow[] | undefined;
  error: unknown;
  onRetry: () => void;
  active: string;
  onPick: (id: string) => void;
  onNew: () => void;
}) {
  const [filter, setFilter] = useState("");
  const groups = useMemo(() => groupByDay((rows ?? []).filter((r) => r.title.toLowerCase().includes(filter.trim().toLowerCase()))), [rows, filter]);
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="space-y-3 p-4">
        <Button variant="brand" className="w-full" onClick={onNew}><Plus aria-hidden className="h-5 w-5" /> New chat</Button>
        <label className="relative block">
          <span className="sr-only">Search conversations</span>
          <Search aria-hidden className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Search conversations"
            className="h-11 w-full rounded-lg border border-line-strong bg-surface pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-primary" />
        </label>
      </div>
      <nav aria-label="Conversation history" className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {error ? <div className="px-2"><ErrorState error={error} onRetry={onRetry} /></div> : !rows ? <div className="px-2"><SkeletonRows rows={5} /></div>
          : groups.length === 0 ? <p className="px-3 text-sm text-muted">{filter ? "No conversation matches." : "No conversations yet."}</p>
            : groups.map(([label, items]) => (
              <div key={label} className="mb-3">
                <p className="px-3 pb-1 text-xs font-bold uppercase tracking-[0.12em] text-muted">{label}</p>
                <ul className="space-y-0.5">
                  {items.map((s) => {
                    const Icon = s.inspection_id ? Camera : MessageSquare;
                    const on = active === s.id;
                    return (
                      <li key={s.id}>
                        <button type="button" onClick={() => onPick(s.id)} aria-current={on ? "true" : undefined}
                          className={clsx("flex w-full items-start gap-2.5 rounded-lg px-3 py-2 text-left transition",
                            on ? "bg-accent-soft text-ink shadow-[inset_3px_0_0_var(--brand-yellow)]" : "text-ink-soft hover:bg-surface-muted")}>
                          <Icon aria-hidden className={clsx("mt-0.5 h-4 w-4 shrink-0", s.inspection_id ? "text-primary" : "text-muted")} />
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-semibold">{s.title}</span>
                            <span className="block text-xs text-muted">{ago(s.updated_at)}{s.inspection_id ? " · inspected" : ""}</span>
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
      </nav>
      <details className="group border-t border-line px-4 py-3 text-sm">
        <summary className="flex cursor-pointer list-none items-center gap-2 font-semibold text-ink">
          <ShieldCheck aria-hidden className="h-4 w-4 text-primary" /> How {ASSISTANT_NAME} works
          <ChevronDown aria-hidden className="ml-auto h-4 w-4 transition group-open:rotate-180" />
        </summary>
        <ol className="mt-2 space-y-1.5">
          {GRAPH.map(([node, what], i) => (
            <li key={node} className="flex gap-2 text-xs">
              <span className="mono flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary-soft font-bold text-primary-ink">{i + 1}</span>
              <span><span className="mono font-semibold text-ink">{node}</span> <span className="text-ink-soft">{what}</span></span>
            </li>
          ))}
        </ol>
        <p className="mt-2 text-xs text-muted">It never certifies a tyre as safe and never contradicts the YOLO triage.</p>
      </details>
    </div>
  );
}

function Welcome({ canAttach, blockedReason, onAsk }: { canAttach: boolean; blockedReason: string | null; onAsk: (s: string) => void }) {
  return (
    <div className="mx-auto flex min-h-[52vh] max-w-2xl flex-col items-center justify-center gap-6 text-center">
      <h2 className="text-primary">
        <span className="headline block text-4xl sm:text-5xl">Hey, I&apos;m {ASSISTANT_NAME}</span>
        <span className="headline headline-light block text-3xl sm:text-4xl">Here to help you inspect</span>
      </h2>
      <p className="max-w-md text-ink-soft">
        {canAttach
          ? <>Attach a tyre photo <ImagePlus aria-label="with the photo button" className="mx-0.5 inline h-4 w-4 align-[-2px] text-primary" /> or drop it here. I&apos;ll run the YOLO inspection step by step and suggest what to do with the casing.</>
          : <>Ask about the model, drift or circularity.{blockedReason ? <span className="mt-1 block text-sm text-warn">{blockedReason}</span> : null}</>}
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        {STARTERS.slice(0, 3).map((s) => (
          <button key={s} type="button" onClick={() => onAsk(s)}
            className="rounded-full border border-line-strong bg-surface px-3.5 py-1.5 text-sm text-ink-soft transition hover:border-ink hover:text-ink">{s}</button>
        ))}
      </div>
    </div>
  );
}

/** Inspect + copilot in one chat window: the photo goes into the chat and the agent's YOLO tool does the inspection. */
export function BibView() {
  const me = useMe();
  const models = useModels();
  const [{ session }, setUrl] = useUrlState(["session"] as const);
  const llm = useQuery({ queryKey: qk.copilotStatus, queryFn: () => api<CopilotStatus>("/v1/copilot/status"), staleTime: 60_000 });
  const unavailable = llm.data && !llm.data.configured
    ? "Bib needs a language model: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env and restart the API." : null;
  const sessions = useQuery({ queryKey: qk.copilotSessions, queryFn: () => api<{ items: SessionRow[] }>("/v1/copilot/sessions"), refetchInterval: 30_000 });
  const thread = useCopilotSession(session || null);
  const send = useSendTurn((r) => setUrl({ session: r.session_id }));
  const [progress, setProgress] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const { scroller, content, atBottom, unread, jump } = useStickToBottom(session);
  const isInspector = hasRole(me.data?.roles, "inspector");
  const photoBlocked = !me.data ? "Loading your role…" : !isInspector ? "Attaching tyre photos requires the inspector role."
    : models.data?.state === "untrained" ? "No trained model is active yet: promote a validated bundle first." : null;
  const photo = usePhotoPicker(photoBlocked);
  const ask = (text: string, file: File | null = null) => {
    setProgress(file ? 0 : null);
    jump();
    send.mutate({ text, file, sessionId: session || null, onProgress: setProgress }, { onSettled: () => setProgress(null) });
  };
  const pick = (id: string) => { setUrl({ session: id }); setDrawer(false); };
  const startNew = () => { setUrl({ session: "" }); setDrawer(false); };
  const title = sessions.data?.items.find((s) => s.id === session)?.title;
  // The chat owns the viewport while it is shown. An effect (not CSS :has) so a copy kept alive in a hidden
  // <Activity> after navigating away never changes other pages.
  useEffect(() => {
    document.documentElement.classList.add("chat-mode");
    return () => document.documentElement.classList.remove("chat-mode");
  }, []);
  const sidebar = (
    <HistorySidebar rows={sessions.data?.items} error={sessions.isError ? sessions.error : null} onRetry={() => sessions.refetch()}
      active={session} onPick={pick} onNew={startNew} />
  );

  return (
    <div className="fixed inset-x-0 bottom-0 top-[var(--header-h)] z-20 flex bg-canvas">
      <aside aria-label="Conversations" className="hidden w-72 shrink-0 border-r border-line bg-surface md:block">{sidebar}</aside>
      {drawer && (
        <div className="fixed inset-x-0 bottom-0 top-[var(--header-h)] z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Conversations">
          <button type="button" aria-label="Close conversations" className="absolute inset-0 bg-black/50" onClick={() => setDrawer(false)} />
          <div className="stage-in absolute inset-y-0 left-0 w-[86%] max-w-80 bg-surface shadow-xl">
            <button type="button" aria-label="Close" onClick={() => setDrawer(false)} className="absolute right-2 top-2 rounded-full p-2 text-muted hover:bg-surface-muted"><X className="h-5 w-5" /></button>
            <div className="h-full pt-8">{sidebar}</div>
          </div>
        </div>
      )}

      <section aria-label={`Chat with ${ASSISTANT_NAME}`} className="relative flex min-w-0 flex-1 flex-col"
        onDragEnter={(e) => { if (e.dataTransfer.types.includes("Files")) setDragging(true); }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false); }}
        onDrop={(e) => { e.preventDefault(); setDragging(false); photo.pick(e.dataTransfer.files?.[0]); }}>
        <header className="flex h-14 items-center gap-3 border-b border-line bg-surface px-4 lg:px-6">
          <button type="button" onClick={() => setDrawer(true)} aria-label="Show conversations" className="rounded-full p-2 text-ink-soft hover:bg-surface-muted md:hidden">
            <History className="h-5 w-5" />
          </button>
          <div className="flex min-w-0 flex-1 items-baseline gap-3">
            <h1 className="headline shrink-0 text-lg text-primary">{ASSISTANT_NAME} · casing copilot</h1>
            {title && <p className="truncate text-sm text-muted">{title}</p>}
          </div>
          <span className="hidden lg:inline-flex"><CopilotModeBadge /></span>
        </header>

        <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <div ref={content} className="mx-auto w-full max-w-4xl px-4 py-6 lg:px-6">
            {!session ? <Welcome canAttach={!photoBlocked} blockedReason={me.data ? photoBlocked : null} onAsk={(s) => ask(s)} />
              : thread.isError ? <ErrorState error={thread.error} onRetry={() => thread.refetch()} />
                : !thread.data ? <SkeletonRows rows={4} />
                  : <ChatThread messages={thread.data.messages} canReview={isInspector} autoScroll={false} sessionId={session} onSuggest={(s) => ask(s)} />}
          </div>
        </div>

        {!atBottom && (
          <button type="button" onClick={jump} aria-label="Jump to the latest message"
            className="pop-in absolute bottom-32 right-5 z-10 flex h-10 w-10 items-center justify-center rounded-full border border-line-strong bg-surface text-ink shadow-md hover:bg-surface-muted">
            <ArrowDown aria-hidden className="h-5 w-5" />
            {unread && <span aria-hidden className="pulse-dot absolute -right-0.5 -top-0.5 h-3 w-3 rounded-full border-2 border-surface bg-brand-yellow" />}
          </button>
        )}

        <div className="px-4 pb-3 pt-2 lg:px-6">
          <div className="mx-auto w-full max-w-4xl">
            <Composer busy={send.isPending} photo={photo} progress={progress} error={send.error} unavailable={unavailable} onSend={(t, f) => ask(t, f)} />
            <p className="mt-1.5 text-center text-xs text-muted" title={ASSISTANT_DISCLAIMER}>
              Triage aid only · a qualified inspector gives final clearance · not affiliated with Michelin
            </p>
          </div>
        </div>

        {dragging && (
          <div aria-hidden className="pointer-events-none absolute inset-3 z-20 flex flex-col items-center justify-center gap-2 rounded-3xl border-2 border-dashed border-primary bg-primary-soft/90 text-center">
            <ImagePlus className="h-10 w-10 text-primary" />
            <span className="font-display text-xl font-bold text-primary-ink">{photoBlocked ?? `Drop the tyre photo for ${ASSISTANT_NAME}`}</span>
          </div>
        )}
      </section>
    </div>
  );
}
