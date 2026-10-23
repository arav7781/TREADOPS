"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowDown, Camera, ChevronDown, History, ImagePlus, MessageSquare, Plus, ScanLine, Search, ShieldCheck, Sparkles, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { BibAvatar, ChatThread, Composer, CopilotModeBadge, useCopilotSession, usePhotoPicker, useSendTurn } from "@/components/domain/ChatThread";
import { useMe } from "@/components/AppShell";
import { ArrowDot, Button, ErrorState, SkeletonRows, TreadArc } from "@/components/ui";
import { api } from "@/lib/api";
import { ASSISTANT_DISCLAIMER, ASSISTANT_NAME } from "@/lib/brand";
import { ago } from "@/lib/format";
import { useModels } from "@/lib/hooks";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import { useUrlState } from "@/lib/useUrlState";

interface SessionRow { id: string; title: string; updated_at: string; inspection_id: string | null }

const STARTERS = [
  "How reliable is the current model?",
  "How much CO2 does retreading 25 truck casings save?",
  "Is there any drift right now?",
  "What visual cues mean a casing should not be retreaded?",
];

const HOW_TO = [
  { icon: Camera, title: "Attach a tyre photo", text: "Use the photo button, drop it on this window or paste it. Fill the frame, avoid glare, one tyre per photo." },
  { icon: ScanLine, title: "Watch the pipeline", text: `${ASSISTANT_NAME} runs the deployed YOLO model as a tool. Every preprocessing stage appears here with its real timing.` },
  { icon: Sparkles, title: "Get the decision", text: "Triage, calibrated probability, the circular pathway and next steps. An inspector confirms the disposition." },
];

const GRAPH = [
  ["guard_input", "PII redaction, injection and off-topic checks"],
  ["plan", "the LLM picks tools (or the deterministic planner)"],
  ["tools", "classify_tyre_image runs the pinned YOLO bundle"],
  ["guard_output", "no certification claims; every figure grounded"],
  ["suggest", "next steps for the inspector"],
] as const;

/** Keep the newest content in view while the user is at the bottom; leave them alone once they scroll up. */
function useStickToBottom(resetKey: string) {
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [atBottom, setAtBottom] = useState(true);
  useEffect(() => {
    const s = scroller.current;
    const c = content.current;
    if (!s || !c) return;
    const toBottom = () => s.scrollTo({ top: s.scrollHeight });
    const onScroll = () => {
      const near = s.scrollHeight - s.scrollTop - s.clientHeight < 96;
      pinned.current = near;
      setAtBottom(near);
    };
    s.addEventListener("scroll", onScroll, { passive: true });
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => { if (pinned.current) toBottom(); });
    ro?.observe(c);
    return () => { s.removeEventListener("scroll", onScroll); ro?.disconnect(); };
  }, []);
  useEffect(() => { // a conversation opens at its latest message; the welcome screen opens at the top
    pinned.current = Boolean(resetKey);
    scroller.current?.scrollTo({ top: resetKey ? scroller.current.scrollHeight : 0 });
  }, [resetKey]);
  const jump = () => {
    pinned.current = true;
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  };
  return { scroller, content, atBottom, jump };
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
    <div className="mx-auto max-w-4xl">
      <div className="relative overflow-hidden rounded-3xl bg-black px-6 pb-16 pt-9 text-white sm:px-10">
        <TreadArc className="-right-10 -top-16 h-80 w-80 text-white/[0.09]" />
        <div className="relative flex items-start gap-4">
          <div className="pop-in hidden sm:block"><BibAvatar size="lg" /></div>
          <div className="min-w-0">
            <p className="text-sm font-bold uppercase tracking-[0.16em] text-brand-yellow">{ASSISTANT_NAME} · casing copilot</p>
            <h2 className="headline mt-2 text-3xl sm:text-5xl">Hey, I&apos;m {ASSISTANT_NAME}.<br /><span className="headline-light">Here to help you inspect</span></h2>
            <p className="mt-3 max-w-xl text-white/80">
              Send me a tyre photo: I run the YOLO inspection step by step, explain the result and suggest what to do
              with the casing. Or ask about the model, drift and circularity.
            </p>
          </div>
        </div>
      </div>
      <div className="relative mx-3 -mt-10 rounded-3xl bg-navy p-3 shadow-xl sm:mx-6 sm:p-5">
        <p className="mb-3 inline-flex items-center gap-2 border-b-2 border-white px-1 pb-2 font-semibold text-white">
          <ScanLine aria-hidden className="h-5 w-5" /> Start here
        </p>
        <div className="grid overflow-hidden rounded-2xl bg-white sm:grid-cols-2 sm:divide-x sm:divide-line">
          {canAttach ? (
            <label htmlFor="chat-photo" className="group flex cursor-pointer items-center gap-4 px-5 py-4 hover:bg-surface-muted">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border-2 border-primary text-primary"><Camera aria-hidden className="h-5 w-5" /></span>
              <span className="min-w-0 flex-1"><span className="block text-sm text-ink-soft">Inspect</span><span className="block text-lg font-bold text-ink">A tyre photo</span></span>
              <ArrowDot />
            </label>
          ) : (
            <div className="flex items-center gap-4 px-5 py-4 opacity-70">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border-2 border-line-strong text-muted"><Camera aria-hidden className="h-5 w-5" /></span>
              <span className="min-w-0 flex-1"><span className="block text-sm text-ink-soft">Inspect a tyre photo</span><span className="block text-sm font-semibold text-warn">{blockedReason ?? "Loading…"}</span></span>
            </div>
          )}
          <button type="button" onClick={() => document.getElementById("chat-text")?.focus()}
            className="group flex items-center gap-4 border-t border-line px-5 py-4 text-left hover:bg-surface-muted sm:border-t-0">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border-2 border-primary text-primary"><MessageSquare aria-hidden className="h-5 w-5" /></span>
            <span className="min-w-0 flex-1"><span className="block text-sm text-ink-soft">Ask about</span><span className="block text-lg font-bold text-ink">The model &amp; circularity</span></span>
            <ArrowDot />
          </button>
        </div>
      </div>
      <ol className="mt-6 grid overflow-hidden rounded-2xl bg-primary text-white sm:grid-cols-3 sm:divide-x sm:divide-white/30">
        {HOW_TO.map(({ title, text }, i) => (
          <li key={title} className="stage-in px-5 py-5" style={{ animationDelay: `${i * 90}ms` }}>
            <span className="headline block text-lg">{i + 1}. {title}</span>
            <span className="mt-1 block text-sm text-white/80">{text}</span>
          </li>
        ))}
      </ol>
      <div className="mt-6 pb-2">
        <p className="mb-2 text-sm font-bold uppercase tracking-[0.12em] text-muted">Popular questions</p>
        <div className="flex flex-wrap gap-2">
          {STARTERS.map((s) => (
            <button key={s} type="button" onClick={() => onAsk(s)}
              className="rounded-lg border border-ink/70 bg-surface px-3.5 py-2 text-sm font-semibold text-ink transition hover:bg-brand-yellow">{s}</button>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Inspect + copilot in one chat window: the photo goes into the chat and the agent's YOLO tool does the inspection. */
export function BibView() {
  const me = useMe();
  const models = useModels();
  const [{ session }, setUrl] = useUrlState(["session"] as const);
  const sessions = useQuery({ queryKey: qk.copilotSessions, queryFn: () => api<{ items: SessionRow[] }>("/v1/copilot/sessions"), refetchInterval: 30_000 });
  const thread = useCopilotSession(session || null);
  const send = useSendTurn((r) => setUrl({ session: r.session_id }));
  const [progress, setProgress] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const { scroller, content, atBottom, jump } = useStickToBottom(session);
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
        <header className="flex items-center gap-3 border-b border-line bg-surface px-4 py-2.5 lg:px-6">
          <button type="button" onClick={() => setDrawer(true)} aria-label="Show conversations" className="rounded-full p-2 text-ink-soft hover:bg-surface-muted md:hidden">
            <History className="h-5 w-5" />
          </button>
          <BibAvatar />
          <div className="min-w-0 flex-1">
            <h1 className="headline truncate text-xl text-ink">{ASSISTANT_NAME} · casing copilot</h1>
            <p className="truncate text-sm text-muted">{title ?? "LangGraph agent · YOLO inspection tool · guarded and traced"}</p>
          </div>
          <span className="hidden lg:inline-flex"><CopilotModeBadge /></span>
          <Button variant="secondary" className="hidden sm:inline-flex" onClick={startNew}><Plus aria-hidden className="h-4 w-4" /> New</Button>
        </header>

        <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <div ref={content} className="mx-auto w-full max-w-4xl px-4 py-6 lg:px-6">
            {!session ? <Welcome canAttach={!photoBlocked} blockedReason={me.data ? photoBlocked : null} onAsk={(s) => ask(s)} />
              : thread.isError ? <ErrorState error={thread.error} onRetry={() => thread.refetch()} />
                : !thread.data ? <SkeletonRows rows={4} />
                  : <ChatThread messages={thread.data.messages} canReview={isInspector} autoScroll={false} onSuggest={(s) => ask(s)} />}
          </div>
        </div>

        {!atBottom && (
          <button type="button" onClick={jump} className="pop-in absolute bottom-36 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full bg-black px-4 py-2 text-sm font-semibold text-white shadow-lg">
            <ArrowDown aria-hidden className="h-4 w-4" /> Latest
          </button>
        )}

        <div className="border-t border-line bg-surface/95 px-4 pb-3 pt-3 backdrop-blur lg:px-6">
          <div className="mx-auto w-full max-w-4xl">
            <Composer busy={send.isPending} photo={photo} progress={progress} error={send.error} onSend={(t, f) => ask(t, f)} />
            <p className="mt-2 hidden text-center text-xs text-muted sm:block">
              Triage aid only: a qualified inspector gives final clearance. {ASSISTANT_DISCLAIMER}
            </p>
            <p className="mt-1.5 text-center text-xs text-muted sm:hidden" title={ASSISTANT_DISCLAIMER}>
              Triage aid only · independent prototype, not affiliated with Michelin
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
