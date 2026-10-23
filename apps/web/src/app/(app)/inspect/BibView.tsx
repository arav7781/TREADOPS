"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowDown, Camera, ChevronDown, History, ImagePlus, MessageSquare, Plus, ScanLine, Search, ShieldCheck, Sparkles, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { BibAvatar, ChatThread, Composer, CopilotModeBadge, useCopilotSession, usePhotoPicker, useSendTurn } from "@/components/domain/ChatThread";
import { useMe } from "@/components/AppShell";
import { Button, ErrorState, SkeletonRows } from "@/components/ui";
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
            className="h-10 w-full rounded-full border border-line bg-surface-muted pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-primary" />
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
                            on ? "bg-primary-soft text-primary-ink shadow-[inset_3px_0_0_var(--primary)]" : "text-ink-soft hover:bg-surface-muted")}>
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
    <div className="mx-auto flex max-w-3xl flex-col items-center gap-6 py-6 text-center">
      <div className="pop-in"><BibAvatar size="lg" /></div>
      <div>
        <h2 className="text-3xl font-extrabold text-ink">Hi, I&apos;m {ASSISTANT_NAME}, your casing copilot.</h2>
        <p className="mx-auto mt-1 max-w-xl text-ink-soft">
          Send me a tyre photo and I will inspect it step by step, explain the result and suggest what to do with the
          casing. You can also ask about the model, drift or circularity.
        </p>
      </div>
      <ol className="grid w-full gap-3 text-left sm:grid-cols-3">
        {HOW_TO.map(({ icon: Icon, title, text }, i) => (
          <li key={title} className="stage-in rounded-xl border border-line bg-surface p-4" style={{ animationDelay: `${i * 90}ms` }}>
            <span className="mb-2 flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-navy text-sm font-bold text-brand-yellow">{i + 1}</span>
              <Icon aria-hidden className="h-5 w-5 text-primary" />
            </span>
            <span className="block font-display text-lg font-bold text-ink">{title}</span>
            <span className="block text-sm text-ink-soft">{text}</span>
          </li>
        ))}
      </ol>
      {canAttach ? (
        <label htmlFor="chat-photo" className="flex w-full max-w-lg cursor-pointer flex-col items-center gap-1.5 rounded-2xl border-2 border-dashed border-primary/40 bg-primary-soft/50 px-6 py-6 transition hover:border-primary hover:bg-primary-soft">
          <ImagePlus aria-hidden className="h-8 w-8 text-primary" />
          <span className="font-display text-lg font-bold text-ink">Attach or drop a tyre photo</span>
          <span className="text-sm text-muted">JPEG, PNG or WebP · full frame analysed, no centre crop · metadata stripped</span>
        </label>
      ) : blockedReason ? <p className="rounded-full bg-warn-soft px-4 py-1.5 text-sm font-semibold text-warn">{blockedReason}</p> : null}
      <div className="w-full">
        <p className="mb-2 text-sm font-semibold text-muted">Or ask</p>
        <div className="flex flex-wrap justify-center gap-2">
          {STARTERS.map((s) => (
            <button key={s} type="button" onClick={() => onAsk(s)}
              className="rounded-full border border-line-strong bg-surface px-3.5 py-1.5 text-sm font-semibold text-primary-ink transition hover:border-primary hover:bg-primary-soft">{s}</button>
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
  const sidebar = (
    <HistorySidebar rows={sessions.data?.items} error={sessions.isError ? sessions.error : null} onRetry={() => sessions.refetch()}
      active={session} onPick={pick} onNew={startNew} />
  );

  return (
    <div data-fullbleed className="flex min-h-0 w-full flex-1">
      <aside aria-label="Conversations" className="hidden w-72 shrink-0 border-r border-line bg-surface md:block">{sidebar}</aside>
      {drawer && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Conversations">
          <button type="button" aria-label="Close conversations" className="absolute inset-0 bg-navy/50" onClick={() => setDrawer(false)} />
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
            <h1 className="truncate text-xl font-extrabold leading-tight text-ink">{ASSISTANT_NAME} · casing copilot</h1>
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
          <button type="button" onClick={jump} className="pop-in absolute bottom-36 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full bg-navy px-4 py-2 text-sm font-semibold text-white shadow-lg">
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
