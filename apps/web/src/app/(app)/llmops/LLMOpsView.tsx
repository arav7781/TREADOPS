"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Coins, Gauge, MessagesSquare, ShieldCheck, ThumbsUp, Zap } from "lucide-react";

import { LatencyLine, SimpleBars } from "@/components/domain/charts";
import { Markdown } from "@/components/domain/Markdown";
import { TraceWaterfall } from "@/components/domain/TraceWaterfall";
import { CopilotModeBadge } from "@/components/domain/ChatThread";
import { Badge, Card, EmptyState, ErrorState, inputClass, KeyValue, PageHeader, SkeletonRows, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { ago, humanize, ms, num, pct, shortSha } from "@/lib/format";
import { qk } from "@/lib/query";
import type { LLMOpsOverview, TraceDetail, TraceRow } from "@/lib/types";
import { toQuery, useUrlState } from "@/lib/useUrlState";

function TracePanel({ id }: { id: string }) {
  const q = useQuery({ queryKey: qk.trace(id), queryFn: () => api<TraceDetail>(`/v1/llmops/traces/${id}`) });
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const t = q.data;
  if (!t) return <SkeletonRows rows={5} />;
  return (
    <div className="space-y-5">
      <KeyValue items={[
        ["Status / mode", `${humanize(t.status)} · ${humanize(t.mode)}`],
        ["Model", t.model ?? "none (deterministic)"],
        ["Prompt", <span key="p" className="mono">{t.prompt_version} · {shortSha(t.prompt_sha256)}</span>],
        ["Latency", ms(t.latency_ms)],
        ["Tokens", `${t.usage.prompt} prompt (${t.usage.cached} cached) · ${t.usage.completion} completion`],
        ["Cost", t.cost_usd === null ? "n/a (model not priced)" : `$${t.cost_usd.toFixed(6)}`],
        ["Grounding", t.grounding ? `${t.grounding.pass ? "pass" : "fail"} · ${t.grounding.checked} figures checked${t.grounding.ungrounded.length ? ` · ungrounded ${t.grounding.ungrounded.join(", ")}` : ""}` : "n/a"],
        ...(t.error ? [["Error", t.error] as [string, string]] : []),
      ]} />
      <div>
        <h3 className="mb-2 font-bold text-ink">Span waterfall</h3>
        <TraceWaterfall spans={t.spans} total={t.latency_ms} />
      </div>
      <div>
        <h3 className="mb-2 font-bold text-ink">Guardrail events</h3>
        <ul className="space-y-1">
          {t.guardrail_events.map((e, i) => (
            <li key={i} className="flex flex-wrap items-center gap-2 text-sm">
              <Badge tone={e.action === "block" ? "bad" : e.action === "pass" ? "good" : "warn"}>{e.stage} · {e.action}</Badge>
              <span className="font-semibold">{humanize(e.rule)}</span>
              {Object.keys(e.detail).length > 0 && <code className="mono truncate text-xs text-muted">{JSON.stringify(e.detail)}</code>}
            </li>
          ))}
        </ul>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div><h3 className="mb-2 font-bold text-ink">User (stored redacted)</h3><p className="rounded-xl bg-surface-muted p-3 text-ink-soft">{t.user_message ?? "n/a"}</p></div>
        <div><h3 className="mb-2 font-bold text-ink">Assistant (after output guardrails)</h3><div className="rounded-xl bg-surface-muted p-3">{t.assistant_message ? <Markdown text={t.assistant_message} /> : "n/a"}</div></div>
      </div>
    </div>
  );
}

export function LLMOpsView() {
  const [f, setF] = useUrlState(["trace", "status", "rule"] as const);
  const o = useQuery({ queryKey: qk.llmopsOverview, queryFn: () => api<LLMOpsOverview>("/v1/llmops/overview"), refetchInterval: 20_000 });
  const traces = useQuery({ queryKey: qk.traces({ status: f.status, rule: f.rule }), queryFn: () => api<{ items: TraceRow[] }>(`/v1/llmops/traces${toQuery({ status: f.status, rule: f.rule, limit: 50 })}`), refetchInterval: 20_000 });
  const evals = useQuery({ queryKey: qk.evals, queryFn: () => api<{ items: Record<string, unknown>[] }>("/v1/llmops/evals") });
  const d = o.data;
  const fallback = d && d.turns ? (d.by_mode.deterministic_fallback ?? 0) / d.turns : null;
  return (
    <>
      <PageHeader eyebrow="LLMOps" title="Copilot observability"
        subtitle="Every copilot turn is a trace: graph nodes, LLM calls and tool calls as spans, token usage, cost, guardrail events, grounding checks and user feedback — versioned by prompt."
        actions={<CopilotModeBadge />} />
      {o.isError ? <ErrorState error={o.error} onRetry={() => o.refetch()} /> : !d ? <SkeletonRows rows={4} /> : d.turns === 0 ? (
        <EmptyState title="No copilot turns yet">Ask the copilot something to generate the first trace.</EmptyState>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Turns (7 days)" icon={<MessagesSquare className="h-5 w-5" />} value={num(d.turns)} hint={Object.entries(d.by_status).map(([k, v]) => `${k} ${v}`).join(" · ")} />
            <Stat label="Latency p50 / p95" icon={<Gauge className="h-5 w-5" />} value={`${ms(d.latency_ms.p50)} / ${ms(d.latency_ms.p95)}`} hint="whole turn incl. YOLO tool" />
            <Stat label="Tokens" icon={<Zap className="h-5 w-5" />} value={num(d.tokens.prompt + d.tokens.completion)}
              hint={`${num(d.tokens.llm_calls)} LLM calls · cache hit ${d.tokens.cache_hit_rate === null ? "n/a" : pct(d.tokens.cache_hit_rate)}`} />
            <Stat label="Cost" icon={<Coins className="h-5 w-5" />} value={d.cost_usd === null ? "n/a" : `$${d.cost_usd.toFixed(4)}`} hint="from configured per-model pricing" />
            <Stat label="Fallback rate" icon={<Zap className="h-5 w-5" />} tone="warn" value={pct(fallback)} hint="deterministic tool-grounded answers" />
            <Stat label="Grounding pass" icon={<ShieldCheck className="h-5 w-5" />} tone="good" value={pct(d.grounding.pass_rate)} hint={`${d.grounding.passed}/${d.grounding.checked} turns`} />
            <Stat label="Feedback" icon={<ThumbsUp className="h-5 w-5" />} value={`${d.feedback.up} 👍 · ${d.feedback.down} 👎`} hint="per assistant message" />
            <Stat label="Guardrail triggers" icon={<ShieldCheck className="h-5 w-5" />} tone="bad" value={num(d.guardrails.reduce((a, g) => a + g.count, 0))} hint={`${d.guardrails.length} distinct rules`} />
          </div>
          <div className="mt-6 grid gap-6 xl:grid-cols-2">
            <Card title="Turns and p95 latency by hour"><LatencyLine data={d.timeline} /></Card>
            <Card title="Tool usage"><SimpleBars label="Tool calls" color="#0f9f8f" data={Object.entries(d.tools).map(([k, v]) => ({ name: k.replace(/^get_/, ""), value: v }))} /></Card>
          </div>
          <div className="mt-6 grid gap-6 xl:grid-cols-2">
            <Card title="Guardrail triggers">
              {d.guardrails.length === 0 ? <p className="text-muted">No guardrail interventions.</p> : (
                <table className="w-full text-left text-sm">
                  <thead><tr className="border-b border-line text-muted"><th scope="col" className="py-1.5">Stage</th><th scope="col">Rule</th><th scope="col">Action</th><th scope="col">Count</th></tr></thead>
                  <tbody>{d.guardrails.map((g, i) => (
                    <tr key={i} className="border-b border-line/60"><td className="py-1.5">{g.stage}</td><td>{humanize(g.rule)}</td>
                      <td><Badge tone={g.action === "block" ? "bad" : "warn"}>{g.action}</Badge></td><td className="mono">{g.count}</td></tr>
                  ))}</tbody>
                </table>
              )}
            </Card>
            <Card title="By prompt version">
              <table className="w-full text-left text-sm">
                <thead><tr className="border-b border-line text-muted"><th scope="col" className="py-1.5">Version</th><th scope="col">SHA</th><th scope="col">Turns</th><th scope="col">p95</th></tr></thead>
                <tbody>{d.by_prompt_version.map((p) => (
                  <tr key={p.prompt_sha256} className="border-b border-line/60"><td className="py-1.5 font-semibold">{p.prompt_version}</td><td className="mono">{p.prompt_sha256}</td><td className="mono">{p.turns}</td><td className="mono">{ms(p.p95_ms)}</td></tr>
                ))}</tbody>
              </table>
            </Card>
          </div>
        </>
      )}
      <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <Card title="Traces" actions={
          <>
            <label htmlFor="t-status" className="sr-only">Status</label>
            <select id="t-status" className={inputClass} value={f.status} onChange={(e) => setF({ status: e.target.value })}>
              <option value="">All statuses</option><option value="ok">ok</option><option value="fallback">fallback</option><option value="blocked">blocked</option>
            </select>
            <label htmlFor="t-rule" className="sr-only">Guardrail rule</label>
            <select id="t-rule" className={inputClass} value={f.rule} onChange={(e) => setF({ rule: e.target.value })}>
              <option value="">Any guardrail</option>
              {["prompt_injection", "off_topic", "pii_redaction", "certification_request", "certification_claim", "ungrounded_number", "triage_contradiction", "system_prompt_leak"].map((r) => <option key={r} value={r}>{humanize(r)}</option>)}
            </select>
          </>}>
          {traces.isError ? <ErrorState error={traces.error} onRetry={() => traces.refetch()} /> : !traces.data ? <SkeletonRows rows={5} />
            : traces.data.items.length === 0 ? <EmptyState title="No traces match" /> : (
              <ul className="max-h-[640px] space-y-1.5 overflow-auto pr-1">
                {traces.data.items.map((t) => (
                  <li key={t.id}>
                    <button type="button" onClick={() => setF({ trace: t.id })} aria-pressed={f.trace === t.id}
                      className={clsx("w-full rounded-xl border px-3 py-2 text-left", f.trace === t.id ? "border-primary bg-primary-soft" : "border-transparent hover:bg-surface-muted")}>
                      <span className="flex flex-wrap items-center gap-2">
                        <Badge tone={t.status === "ok" ? "good" : t.status === "blocked" ? "bad" : "warn"}>{t.status}</Badge>
                        <span className="text-sm text-ink-soft">{humanize(t.mode)}</span>
                        <span className="mono ml-auto text-sm">{ms(t.latency_ms)}</span>
                      </span>
                      <span className="mt-0.5 block text-sm text-muted">{ago(t.created_at)} · {t.tools.join(", ") || "no tools"}{t.guardrail_flags.length ? ` · ⚑ ${t.guardrail_flags.join(", ")}` : ""}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
        </Card>
        <Card title="Trace detail">{f.trace ? <TracePanel key={f.trace} id={f.trace} /> : <EmptyState title="Select a trace" />}</Card>
      </div>
      <Card className="mt-6" title="Offline evaluation reports" subtitle="Golden conversations run through the graph (guardrails, tool routing, grounding)">
        {evals.isError ? <ErrorState error={evals.error} onRetry={() => evals.refetch()} /> : !evals.data ? <SkeletonRows rows={2} />
          : evals.data.items.length === 0 ? <EmptyState title="No eval reports yet">Run the copilot eval suite to produce docs/evidence/llmops/copilot_eval_*.json.</EmptyState> : (
            <ul className="space-y-2">
              {evals.data.items.map((r, i) => (
                <li key={i} className="rounded-xl bg-surface-muted p-3">
                  <p className="font-semibold text-ink">{String(r.file ?? `report ${i + 1}`)}</p>
                  <code className="mono block truncate text-xs text-ink-soft" title={JSON.stringify(r)}>{JSON.stringify(Object.fromEntries(Object.entries(r).filter(([k]) => !["cases", "results", "file"].includes(k))))}</code>
                </li>
              ))}
            </ul>
          )}
      </Card>
    </>
  );
}
