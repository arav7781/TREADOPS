"use client";

import { useInfiniteQuery } from "@tanstack/react-query";

import { Badge, Button, Card, EmptyState, ErrorState, inputClass, PageHeader, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { dateTime } from "@/lib/format";
import { qk } from "@/lib/query";
import type { AuditItem } from "@/lib/types";
import { toQuery, useUrlState } from "@/lib/useUrlState";

export function AuditView() {
  const [f, setF] = useUrlState(["action", "resource_type", "actor", "outcome"] as const);
  const q = useInfiniteQuery({
    queryKey: qk.audit(f),
    queryFn: ({ pageParam }) => api<{ items: AuditItem[]; next_before_id: number | null }>(`/v1/audit-events${toQuery({ ...f, limit: 50, before_id: pageParam || undefined })}`),
    initialPageParam: 0,
    getNextPageParam: (last) => last.next_before_id ?? undefined,
  });
  const items = q.data?.pages.flatMap((p) => p.items) ?? [];
  return (
    <>
      <PageHeader eyebrow="Governance" title="Audit trail" subtitle="Append-only record of logins, reviews, releases, rollbacks, snapshots, training, replays and drift reviews." />
      <form className="card mb-6 grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-4" onSubmit={(e) => e.preventDefault()} aria-label="Audit filters">
        {([["action", "Action prefix", "e.g. release."], ["resource_type", "Resource type", "e.g. deployment"], ["actor", "Actor email", "admin@treadops.local"]] as const).map(([k, label, ph]) => (
          <div key={k} className="flex flex-col gap-1">
            <label htmlFor={`a-${k}`} className="text-sm font-semibold text-ink-soft">{label}</label>
            <input id={`a-${k}`} className={inputClass} defaultValue={f[k]} placeholder={ph}
              onBlur={(e) => setF({ [k]: e.target.value.trim() })}
              onKeyDown={(e) => { if (e.key === "Enter") setF({ [k]: (e.target as HTMLInputElement).value.trim() }); }} />
          </div>
        ))}
        <div className="flex flex-col gap-1">
          <label htmlFor="a-outcome" className="text-sm font-semibold text-ink-soft">Outcome</label>
          <select id="a-outcome" className={inputClass} value={f.outcome} onChange={(e) => setF({ outcome: e.target.value })}>
            <option value="">Any</option><option value="success">Success</option><option value="denied">Denied</option><option value="failure">Failure</option>
          </select>
        </div>
      </form>
      <Card title="Events">
        {q.isError ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : q.isLoading ? <SkeletonRows rows={8} />
          : items.length === 0 ? <EmptyState title="No audit events match" /> : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[860px] text-left text-sm">
                <thead><tr className="border-b border-line text-muted">
                  {["Time", "Actor", "Action", "Resource", "Outcome", "Correlation", "Payload"].map((h) => <th key={h} scope="col" className="py-2 pr-3 font-semibold">{h}</th>)}
                </tr></thead>
                <tbody>
                  {items.map((e) => (
                    <tr key={e.id} className="border-b border-line/60 align-top">
                      <td className="py-2 pr-3 whitespace-nowrap">{dateTime(e.at)}</td>
                      <td className="py-2 pr-3">{e.actor ?? <span className="text-muted">system</span>}</td>
                      <td className="mono py-2 pr-3">{e.action}</td>
                      <td className="py-2 pr-3">{e.resource_type}{e.resource_id ? <span className="mono block text-xs text-muted">{e.resource_id.slice(0, 12)}</span> : null}</td>
                      <td className="py-2 pr-3"><Badge tone={e.outcome === "success" ? "good" : e.outcome === "denied" ? "warn" : "bad"}>{e.outcome}</Badge></td>
                      <td className="mono py-2 pr-3 text-xs">{e.correlation_id?.slice(0, 12) ?? ""}</td>
                      <td className="py-2 pr-3"><code className="mono block max-w-md truncate text-xs text-ink-soft" title={JSON.stringify(e.payload)}>{JSON.stringify(e.payload)}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        {q.hasNextPage && <div className="mt-4 flex justify-center"><Button variant="secondary" busy={q.isFetchingNextPage} onClick={() => q.fetchNextPage()}>Older events</Button></div>}
      </Card>
    </>
  );
}
