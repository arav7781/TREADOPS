"use client";

import { useInfiniteQuery } from "@tanstack/react-query";

import { InspectionTable } from "@/components/domain/InspectionTable";
import { Button, Card, EmptyState, ErrorState, inputClass, PageHeader, SkeletonRows } from "@/components/ui";
import { api } from "@/lib/api";
import { qk } from "@/lib/query";
import type { InspectionList } from "@/lib/types";
import { toQuery, useUrlState } from "@/lib/useUrlState";

const KEYS = ["status", "triage", "source", "camera_id", "line_id", "model", "metadata_origin", "reviewed"] as const;

export function Inspections() {
  const [f, setF] = useUrlState(KEYS);
  const q = useInfiniteQuery({
    queryKey: qk.inspections(f),
    queryFn: ({ pageParam }) => api<InspectionList>(`/v1/inspections${toQuery({ ...f, limit: 25, cursor: pageParam })}`),
    initialPageParam: "",
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    refetchInterval: 15_000,
  });
  const items = q.data?.pages.flatMap((p) => p.items) ?? [];
  const select = (id: string, label: string, opts: [string, string][], key: (typeof KEYS)[number]) => (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-semibold text-ink-soft">{label}</label>
      <select id={id} className={inputClass} value={f[key]} onChange={(e) => setF({ [key]: e.target.value })}>
        {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </div>
  );
  return (
    <>
      <PageHeader eyebrow="Records" title="Inspections" subtitle="Every inspection is pinned to the deployment generation that accepted it." />
      <form className="card mb-6 grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-4" aria-label="Inspection filters" onSubmit={(e) => e.preventDefault()}>
        {select("i-status", "Status", [["", "Any"], ["queued", "Queued"], ["running", "Running"], ["succeeded", "Succeeded"], ["failed", "Failed"]], "status")}
        {select("i-triage", "Triage", [["", "Any"], ["good_candidate", "Retread candidate"], ["manual_review", "Expert review"], ["defect_flagged", "Defect flagged"]], "triage")}
        {select("i-source", "Source", [["", "Any"], ["upload", "Upload"], ["camera", "Line camera"], ["replay", "Replay"], ["copilot", "Bib chat"]], "source")}
        {select("i-reviewed", "Human label", [["", "Any"], ["true", "Reviewed"], ["false", "Awaiting review"]], "reviewed")}
        {select("i-origin", "Metadata origin", [["", "Any"], ["user_provided", "User provided"], ["simulated", "Simulated"], ["observed", "Observed"]], "metadata_origin")}
        {(["camera_id", "line_id", "model"] as const).map((k) => (
          <div key={k} className="flex flex-col gap-1">
            <label htmlFor={`i-${k}`} className="text-sm font-semibold text-ink-soft">{k === "model" ? "Model version / bundle" : k === "camera_id" ? "Camera" : "Line"}</label>
            <input id={`i-${k}`} className={inputClass} defaultValue={f[k]}
              onBlur={(e) => setF({ [k]: e.target.value.trim() })}
              onKeyDown={(e) => { if (e.key === "Enter") setF({ [k]: (e.target as HTMLInputElement).value.trim() }); }} />
          </div>
        ))}
      </form>
      <Card title={`${items.length}${q.hasNextPage ? "+" : ""} inspections`} subtitle="Newest first · cursor pagination (max 100 per page server-side)">
        {q.isError ? <ErrorState error={q.error} onRetry={() => q.refetch()} />
          : q.isLoading ? <SkeletonRows rows={8} />
            : items.length === 0 ? <EmptyState title="No inspections match these filters" />
              : <InspectionTable items={items} />}
        {q.hasNextPage && (
          <div className="mt-4 flex justify-center">
            <Button variant="secondary" busy={q.isFetchingNextPage} onClick={() => q.fetchNextPage()}>Load more</Button>
          </div>
        )}
      </Card>
    </>
  );
}
