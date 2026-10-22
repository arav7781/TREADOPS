"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { StatusBadge } from "@/components/domain/badges";
import { Button, ErrorState, ProgressBar } from "@/components/ui";
import { api } from "@/lib/api";
import { humanize } from "@/lib/format";
import { useJob } from "@/lib/hooks";
import { qk } from "@/lib/query";
import type { JobView } from "@/lib/types";

/** Live job status with cancellation; calls onDone once when the job reaches a terminal state. */
export function JobProgress({ jobId, label, cancellable, onDone }: {
  jobId: string;
  label: string;
  cancellable?: boolean;
  onDone?: (job: JobView) => void;
}) {
  const q = useJob(jobId);
  const qc = useQueryClient();
  const notified = useRef(false);
  const cancel = useMutation({
    mutationFn: () => api<JobView>(`/v1/jobs/${jobId}/cancel`, { method: "POST", json: { reason: "cancelled from the UI" } }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.job(jobId) }),
  });
  const job = q.data;
  const terminal = job && ["succeeded", "failed", "cancelled"].includes(job.status);
  useEffect(() => {
    if (terminal && job && !notified.current) {
      notified.current = true;
      onDone?.(job);
    }
  }, [terminal, job, onDone]);
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  if (!job) return <p className="text-muted">Loading job…</p>;
  return (
    <div className="space-y-2 rounded-2xl border border-line bg-surface-muted p-4" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-bold text-ink">{label}</span>
        <StatusBadge status={job.status} />
        <span className="mono text-sm text-muted">job {job.job_id.slice(0, 8)} · attempt {job.attempt}/{job.max_attempts}</span>
        {cancellable && !terminal && (
          <Button variant="secondary" className="ml-auto" busy={cancel.isPending} onClick={() => cancel.mutate()}
            disabled={job.cancel_requested}>{job.cancel_requested ? "Cancelling…" : "Cancel"}</Button>
        )}
      </div>
      {!terminal && <ProgressBar value={job.progress} label={`${label} progress`} />}
      {job.progress_message && !terminal && <p className="text-sm text-ink-soft">{job.progress_message}</p>}
      {job.error && <p className="text-bad">{humanize(job.error.code)}: {job.error.message}</p>}
      {cancel.error ? <ErrorState error={cancel.error} title="Cancel failed" /> : null}
    </div>
  );
}
