"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Maximize2, Minimize2 } from "lucide-react";
import Link from "next/link";
import { use, useState } from "react";

import { JobTimeline } from "@/components/domain/JobTimeline";
import { ResultPanel } from "@/components/domain/ResultPanel";
import { ReviewForm } from "@/components/domain/ReviewForm";
import { PipelineConsole, StageGallery } from "@/components/domain/StageGallery";
import { SourceBadge, StatusBadge, TriageBadge, VerdictBadge } from "@/components/domain/badges";
import { useMe } from "@/components/AppShell";
import { Badge, Card, ErrorState, KeyValue, PageHeader, SkeletonRows } from "@/components/ui";
import { mediaUrl } from "@/lib/api";
import { dateTime, humanize, shortSha } from "@/lib/format";
import { useInspection } from "@/lib/hooks";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";

const SUGGESTED_DISPOSITION: Record<string, string> = {
  good_candidate: "retread", manual_review: "reinspect", defect_flagged: "material_recovery",
};

export function InspectionView({ idPromise }: { idPromise: Promise<string> }) {
  const id = use(idPromise);
  const q = useInspection(id);
  const me = useMe();
  const qc = useQueryClient();
  const [zoom, setZoom] = useState(false);
  const [saved, setSaved] = useState<number | null>(null); // survives the form remounting on the new revision
  const [view, setView] = useState<"original" | "heatmap" | "classified">("original");
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const d = q.data;
  if (!d) return <SkeletonRows rows={8} />;
  const running = d.status === "queued" || d.status === "running";
  const src = mediaUrl(view === "heatmap" ? d.prediction?.heatmap_url : view === "classified" ? d.prediction?.classified_image_url : d.image.url);
  const canReview = hasRole(me.data?.roles, "inspector");

  return (
    <>
      <PageHeader eyebrow={`Inspection ${d.inspection_id.slice(0, 8)}`}
        title={d.prediction ? (d.prediction.binary_defective ? "Damaged" : "Not damaged") : humanize(d.status)}
        subtitle={<>Created {dateTime(d.created_at)} · source {d.source}{d.scenario_id ? ` · scenario ${d.scenario_id}` : ""}{d.line_id ? ` · ${d.line_id}/${d.camera_id}` : ""}</>}
        actions={<><StatusBadge status={d.status} /><VerdictBadge damaged={d.prediction?.binary_defective} triage={d.prediction?.triage} quality={d.prediction?.quality_status} /><TriageBadge triage={d.prediction?.triage} /><SourceBadge origin={d.metadata_origin} />
          <Link href={`/circularity/passport/${d.inspection_id}`} className="font-semibold text-primary-ink hover:underline">Casing passport →</Link></>} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <Card title="Photo" actions={
          <>
            <div role="group" aria-label="Image view" className="flex rounded-xl border border-line-strong p-0.5">
              {(["original", "heatmap", "classified"] as const).map((v) => (
                <button key={v} type="button" aria-pressed={view === v} disabled={v !== "original" && !d.prediction}
                  onClick={() => setView(v)}
                  className={`rounded-lg px-2.5 py-1 text-sm font-semibold ${view === v ? "bg-primary text-white" : "text-ink-soft disabled:text-muted"}`}>
                  {v === "original" ? "Original" : v === "heatmap" ? "Attribution" : "Classified"}
                </button>
              ))}
            </div>
            <button type="button" onClick={() => setZoom((z) => !z)} aria-pressed={zoom}
              className="flex items-center gap-1 rounded-xl border border-line-strong px-2.5 py-1.5 text-sm font-semibold text-ink-soft hover:bg-surface-muted">
              {zoom ? <Minimize2 aria-hidden className="h-4 w-4" /> : <Maximize2 aria-hidden className="h-4 w-4" />} {zoom ? "Fit" : "Zoom 1:1"}
            </button>
          </>}>
          <div className={`rounded-xl bg-surface-muted ${zoom ? "max-h-[70vh] overflow-auto" : "flex items-center justify-center"}`}>
            {src ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={src} alt={`Tyre photo (${view}) for inspection ${d.inspection_id.slice(0, 8)}`}
                className={zoom ? "max-w-none" : "max-h-[60vh] w-auto object-contain"} />
            ) : <p className="p-10 text-muted">Image not available.</p>}
          </div>
          {view === "heatmap" && <p className="mt-2 text-sm text-muted">Occlusion attribution: red regions raised p(defective), blue lowered it. Not a defect location.</p>}
          <KeyValue className="mt-4" items={[
            ["Image", <span key="i" className="mono">{shortSha(d.image.original_sha256, 16)}</span>],
            ["Size", d.image.width ? `${d.image.width}×${d.image.height}` : "n/a"],
            ["Origin", humanize(d.image.image_origin)],
            ["Tyre category", humanize(d.tyre_category) || "n/a"],
            ...(d.image.transform && Object.keys(d.image.transform).length ? [["Derivative transform", <span key="t" className="mono">{JSON.stringify(d.image.transform)}</span>] as [string, React.ReactNode]] : []),
            ...(d.image.publisher_label ? [["Publisher label", <Badge key="p">{d.image.publisher_label}</Badge>] as [string, React.ReactNode]] : []),
            ...(d.notes ? [["Notes", d.notes] as [string, React.ReactNode]] : []),
          ]} />
        </Card>
        <div className="flex min-w-0 flex-col gap-6">
          <Card title="Human review" subtitle={`Current revision r${d.review.revision}${d.review.label ? ` · ${humanize(d.review.label)}` : " · not yet reviewed"}`}>
            {canReview ? (
              d.status === "succeeded" ? (
                <>
                {saved !== null && saved === d.review.revision && (
                  <p role="status" className="mb-3 rounded-lg bg-good-soft px-3 py-2 font-semibold text-good">Saved revision r{saved}</p>
                )}
                <ReviewForm key={d.review.revision} inspectionId={d.inspection_id} expectedRevision={d.review.revision}
                  current={{ label: d.review.label, disposition: d.review.disposition }}
                  suggestedDisposition={d.prediction ? SUGGESTED_DISPOSITION[d.prediction.triage] : null}
                  onSaved={(r) => { setSaved(r.revision); qc.invalidateQueries({ queryKey: qk.inspection(d.inspection_id) }); }} />
                </>
              ) : <p className="text-muted">Reviews open once the inspection completes.</p>
            ) : <p className="text-muted">Recording a verified label requires the inspector role.</p>}
            <h3 className="mb-2 mt-6 font-bold text-ink">Review history (append-only)</h3>
            {d.review_history.length === 0 ? <p className="text-muted">No reviews yet.</p> : (
              <ol className="space-y-2">
                {[...d.review_history].reverse().map((r) => (
                  <li key={r.revision} className="rounded-xl bg-surface-muted px-3 py-2">
                    <p className="font-semibold text-ink">r{r.revision} · {humanize(r.label)}{r.disposition ? ` · ${humanize(r.disposition)}` : ""}</p>
                    <p className="text-ink-soft">“{r.reason}”</p>
                    <p className="text-sm text-muted">{r.reviewer ?? "unknown"} · {dateTime(r.verified_at)}</p>
                  </li>
                ))}
              </ol>
            )}
          </Card>
          <Card title="Processing timeline">
            <JobTimeline events={d.timeline} attempt={d.job?.attempt} maxAttempts={d.job?.max_attempts} nextRetryAt={d.job?.next_retry_at} />
            {d.job?.error_code && <p className="mt-3 text-bad">{d.job.error_code}: {d.job.error_message}</p>}
          </Card>
        </div>
      </div>

      {d.prediction && <div className="mt-6"><ResultPanel d={d} /></div>}

      <Card className="mt-6" title="Pipeline stages" subtitle="Each image is what the serving path actually produced, with the recorded duration">
        <StageGallery stages={d.stages} running={running} animate={false} />
        <div className="mt-4"><PipelineConsole stages={d.stages} timeline={d.timeline} status={d.status} error={d.job?.error_message} /></div>
      </Card>
    </>
  );
}
