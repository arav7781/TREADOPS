"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { useState } from "react";

import { GateStatus } from "@/components/domain/badges";
import { JobProgress } from "@/components/domain/JobProgress";
import { useMe } from "@/components/AppShell";
import { Badge, Button, Card, EmptyState, ErrorState, inputClass, KeyValue, PageHeader, SkeletonRows, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { dateTime, humanize, num, pct, shortSha } from "@/lib/format";
import { qk } from "@/lib/query";
import { hasRole } from "@/lib/roles";
import type { DatasetsResponse, ManifestPage, SnapshotView } from "@/lib/types";
import { toQuery } from "@/lib/useUrlState";

const ROLE_ORDER = ["dev_train", "dev_val", "calibration", "policy", "test"];
const ROLE_PURPOSE: Record<string, string> = {
  dev_train: "fit weights", dev_val: "select checkpoint", calibration: "fit temperature only",
  policy: "select thresholds only", test: "locked final benchmark",
};

function Manifest({ snap }: { snap: SnapshotView }) {
  const [role, setRole] = useState("");
  const [offset, setOffset] = useState(0);
  const q = useQuery({
    queryKey: qk.manifest(snap.id, role, offset),
    queryFn: () => api<ManifestPage>(`/v1/datasets/${snap.id}/manifest${toQuery({ role, offset, limit: 20 })}`),
  });
  return (
    <div className="mt-4">
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor="m-role" className="text-sm font-semibold text-ink-soft">Role</label>
          <select id="m-role" className={inputClass} value={role} onChange={(e) => { setRole(e.target.value); setOffset(0); }}>
            <option value="">All</option><option value="dev_train">dev_train</option><option value="dev_val">dev_val</option>
          </select>
        </div>
        {q.data && <p className="text-sm text-muted">rows {offset + 1}–{Math.min(offset + 20, q.data.total)} of {q.data.total}</p>}
        <div className="ml-auto flex gap-2">
          <Button variant="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>Previous</Button>
          <Button variant="secondary" disabled={!q.data || offset + 20 >= q.data.total} onClick={() => setOffset(offset + 20)}>Next</Button>
        </div>
      </div>
      {q.isError ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : !q.data ? <SkeletonRows rows={5} /> : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead><tr className="border-b border-line text-muted">
              {["image_id", "group_id", "label", "label_origin", "role", "image_origin", "metadata_origin"].map((h) => <th key={h} scope="col" className="py-2 pr-3 font-semibold">{h}</th>)}
            </tr></thead>
            <tbody>
              {q.data.items.map((r) => (
                <tr key={r.image_id} className="border-b border-line/60">
                  <td className="mono py-1.5 pr-3">{r.image_id.slice(0, 8)}</td>
                  <td className="mono py-1.5 pr-3">{r.group_id}</td>
                  <td className="py-1.5 pr-3">{r.label}</td>
                  <td className="py-1.5 pr-3">{r.label_origin}</td>
                  <td className="py-1.5 pr-3">{r.role}</td>
                  <td className="py-1.5 pr-3">{r.image_origin}</td>
                  <td className="py-1.5 pr-3">{r.metadata_origin}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function DatasetsView() {
  const qc = useQueryClient();
  const me = useMe();
  const q = useQuery({ queryKey: qk.datasets, queryFn: () => api<DatasetsResponse>("/v1/datasets") });
  const [job, setJob] = useState<string | null>(null);
  const [desc, setDesc] = useState("");
  const [includeUploads, setIncludeUploads] = useState(true);
  const [open, setOpen] = useState<string | null>(null);
  const create = useMutation({
    mutationFn: () => api<{ job_id: string }>("/v1/datasets/snapshots", { method: "POST", json: { description: desc || null, include_verified_uploads: includeUploads } }),
    onSuccess: (r) => setJob(r.job_id),
  });
  if (q.isError) return <><PageHeader eyebrow="Data engineering" title="Datasets" /><ErrorState error={q.error} onRetry={() => q.refetch()} /></>;
  const d = q.data;
  if (!d) return <SkeletonRows rows={8} />;
  const src = d.source;
  const leakage = d.split?.leakage ?? {};
  const leakFree = Object.values(leakage).every((v) => v === 0);
  return (
    <>
      <PageHeader eyebrow="Data engineering" title="Datasets"
        subtitle="Audited public archive, reviewed same-tyre groups, frozen roles and immutable training-eligible snapshots." />
      {src ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="Archive images" value={num(src.actual_counts.archive_members_images)} hint={`publisher describes ${src.publisher_described_images}`} />
          <Stat label="Unique images" value={num(src.actual_counts.canonical_images)} hint={`${src.actual_counts.exact_duplicates} byte-identical duplicates removed`} tone="accent" />
          <Stat label="Same-tyre groups" value={num(src.actual_counts.groups)} hint={`${src.actual_counts.multi_member_groups} multi-photo groups`} />
          <Stat label="Quarantined" value={num(src.actual_counts.quarantined)} hint={`${src.actual_counts.images_in_label_conflicting_groups} in label-conflicting groups`} tone={src.actual_counts.quarantined ? "warn" : "good"} />
        </div>
      ) : <EmptyState title="Source manifest not found">Run the prepare stage to audit the archive.</EmptyState>}

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card title="Source and licence" subtitle="Attribution is preserved in every snapshot">
          {src && (
            <KeyValue items={[
              ["Dataset", src.name],
              ["DOI", <a key="doi" href={`https://doi.org/${src.doi}`} className="break-all text-primary-ink hover:underline" target="_blank" rel="noreferrer">{src.doi} <ExternalLink aria-hidden className="inline h-3.5 w-3.5" /></a>],
              ["Licence", <Badge key="l" tone="accent">{src.license}</Badge>],
              ["Attribution", src.attribution],
              ["Retrieved", dateTime(src.retrieved_at)],
              ["Archive SHA-256", <span key="s" className="mono">{shortSha(src.archive_sha256, 20)}</span>],
              ["Archive size", `${(src.archive_bytes / 1e9).toFixed(2)} GB`],
              ["By label", Object.entries(src.actual_counts.by_label).map(([k, v]) => `${k} ${v}`).join(" · ")],
              ["Preprocess contract", <span key="p" className="mono">{shortSha(src.preprocess_sha256 ?? "", 16)}</span>],
            ]} />
          )}
          <p className="mt-4 text-sm text-muted">Public photographs only. No record is Michelin production data; replay metadata is simulated.</p>
        </Card>
        <Card title="Audit evidence" subtitle="Local file audit — the publisher description is not a substitute">
          {d.quality ? (
            <KeyValue items={[
              ["Publisher hash checks", Object.entries(d.quality.publisher_hash_checks).map(([k, v]) => `${k} ${v}`).join(" · ")],
              ["Quarantine reasons", Object.keys(d.quality.quarantine_reasons).length ? Object.entries(d.quality.quarantine_reasons).map(([k, v]) => `${humanize(k)} ${v}`).join(" · ") : "none"],
              ["Top resolutions", Object.entries(d.quality.source_resolutions).slice(0, 3).map(([k, v]) => `${k} (${v})`).join(", ")],
              ["EXIF orientations", Object.entries(d.quality.exif_orientations).map(([k, v]) => `${k === "0" ? "none" : k}: ${v}`).join(" · ")],
            ]} />
          ) : <p className="text-muted">No quality report.</p>}
          {d.quality && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <caption className="mb-1 text-left text-muted">Descriptor distribution (canonical images)</caption>
                <thead><tr className="border-b border-line text-muted"><th scope="col" className="py-1">Descriptor</th><th scope="col">p05</th><th scope="col">p50</th><th scope="col">p95</th></tr></thead>
                <tbody>{Object.entries(d.quality.descriptor_summary).filter(([k]) => !["width", "height"].includes(k)).map(([k, v]) => (
                  <tr key={k} className="border-b border-line/60"><td className="py-1">{k}</td><td className="mono">{v.p05.toFixed(3)}</td><td className="mono">{v.p50.toFixed(3)}</td><td className="mono">{v.p95.toFixed(3)}</td></tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <Card className="mt-6" title="Frozen split" subtitle={d.split ? `seed ${d.split.config?.seed ?? 42} · split ${shortSha(d.split.split_sha256)} · grouped by same-tyre cluster` : undefined}
        actions={<GateStatus status={leakFree ? "passed" : "failed"} label="Leakage checks" />}>
        {d.split ? (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-left">
                <thead><tr className="border-b border-line text-sm text-muted">
                  <th scope="col" className="py-2">Role</th><th scope="col">Purpose</th><th scope="col">Images</th><th scope="col">Groups</th><th scope="col">Defective</th><th scope="col">Good</th>
                </tr></thead>
                <tbody>
                  {ROLE_ORDER.filter((r) => d.split?.counts[r]).map((r) => {
                    const c = d.split!.counts[r];
                    return (
                      <tr key={r} className="border-b border-line/60">
                        <td className="py-2 font-semibold">{r}</td><td className="text-ink-soft">{ROLE_PURPOSE[r]}</td>
                        <td className="mono">{c.images}</td><td className="mono">{c.groups}</td>
                        <td className="mono">{c.by_label.defective ?? 0}</td><td className="mono">{c.by_label.good ?? 0}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-sm text-muted">
              Actual outer fractions: {Object.entries(d.split.actual_outer_fractions).map(([k, v]) => `${k} ${pct(v)}`).join(" · ")}.
              Leakage: {Object.entries(leakage).map(([k, v]) => `${humanize(k)} = ${v}`).join(" · ")}.
            </p>
          </>
        ) : <EmptyState title="No split manifest" />}
      </Card>

      <Card className="mt-6" title="Eligible snapshots" subtitle="Immutable manifests: frozen dev roles plus inspector-verified uploads; protected roles and replays are excluded">
        {hasRole(me.data?.roles, "ml_engineer") && (
          <form className="mb-5 grid gap-3 rounded-2xl bg-surface-muted p-4 md:grid-cols-[1fr_auto_auto]" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
            <div className="flex flex-col gap-1">
              <label htmlFor="snap-desc" className="text-sm font-semibold text-ink-soft">Description</label>
              <input id="snap-desc" className={inputClass} value={desc} maxLength={500} onChange={(e) => setDesc(e.target.value)} placeholder="e.g. week 43 verified uploads" />
            </div>
            <label className="flex items-center gap-2 self-end pb-2 text-ink-soft">
              <input type="checkbox" checked={includeUploads} onChange={(e) => setIncludeUploads(e.target.checked)} className="h-4 w-4 accent-[var(--primary)]" />
              Include verified uploads
            </label>
            <Button type="submit" busy={create.isPending} className="self-end">Create snapshot</Button>
            {create.error ? <div className="md:col-span-3"><ErrorState error={create.error} title="Snapshot not started" /></div> : null}
            {job && <div className="md:col-span-3"><JobProgress jobId={job} label="Snapshot" onDone={() => qc.invalidateQueries({ queryKey: qk.datasets })} /></div>}
          </form>
        )}
        {d.snapshots.length === 0 ? <EmptyState title="No snapshots yet" /> : (
          <ul className="space-y-3">
            {d.snapshots.map((s) => (
              <li key={s.id} className="rounded-2xl border border-line p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-bold text-ink">{s.description || "Snapshot"}</span>
                  <Badge tone="primary">{s.counts.included} images</Badge>
                  <Badge>{s.counts.groups} groups</Badge>
                  <span className="mono text-sm text-muted">{shortSha(s.manifest_sha256)} · {dateTime(s.created_at)}</span>
                  <Button variant="ghost" className="ml-auto" aria-expanded={open === s.id} onClick={() => setOpen(open === s.id ? null : s.id)}>
                    {open === s.id ? "Hide manifest" : "Preview manifest"}
                  </Button>
                </div>
                <p className="mt-1 text-sm text-ink-soft">
                  Included: {Object.entries(s.counts.by_role_label).map(([k, v]) => `${k} ${v}`).join(" · ")}
                </p>
                <p className="text-sm text-muted">
                  Excluded: {Object.entries(s.counts.excluded).map(([k, v]) => `${humanize(k)} ${v}`).join(" · ") || "none"}
                </p>
                {open === s.id && <Manifest snap={s} />}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  );
}
