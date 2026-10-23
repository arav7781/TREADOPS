/**
 * Response contracts for the /v1 API. Request/path types are generated into api-schema.ts from the
 * FastAPI OpenAPI document; most responses are plain dicts there, so they are mirrored here from the
 * backend routers (services/api/treadops_api/routers/*.py). Every field that can be absent is optional.
 */

export type Triage = "good_candidate" | "manual_review" | "defect_flagged";
export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "retry_wait"
  | "cancellation_requested"
  | "cancelled";
export type GateState = "passed" | "failed" | "insufficient_evidence" | "not_applicable";
export type MetadataOrigin = "simulated" | "user_provided" | "observed";
export type Role = "viewer" | "inspector" | "ml_engineer" | "admin";

export interface ApiErrorBody {
  error: { code: string; message: string; request_id?: string | null; retryable?: boolean; details?: Record<string, unknown> };
}

export interface Me {
  id: string;
  email: string;
  display_name: string;
  roles: Role[];
  allowed_actions: string[];
}

export interface Ratio {
  value: number | null;
  numerator: number;
  denominator: number;
  ci95: [number, number] | null;
  status: "ok" | "insufficient_evidence";
}

export interface BinaryMetrics {
  threshold: number;
  support: { defective: number; good: number; total: number };
  confusion: { tp: number; fn: number; tn: number; fp: number };
  defect_recall: Ratio;
  specificity: Ratio;
  precision: Ratio;
  accuracy: Ratio;
}

export interface ReliabilityBin {
  bin: number;
  lo: number;
  hi: number;
  count: number;
  mean_predicted: number;
  observed_defective_rate: number;
}

export interface ProbabilisticMetrics {
  n: number;
  status: string;
  average_precision?: number;
  brier?: number;
  nll?: number;
  ece?: number;
  reliability?: ReliabilityBin[];
}

export interface TriageCoverageBucket {
  count: number;
  fraction: number | null;
  defective: number;
  good: number;
}

export interface TriageCoverage {
  n: number;
  good_candidate: TriageCoverageBucket;
  manual_review: TriageCoverageBucket;
  defect_flagged: TriageCoverageBucket;
  false_negatives_routed_to_good_candidate: number;
  defective_support: number;
}

export interface CohortReport {
  binary: BinaryMetrics;
  probabilistic: ProbabilisticMetrics;
  triage: TriageCoverage;
  parents: number;
  evidence_label?: string;
  status?: string;
  by_role?: Record<string, BinaryMetrics>;
  finite_probabilities?: boolean;
}

export interface LatencySummary {
  n: number;
  mean_ms?: number;
  p50_ms?: number;
  p95_ms?: number;
  p99_ms?: number;
  max_ms?: number;
}

export interface Stage {
  index: number;
  name: string;
  title: string;
  duration_ms: number;
  params: Record<string, unknown>;
  log: string[];
  image_url: string | null;
}

export interface Circularity {
  pathway: "retread_intake" | "expert_inspection" | "material_recovery";
  label: string;
  action: string;
  estimate_if_verified: { raw_material_saved_kg: number; co2e_avoided_kg: number } | null;
  requires_human_clearance: boolean;
  note?: string;
  factors_version: string;
  scope: string;
}

export interface PredictionView {
  task: "classify";
  raw_probs: Record<string, number>;
  calibrated_probs: Record<string, number>;
  p_defective: number;
  triage: Triage;
  triage_reason: string;
  binary_defective: boolean;
  quality_status: "accepted" | "low_quality";
  quality: { descriptors?: Record<string, number>; reasons?: string[] };
  timing: Record<string, number>;
  lineage: Record<string, unknown>;
  attribution: { method?: string; grid?: number; max_delta?: number; min_delta?: number; strongest_cell?: number[]; note?: string } | null;
  circularity: Circularity | null;
  heatmap_url: string | null;
  classified_image_url: string | null;
}

export interface ModelTuple {
  deployment_id?: string;
  generation?: number;
  model_version_id?: string;
  registry_version?: string | null;
  bundle_sha256?: string;
  bundle_uri?: string;
  preprocess_version?: string;
  calibration_version?: string;
  policy_version?: string;
  model_family?: string;
}

export interface ReviewEntry {
  revision: number;
  label: "good" | "defective" | "unjudgeable";
  disposition: string | null;
  reason: string;
  reviewer: string | null;
  verified_at: string;
}

export interface TimelineEvent {
  at: string;
  from: string | null;
  to: string;
  attempt: number;
  reason: string | null;
}

export interface InspectionDetail {
  inspection_id: string;
  status: string;
  created_at: string;
  completed_at: string | null;
  source: string;
  line_id: string | null;
  camera_id: string | null;
  scenario_id: string | null;
  replay_sequence?: number | null;
  metadata_origin: MetadataOrigin;
  tyre_category: string | null;
  notes: string | null;
  image: {
    image_id: string;
    url: string;
    width: number | null;
    height: number | null;
    original_sha256: string;
    image_origin: string;
    parent_image_id: string | null;
    transform: Record<string, unknown> | null;
    label_origin: string;
    publisher_label: string | null;
    role: string | null;
  };
  deployment: ModelTuple;
  prediction: PredictionView | null;
  review: { revision: number; label: string | null; disposition: string | null };
  review_history: ReviewEntry[];
  stages: Stage[];
  job: {
    job_id: string;
    status: JobStatus;
    attempt: number;
    max_attempts: number;
    progress: number;
    error_code: string | null;
    error_message: string | null;
    next_retry_at: string | null;
  } | null;
  timeline?: TimelineEvent[];
}

export interface InspectionSummary {
  inspection_id: string;
  status: string;
  created_at: string;
  source: string;
  line_id: string | null;
  camera_id: string | null;
  scenario_id: string | null;
  metadata_origin: MetadataOrigin;
  thumbnail_url: string;
  triage: Triage | null;
  p_defective: number | null;
  binary_defective?: boolean | null;
  quality_status: string | null;
  human_label: string | null;
  disposition: string | null;
  model_version: string | null;
  bundle_sha256: string | null;
  deployment_generation: number;
}

export interface InspectionList {
  items: InspectionSummary[];
  next_cursor: string | null;
  limit: number;
  filters: Record<string, unknown>;
}

export interface InspectionAccepted {
  inspection_id: string;
  job_id: string;
  status: string;
  deployment_id: string;
  deployment_generation: number;
  metadata_origin: MetadataOrigin;
  poll_after_ms: number;
  status_url: string;
}

export interface ReviewQueue {
  items: InspectionSummary[];
  total: number;
  sort: string;
  oldest_waiting_at: string | null;
}

export interface ReviewCreated {
  inspection_id: string;
  revision: number;
  label: string;
  disposition: string | null;
}

export interface MetricsOverview {
  observed_at: string;
  filters: Record<string, string | null>;
  cohort_size: number;
  inspections: {
    by_status: Record<string, number>;
    by_triage: Record<string, number>;
    by_quality: Record<string, number>;
    by_metadata_origin: Record<string, number>;
  };
  latency_ms: {
    n: number;
    worker_total_p50: number | null;
    worker_total_p95: number | null;
    inference_p50: number | null;
    inference_p95: number | null;
    queue_p95: number | null;
  };
  timeline: { t: string; good_candidate: number; manual_review: number; defect_flagged: number; avg_total_ms: number | null }[];
  review: { awaiting: number; oldest_waiting_at: string | null };
  queue: { by_status: Record<string, number>; oldest_queued_age_s: number | null };
  workers: { name: string; queue: string; alive: boolean; paused: boolean; last_seen_at: string; ready_bundles: string[] }[];
  active_model: {
    state: "ready" | "untrained";
    generation: number;
    registry_version: string | null;
    bundle_sha256: string | null;
    policy_version: string | null;
    gate_status: string | null;
    snapshot: string | null;
    message: string | null;
  };
  drift: {
    latest_window: { status: string; created_at: string; n: number; distinct_parents: number } | null;
    open_alerts: number;
  };
}

export interface ModelView {
  id: string;
  registry_name: string;
  registry_version: string | null;
  bundle_sha256: string;
  model_family: string;
  stage: string;
  gate_status: GateState;
  is_active: boolean;
  created_at: string;
  mlflow_run_id: string | null;
  headline: {
    gate_recall: number | null;
    gate_specificity: number | null;
    test_recall: number | null;
    test_specificity: number | null;
    model_p95_ms: number | null;
  };
  training: Record<string, unknown> | null;
  runtimes: string[] | null;
  deployments: { generation: number; action: string; state: string; at: string }[];
}

export interface ModelsResponse {
  active: ModelView | null;
  challenger: ModelView | null;
  generation: number;
  items: ModelView[];
  state: "ready" | "untrained";
  message: string | null;
}

export interface GateItem {
  id: string;
  status: GateState;
  detail?: unknown;
}

export interface QualityTarget {
  id: string;
  target: number;
  value: number | null;
  ci95: [number, number] | null;
  numerator: number | null;
  support: number;
  status: GateState;
}

export interface GateReport {
  status: GateState;
  evidence_label?: string;
  threshold?: number;
  hard_gates: GateItem[];
  quality_targets: QualityTarget[];
  comparative: { status: GateState; reason?: string; [k: string]: unknown };
  triage?: TriageCoverage;
  rule?: string;
}

export interface PairedDiff {
  status: string;
  metric?: string;
  difference?: number;
  ci95?: [number, number];
  parents?: number;
  support?: number;
}

export interface StressReport {
  parents?: number;
  pairing?: string;
  clean?: BinaryMetrics;
  scenarios?: Record<
    string,
    {
      transform: Record<string, number>;
      binary: BinaryMetrics;
      triage: TriageCoverage;
      recall_vs_clean: PairedDiff;
      specificity_vs_clean: PairedDiff;
      quality_low_fraction: number;
    }
  >;
}

export interface ParityReport {
  status: string;
  n_images: number;
  mean_abs_probability_error: number;
  max_abs_probability_error: number;
  tolerances: { mean: number; max: number };
  triage_agreement: number;
  triage_mismatches: unknown[];
}

export interface ShadowScenario {
  n: number;
  triage_agreement: number;
  triage_changes: Record<string, number>;
  mean_abs_p_delta: number;
  champion: BinaryMetrics;
  challenger: BinaryMetrics;
  recall_difference: PairedDiff;
  specificity_difference: PairedDiff;
}

export interface Evidence {
  model: ModelView;
  model_card: string | null;
  gate_report: GateReport;
  gate_cohort: CohortReport | null;
  locked_test: CohortReport | null;
  stress: StressReport | null;
  parity: ParityReport | null;
  latency: { device?: string; preprocess?: LatencySummary; model_only?: LatencySummary; preprocess_plus_model?: LatencySummary; hardware?: Record<string, unknown> } | null;
  comparative: Record<string, unknown> | null;
  training: Record<string, unknown> | null;
  calibration: { temperature: number; n_fitted: number; nll_before: number | null; nll_after: number | null; fitted_on_role: string } | null;
  policy: { binary_threshold: number; triage_low: number; triage_high: number; version: string; selection?: Record<string, unknown> } | null;
  manifest: Record<string, unknown>;
  shadow: { job_id: string; status: JobStatus; rows: number; report: { scenarios?: Record<string, ShadowScenario>; cohort?: string } | null } | null;
}

export interface DeploymentItem {
  deployment_id: string;
  generation: number;
  action: string;
  state: string;
  model_version_id: string;
  registry_version: string | null;
  bundle_sha256: string;
  reason: string;
  approved_by: string | null;
  readiness: Record<string, unknown> | null;
  previous_deployment_id: string | null;
  created_at: string;
}

export interface DeploymentsResponse {
  generation: number;
  active_deployment_id: string | null;
  items: DeploymentItem[];
}

export interface ReleaseRequested {
  job_id: string;
  status: string;
  action: string;
  model_version_id: string;
  expected_generation: number;
}

export interface JobView {
  job_id: string;
  kind: string;
  queue: string;
  status: JobStatus;
  attempt: number;
  max_attempts: number;
  progress: number;
  progress_message: string | null;
  error: { code: string; message: string | null; retryable: boolean | null } | null;
  cancel_requested: boolean;
  result: Record<string, unknown> | null;
  resource: { type: string | null; id: string | null };
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  next_retry_at: string | null;
  correlation_id: string | null;
  events?: TimelineEvent[];
}

export interface SnapshotView {
  id: string;
  manifest_sha256: string;
  dvc_rev: string | null;
  source_versions: Record<string, string>;
  split_policy: Record<string, unknown>;
  counts: {
    included: number;
    by_role_label: Record<string, number>;
    by_label_origin: Record<string, number>;
    excluded: Record<string, number>;
    groups: number;
  };
  license_refs: string[];
  description: string | null;
  created_at: string;
  job_id: string | null;
}

export interface DatasetsResponse {
  source: {
    source_id: string;
    name: string;
    doi: string;
    url: string;
    version: number;
    license: string;
    attribution: string;
    retrieved_at: string | null;
    archive_sha256: string;
    archive_bytes: number;
    publisher_described_images: number;
    actual_counts: {
      archive_members_images: number;
      canonical_images: number;
      by_label: Record<string, number>;
      quarantined: number;
      exact_duplicates: number;
      groups: number;
      multi_member_groups: number;
      images_in_label_conflicting_groups: number;
    };
    preprocess_sha256?: string;
  } | null;
  quality: {
    quarantine_reasons: Record<string, number>;
    publisher_hash_checks: Record<string, number>;
    descriptor_summary: Record<string, { p05: number; p50: number; p95: number }>;
    source_resolutions: Record<string, number>;
    exif_orientations: Record<string, number>;
  } | null;
  split: {
    counts: Record<string, { images: number; groups: number; by_label: Record<string, number> }>;
    actual_outer_fractions: Record<string, number>;
    leakage: Record<string, number>;
    split_sha256: string;
    excluded_images: number;
    config?: { seed?: number };
  } | null;
  roles: { role: string | null; label: string | null; images: number; groups: number }[];
  assets_by_origin: Record<string, number>;
  snapshots: SnapshotView[];
}

export interface ManifestPage {
  snapshot_id: string;
  manifest_sha256: string;
  total: number;
  offset: number;
  limit: number;
  items: Record<string, string>[];
}

export interface Recipe {
  id: string;
  model: string;
  epochs: number;
  imgsz: number;
  batch: number;
  optimizer: string;
  lr0: number;
  patience: number;
  seed: number;
  fraction?: number;
  sha256: string;
}

export interface TrainingRunView {
  id: string;
  snapshot_id: string;
  config_id: string;
  config_sha256: string;
  recipe: Record<string, unknown>;
  status: string;
  mlflow_run_id: string | null;
  epoch_metrics: ({ epoch: number } & Record<string, number>)[];
  metrics: Record<string, unknown> | null;
  artifact_uri: string | null;
  checkpoint_sha256: string | null;
  model_version_id: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  job: { job_id: string; status: JobStatus; progress: number; progress_message: string | null; attempt: number } | null;
  model_version?: { id: string; gate_status: GateState; stage: string; registry_version: string | null; bundle_sha256: string };
}

export interface PsiEntry {
  psi: number;
  bins: number;
  reference_mean: number;
  current_mean: number;
}

export interface MonitoringWindow {
  id: string;
  status: "ok" | "alert_candidate" | "insufficient_data";
  created_at: string;
  n_inspections: number;
  n_distinct_parents: number;
  first_at: string | null;
  last_at: string | null;
  reference_version: string;
  cohort_sha256: string;
  descriptor_psi: Record<string, PsiEntry>;
  prediction_shift: { reference_triage_fraction?: Record<string, number | null>; current_triage_fraction?: Record<string, number>; current_mean_p_defective?: number; note?: string };
  embedding_drift: { status?: string; auc?: number | null; pc_psi_mean?: number; note?: string };
  labelled_quality: { labelled?: number; coverage?: number; status?: string; defect_recall?: Ratio; specificity?: Ratio; median_label_age_h?: number; disagreement_rate?: number };
  system_health: { queue_ms?: LatencySummary; worker_total_ms?: LatencySummary; failed_jobs?: number; retries?: number };
  breakdown: { by_camera?: Record<string, unknown>[]; by_scenario?: Record<string, unknown>[] };
  evidently: { status?: string; version?: string; drifted_columns?: { count: number; share: number } | null; per_column_drift_score?: Record<string, number>; error?: string; reason?: string } | null;
  has_report: boolean;
}

export interface WindowsResponse {
  items: MonitoringWindow[];
  state: string;
  message: string | null;
}

export interface DriftAlert {
  id: string;
  state: "open" | "acknowledged" | "investigating" | "resolved";
  severity: string;
  features: string[];
  reference_version: string;
  cohort_sha256: string;
  window_ids: string[];
  details: { feature?: string; value?: number; threshold?: number; affected?: Record<string, unknown>; sample_inspection_ids?: string[] };
  recommendation: string;
  created_at: string;
  updated_at: string;
  reviews: { from: string; to: string; reason: string; by: string; at: string }[];
}

export interface ReplayScenario {
  id: string;
  version: string;
  [k: string]: unknown;
}

export interface ReplayScenarios {
  items: ReplayScenario[];
  defaults: { count: number; rate: number; seed: number };
  disclosure: string;
}

export interface ReplayItem {
  job_id: string;
  status: JobStatus;
  progress: number;
  progress_message: string | null;
  params: { scenario: string; seed: number; rate: number; count: number };
  result: Record<string, unknown> | null;
  inspections: number;
  completed: number;
  created_at: string;
  error: string | null;
}

export interface AuditItem {
  id: number;
  at: string;
  actor: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  outcome: string;
  correlation_id: string | null;
  payload: Record<string, unknown>;
}

export interface Ledger {
  basis: string;
  verified: Record<string, number>;
  recommended_by_model: Record<string, number>;
  estimates: { raw_material_saved_kg: number; co2e_avoided_kg: number; new_tyre_equivalents_avoided_material: number };
  factors: Record<string, number>;
  factors_version: string;
  factors_sha256: string;
  scope: string;
  sources: { id: string; title: string; url: string; used_for: string[]; note?: string }[];
  label: string;
  model_vs_inspector: { model_triage: string; inspector_disposition: string; count: number }[];
  verified_retreads_by_metadata_origin: Record<string, number>;
  since: string;
}

export interface Passport {
  passport_id: string;
  inspection_id: string;
  issued_at: string;
  casing: { tyre_category: string | null; image_sha256: string; image_url: string; metadata_origin: MetadataOrigin };
  model_assessment: { triage: Triage | null; p_defective: number | null; quality_status: string | null; recommended_pathway: string | null; classified_image_url: string | null };
  inspector_verification: { label: string | null; disposition: string | null; revision: number; history: ReviewEntry[] };
  impact_estimate: { raw_material_saved_kg: number; co2e_avoided_kg: number } | null;
  lineage: Record<string, unknown> | null;
  status: "verified" | "awaiting_inspector";
  disclaimer: string;
}

export interface CopilotStatus {
  configured: boolean;
  model: string | null;
  mode: "llm" | "unavailable";
  prompt_version: string;
  prompt_sha256: string;
}

/** One step of the agent's live timeline, appended by the LangGraph nodes and tools while a turn runs. */
export interface ChatEvent {
  at: string;
  kind: "start" | "guard_input" | "plan" | "tool_call" | "tool_progress" | "inspection" | "stage" | "tool_result"
    | "compose" | "guard_output" | (string & {});
  message?: string;
  tool?: string;
  args?: string;
  ok?: boolean;
  summary?: string;
  status?: string;
  rules?: string[];
  mode?: string;
  model?: string | null;
  iteration?: number;
  error?: string;
  prompt_version?: string;
  inspection_id?: string;
  deployment_generation?: number;
  bundle_sha256?: string;
  stage?: number;
  name?: string;
  title?: string;
  duration_ms?: number;
  grounding?: { checked: number; ungrounded: number[]; pass: boolean };
  frame?: number;
  time_s?: number | null;
  verdict?: string;
  triage?: string;
  p_defective?: number;
  camera?: string;
}

export interface VideoFrameResult {
  frame: number;
  time_s: number | null;
  inspection_id?: string;
  status: string;
  verdict?: "damaged" | "not_damaged";
  triage?: Triage;
  p_defective?: number;
  quality_status?: string;
}

export interface VideoResult {
  verdict?: "damaged" | "not_damaged";
  verdict_label?: string;
  needs_expert_check?: boolean;
  triage?: Triage;
  triage_label?: string;
  p_defective?: number;
  decisive_frame?: number;
  decisive_time_s?: number | null;
  decisive_inspection_id?: string;
  frames_submitted?: number;
  frames_classified?: number;
  frames_low_quality?: number;
  error?: string;
  frames: VideoFrameResult[];
}

export interface ChatPhoto {
  url: string;
  width: number;
  height: number;
  bytes: number;
  sha256: string;
  metadata_removed: boolean;
  gps_removed: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments: {
    photo?: ChatPhoto;
    video?: { source: "clip" | "camera"; duration_s: number | null; frames_considered: number | null;
      frames: (Omit<ChatPhoto, "url"> & { url: string; time_s: number | null })[] };
    has_photo?: boolean;
    has_video?: boolean;
    inspection_ids?: string[];
    video_result?: VideoResult;
    status?: "pending" | "complete" | "error";
    events?: ChatEvent[];
    inspection_id?: string | null;
    mode?: string;
    tools?: string[];
    guardrails?: string[];
    grounding?: { checked: number; ungrounded: number[]; pass: boolean } | null;
    error?: string;
  };
  suggestions: string[];
  trace_id: string | null;
  created_at: string;
}

export interface ChatSessionDetail {
  id: string;
  title: string;
  messages: ChatMessage[];
}

export interface ChatTurnStarted {
  session_id: string;
  user_message_id: string;
  assistant_message_id: string;
  inspection_id: string | null;
  status: string;
}

export interface LLMOpsOverview {
  since: string;
  turns: number;
  latency_ms: { p50: number | null; p95: number | null };
  tokens: { prompt: number; completion: number; cached_prompt: number; cache_hit_rate: number | null; llm_calls: number };
  cost_usd: number | null;
  by_status: Record<string, number>;
  by_mode: Record<string, number>;
  by_prompt_version: { prompt_version: string; prompt_sha256: string; turns: number; p95_ms: number | null }[];
  tools: Record<string, number>;
  guardrails: { stage: string; rule: string; action: string; count: number }[];
  grounding: { passed: number; checked: number; pass_rate: number | null };
  feedback: { up: number; down: number };
  timeline: { t: string; turns: number; p95_ms: number | null; flagged: number }[];
}

export interface TraceRow {
  id: string;
  created_at: string;
  status: string;
  mode: string;
  model: string | null;
  prompt_version: string;
  latency_ms: number;
  tokens: number;
  cost_usd: number | null;
  tools: string[];
  guardrail_flags: string[];
  grounding: { checked: number; ungrounded: number[]; pass: boolean } | null;
}

export interface TraceSpan {
  id: string;
  parent: string | null;
  name: string;
  kind: string;
  status: string;
  start_ms: number;
  duration_ms: number;
  attributes: Record<string, unknown>;
}

export interface TraceDetail {
  id: string;
  status: string;
  mode: string;
  model: string | null;
  prompt_version: string;
  prompt_sha256: string;
  latency_ms: number;
  usage: { prompt: number; completion: number; cached: number };
  cost_usd: number | null;
  error: string | null;
  grounding: { checked: number; ungrounded: number[]; pass: boolean } | null;
  spans: TraceSpan[];
  guardrail_events: { stage: string; rule: string; action: string; detail: Record<string, unknown> }[];
  user_message: string | null;
  assistant_message: string | null;
}

export interface CameraCounts { inspected: number; damaged: number; not_damaged: number; expert_check: number }

export interface Camera {
  id: string;
  name: string;
  code: string;
  line_id: string | null;
  stream_url: string;              // always masked by the API
  simulated: boolean;
  enabled: boolean;
  capture_mode: "interval" | "on_change";
  interval_s: number;
  change_threshold: number;
  min_sharpness: number;
  state: "idle" | "connecting" | "streaming" | "error" | "disabled";
  last_error: string | null;
  last_frame_at: string | null;
  last_capture_at: string | null;
  live: boolean;
  fps: number | null;
  resolution: string | null;
  frames_read: number;
  captures: number;
  last_hour: CameraCounts;
}

export interface CameraProbe { ok: boolean; width: number; height: number; fps: number | null; snapshot_url: string; stream_url: string }

