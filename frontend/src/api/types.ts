// Контракт с backend. Отражает pydantic-схемы hpc_algo/AnalysisResult
// и backend/app. Здесь только типы — никакой бизнес-логики.

export interface UserPublic {
  id: number;
  email: string;
  created_at: string;
  email_verified_at: string | null;
  terms_accepted_at: string | null;
  pdn_accepted_at: string | null;
  onboarding_completed_at: string | null;
  csrf_token?: string | null;
}

export type PreparationState = "draft" | "ready";

export interface SourceSummary {
  id: number;
  original_filename: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
  has_analysis: boolean;
  last_analysis_status: string | null;
  has_mapping: boolean;
  preparation_state: PreparationState;
}

export type AnalysisStatus =
  | "audit_only"
  | "baseline_only"
  | "needs_column_mapping"
  | "hmm_ready"
  | "failed";

export type WarningSeverity = "info" | "warning" | "error";

export interface WarningItem {
  code: string;
  message: string;
  severity: WarningSeverity;
  context: Record<string, unknown>;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  non_null_count: number;
  null_count: number;
  null_ratio: number;
  unique_count: number;
  sample_values: unknown[];
}

export interface SheetAudit {
  name: string;
  n_rows: number;
  n_cols: number;
  columns: ColumnInfo[];
  preview: Record<string, unknown>[];
  suspicious: string[];
}

export interface AuditReport {
  sheets: SheetAudit[];
  total_rows: number;
  total_cells: number;
  overall_null_ratio: number;
}

export type HiddenGroup =
  | "маневрирование"
  | "КФВ"
  | "ВУП"
  | "ЗАП"
  | "time"
  | "athlete"
  | "episode"
  | "bout"
  | "weight";

export interface HiddenGroupCandidate {
  group: HiddenGroup;
  sheet: string;
  column: string;
  score: number;
  rationale: string;
  sample_values: unknown[];
}

export interface ColumnDetectionReport {
  candidates: HiddenGroupCandidate[];
  detected_groups: HiddenGroup[];
  missing_groups: HiddenGroup[];
  assumptions: string[];
}

export interface ChartData {
  id: string;
  title: string;
  kind: "bar" | "hbar" | "heatmap" | "missing" | "hist" | "line";
  x: unknown[];
  y: unknown[];
  series: Record<string, unknown>[];
  meta: Record<string, unknown>;
}

export interface TimeStats {
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  median: number | null;
  null_count: number;
}

export type ZapColumnKind = "categorical" | "binary" | "count" | "empty";

export interface BaselineReport {
  zap_value_counts: Record<string, Record<string, number>>;
  missing_values_per_column: Record<string, Record<string, number>>;
  hidden_group_value_counts: Record<string, Record<string, Record<string, number>>>;
  hidden_group_totals: Record<string, number>;

  zap_column_kinds: Record<string, ZapColumnKind>;
  zap_events: Record<string, number>;
  zap_total_triggers: Record<string, number>;
  zap_events_by_channel: Record<string, number>;

  time_statistics: Record<string, TimeStats>;
  episodes_per_sheet: Record<string, number>;
  notes: string[];
}

export interface SourceMetadata {
  filename: string;
  size_bytes: number;
  sha256: string | null;
  sheet_count: number;
  sheet_names: string[];
}

export interface SheetMapping {
  header_rows: number[];
  data_start_row: number | null;
  roles: Partial<Record<HiddenGroup, string[]>>;
}

export interface ColumnMappingConfig {
  version: string;
  sheets: Record<string, SheetMapping>;
}

export interface SheetColumnInfo {
  name: string;
  levels: string[];
  dtype: string;
  non_null_count: number;
  null_count: number;
  sample_values: unknown[];
  role_hint: HiddenGroup | null;
}

export interface SheetColumnsResponse {
  sheet: string;
  header_rows: number[];
  columns: SheetColumnInfo[];
}

export type HMMVariant = "basic_3state" | "detailed_7state";
export type HMMMode = "auto" | "detailed" | "basic" | "off";

export interface HMMParameters {
  n_states: number;
  n_observations: number;
  state_labels: string[];
  observation_labels: string[];
  initial_distribution: number[];
  transition_matrix: number[][];
  emission_matrix: number[][];
  random_seed: number;
  n_iter: number;
  converged: boolean;
  log_likelihood: number;
  variant: HMMVariant;
  bic: number | null;
}

export interface HMMTrajectory {
  sheet: string;
  episode_index: number;
  length: number;
  observation_tokens: string[];
  state_path: string[];
  log_likelihood: number;
}

export interface HMMResult {
  parameters: HMMParameters;
  trajectories: HMMTrajectory[];
  state_distribution: Record<string, number>;
  sanity: Record<string, unknown>;
  interpretation: string;
}

export interface AnalysisResult {
  status: AnalysisStatus;
  generated_at: string;
  algo_version: string;
  source_metadata: SourceMetadata;
  data_audit: AuditReport;
  detected_columns: ColumnDetectionReport;
  basic_statistics: BaselineReport;
  applied_mapping: ColumnMappingConfig | null;
  hmm: HMMResult | null;
  charts: ChartData[];
  warnings: WarningItem[];
  errors: WarningItem[];
  report: string;
}

export interface AnalysisRunFull {
  id: number;
  source_id: number;
  status: AnalysisStatus;
  algo_version: string;
  created_at: string;
  result: AnalysisResult;
}

export interface AnalysisRunSummary {
  id: number;
  source_id: number;
  state: "pending" | "running" | "done" | "failed";
  status: AnalysisStatus | "";
  algo_version: string;
  hmm_mode: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export interface SheetPreview {
  sheet: string;
  header_rows: number[];
  columns: string[];
  preview: Record<string, unknown>[];
}

export type GridCellValue = string | number | boolean | null;

export interface SheetGridFragment {
  sheet: string;
  start_row: number;
  start_col: number;
  n_rows: number;
  n_cols: number;
  total_rows: number;
  total_cols: number;
  cells: GridCellValue[][];
}

export interface CellEdit {
  row: number;
  col: number;
  value: GridCellValue;
}

export interface ApiError {
  detail: string;
}
