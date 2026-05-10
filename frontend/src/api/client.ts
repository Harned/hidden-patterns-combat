import type {
  AnalysisRunFull,
  AnalysisRunSummary,
  AthleteForwardFillResponse,
  CellEdit,
  ColumnMappingConfig,
  HeaderMergeFillResponse,
  HeaderRowsSuggestionResponse,
  MarkovResult,
  SheetColumnsResponse,
  SheetGridFragment,
  SheetPreview,
  SourceSummary,
  UserPublic,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function readCsrfTokenFromCookie(): string | null {
  const match = document.cookie
    .split("; ")
    .find((row) => row.startsWith("hpc_csrf="));
  return match ? decodeURIComponent(match.split("=")[1]) : null;
}

const MUTATING = new Set(["POST", "PUT", "DELETE", "PATCH"]);

function withCsrfHeader(
  method: string | undefined,
  headers: HeadersInit
): HeadersInit {
  if (!method || !MUTATING.has(method.toUpperCase())) return headers;
  const token = readCsrfTokenFromCookie();
  if (!token) return headers;
  return { ...headers, "X-CSRF-Token": token };
}

async function request<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const baseHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...(init.headers as Record<string, string> | undefined),
  };
  const res = await fetch(`${BASE}${path}`, {
    credentials: "include",
    headers: withCsrfHeader(init.method, baseHeaders),
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // no body
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  register(
    email: string,
    password: string,
    accept_terms: boolean,
    accept_pdn: boolean
  ) {
    return request<UserPublic>("/auth/register", {
      method: "POST",
      body: JSON.stringify({
        email,
        password,
        accept_terms,
        accept_pdn,
      }),
    });
  },
  login(email: string, password: string) {
    return request<UserPublic>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  },
  logout() {
    return request<void>("/auth/logout", { method: "POST" });
  },
  me() {
    return request<UserPublic>("/auth/me");
  },
  verifyEmail(code: string) {
    return request<UserPublic>("/auth/verify-email", {
      method: "POST",
      body: JSON.stringify({ code }),
    });
  },
  resendVerification() {
    return request<{ status: string }>("/auth/resend-verification", {
      method: "POST",
    });
  },
  forgotPassword(email: string) {
    return request<{ status: string; message: string }>(
      "/auth/forgot-password",
      {
        method: "POST",
        body: JSON.stringify({ email }),
      }
    );
  },
  resetPassword(
    email: string,
    code: string,
    new_password: string,
    new_password_repeat: string
  ) {
    return request<{ status: string }>("/auth/reset-password", {
      method: "POST",
      body: JSON.stringify({
        email,
        code,
        new_password,
        new_password_repeat,
      }),
    });
  },
  onboardingComplete() {
    return request<UserPublic>("/auth/onboarding-complete", { method: "POST" });
  },
  deleteAccount() {
    return request<void>("/auth/me", { method: "DELETE" });
  },
  listSources() {
    return request<SourceSummary[]>("/sources");
  },
  getSource(id: number) {
    return request<SourceSummary>(`/sources/${id}`);
  },
  deleteSource(id: number) {
    return request<void>(`/sources/${id}`, { method: "DELETE" });
  },
  async uploadSource(
    file: File,
    confirmUpload: boolean
  ): Promise<SourceSummary> {
    const form = new FormData();
    form.append("file", file);
    form.append("confirm_upload", confirmUpload ? "true" : "false");
    const headers: Record<string, string> = {};
    const csrf = readCsrfTokenFromCookie();
    if (csrf) headers["X-CSRF-Token"] = csrf;
    const res = await fetch(`${BASE}/sources`, {
      method: "POST",
      credentials: "include",
      headers,
      body: form,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body?.detail ?? detail;
      } catch {
        // no body
      }
      throw new ApiError(res.status, detail);
    }
    return (await res.json()) as SourceSummary;
  },
  analyze(sourceId: number, mode: "auto" | "detailed" | "basic" | "off" = "auto") {
    const qs = `?mode=${encodeURIComponent(mode)}`;
    return request<AnalysisRunSummary>(`/sources/${sourceId}/analyze${qs}`, {
      method: "POST",
    });
  },
  latestResult(sourceId: number) {
    return request<AnalysisRunFull>(`/sources/${sourceId}/result`);
  },
  runMarkov(sourceId: number) {
    return request<MarkovResult>(`/sources/${sourceId}/markov`, { method: "POST" });
  },
  preflight(sourceId: number, sheetNames?: string[]) {
    const body =
      sheetNames && sheetNames.length > 0
        ? JSON.stringify({ sheet_names: sheetNames })
        : undefined;
    return request<{ mapping: ColumnMappingConfig | null }>(
      `/sources/${sourceId}/preflight`,
      { method: "POST", body }
    );
  },
  finalizeSource(sourceId: number) {
    return request<SourceSummary>(`/sources/${sourceId}/finalize`, {
      method: "POST",
    });
  },
  listSheetNames(sourceId: number) {
    return request<{ sheet_names: string[] }>(`/sources/${sourceId}/sheets`);
  },
  readSheetGrid(
    sourceId: number,
    sheetName: string,
    params: { startRow: number; startCol: number; nRows: number; nCols: number }
  ) {
    const qs = new URLSearchParams({
      start_row: String(params.startRow),
      start_col: String(params.startCol),
      n_rows: String(params.nRows),
      n_cols: String(params.nCols),
    }).toString();
    return request<SheetGridFragment>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(sheetName)}/grid?${qs}`
    );
  },
  applySheetGridEdits(
    sourceId: number,
    sheetName: string,
    edits: CellEdit[]
  ) {
    return request<{ applied: number }>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(sheetName)}/grid`,
      { method: "PUT", body: JSON.stringify({ edits }) }
    );
  },
  removeEmptyRows(
    sourceId: number,
    sheetName: string,
    headerRows?: number[]
  ) {
    return request<{ deleted: number }>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(
        sheetName
      )}/remove-empty-rows`,
      {
        method: "POST",
        body: JSON.stringify({ header_rows: headerRows ?? null }),
      }
    );
  },
  countEmptyRows(
    sourceId: number,
    sheetName: string,
    headerRows?: number[]
  ) {
    const qs =
      headerRows && headerRows.length > 0
        ? `?header_rows=${headerRows.join(",")}`
        : "";
    return request<{ count: number }>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(
        sheetName
      )}/empty-rows-count${qs}`
    );
  },
  headerRowsSuggestion(
    sourceId: number,
    sheetName: string,
    headerRows?: number[]
  ) {
    const qs =
      headerRows && headerRows.length > 0
        ? `?header_rows=${headerRows.join(",")}`
        : "";
    return request<HeaderRowsSuggestionResponse>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(
        sheetName
      )}/suggestions/header-rows${qs}`
    );
  },
  athleteForwardFillSuggestions(
    sourceId: number,
    sheetName: string,
    headerRows?: number[]
  ) {
    const qs =
      headerRows && headerRows.length > 0
        ? `?header_rows=${headerRows.join(",")}`
        : "";
    return request<AthleteForwardFillResponse>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(
        sheetName
      )}/suggestions/athlete-forward-fill${qs}`
    );
  },
  headerMergeFillSuggestions(
    sourceId: number,
    sheetName: string,
    headerRows?: number[]
  ) {
    const qs =
      headerRows && headerRows.length > 0
        ? `?header_rows=${headerRows.join(",")}`
        : "";
    return request<HeaderMergeFillResponse>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(
        sheetName
      )}/suggestions/header-merge-fill${qs}`
    );
  },
  getMapping(sourceId: number) {
    return request<{ mapping: ColumnMappingConfig | null }>(
      `/sources/${sourceId}/mapping`
    );
  },
  putMapping(sourceId: number, mapping: ColumnMappingConfig) {
    return request<{ mapping: ColumnMappingConfig | null }>(
      `/sources/${sourceId}/mapping`,
      {
        method: "PUT",
        body: JSON.stringify(mapping),
      }
    );
  },
  deleteMapping(sourceId: number) {
    return request<void>(`/sources/${sourceId}/mapping`, { method: "DELETE" });
  },
  listSheetColumns(
    sourceId: number,
    sheetName: string,
    headerRows: number[]
  ) {
    const qs =
      headerRows.length > 0
        ? `?header_rows=${headerRows.join(",")}`
        : "";
    return request<SheetColumnsResponse>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(sheetName)}/columns${qs}`
    );
  },
  sheetPreview(
    sourceId: number,
    sheetName: string,
    headerRows: number[],
    rows = 10
  ) {
    const parts: string[] = [];
    if (headerRows.length > 0) parts.push(`header_rows=${headerRows.join(",")}`);
    parts.push(`rows=${rows}`);
    const qs = parts.length ? `?${parts.join("&")}` : "";
    return request<SheetPreview>(
      `/sources/${sourceId}/sheets/${encodeURIComponent(sheetName)}/preview${qs}`
    );
  },
  listRuns(sourceId: number, limit = 20) {
    return request<AnalysisRunSummary[]>(
      `/sources/${sourceId}/runs?limit=${limit}`
    );
  },
  getRunResult(sourceId: number, runId: number) {
    return request<AnalysisRunFull>(
      `/sources/${sourceId}/runs/${runId}/result`
    );
  },
};
