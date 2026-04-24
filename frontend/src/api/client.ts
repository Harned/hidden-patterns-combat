import type {
  AnalysisRunFull,
  AnalysisRunSummary,
  ColumnMappingConfig,
  SheetColumnsResponse,
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

async function request<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(init.headers || {}),
    },
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
  register(email: string, password: string) {
    return request<UserPublic>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
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
  listSources() {
    return request<SourceSummary[]>("/sources");
  },
  getSource(id: number) {
    return request<SourceSummary>(`/sources/${id}`);
  },
  deleteSource(id: number) {
    return request<void>(`/sources/${id}`, { method: "DELETE" });
  },
  async uploadSource(file: File): Promise<SourceSummary> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/sources`, {
      method: "POST",
      credentials: "include",
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
  preflight(sourceId: number) {
    return request<{ mapping: ColumnMappingConfig | null }>(
      `/sources/${sourceId}/preflight`,
      { method: "POST" }
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
};
