import React, { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type {
  ColumnMappingConfig,
  HiddenGroup,
  SheetColumnInfo,
} from "@/api/types";
import { Badge, Button, Card, Section } from "@/components/ui";

const ROLE_OPTIONS: { value: HiddenGroup | "__none__"; label: string }[] = [
  { value: "__none__", label: "— (игнорировать)" },
  { value: "ЗАП", label: "ЗАП (observations)" },
  { value: "маневрирование", label: "маневрирование" },
  { value: "КФВ", label: "КФВ" },
  { value: "ВУП", label: "ВУП" },
  { value: "time", label: "time (время)" },
  { value: "athlete", label: "athlete (спортсмен)" },
  { value: "episode", label: "episode (эпизод)" },
  { value: "bout", label: "bout (схватка)" },
  { value: "weight", label: "weight (вес)" },
];

interface Props {
  sourceId: number;
}

function cloneMapping(m: ColumnMappingConfig): ColumnMappingConfig {
  return JSON.parse(JSON.stringify(m));
}

function mapColumnsToRoles(
  roles: Partial<Record<HiddenGroup, string[]>>
): Record<string, HiddenGroup> {
  const out: Record<string, HiddenGroup> = {};
  for (const [role, cols] of Object.entries(roles) as [HiddenGroup, string[]][]) {
    for (const col of cols ?? []) out[col] = role;
  }
  return out;
}

function rolesFromColumnMap(
  columnToRole: Record<string, HiddenGroup | "__none__">
): Partial<Record<HiddenGroup, string[]>> {
  const roles: Partial<Record<HiddenGroup, string[]>> = {};
  for (const [col, role] of Object.entries(columnToRole)) {
    if (role === "__none__") continue;
    (roles[role] ??= []).push(col);
  }
  return roles;
}

function shortSamples(sample: unknown[]): string {
  if (!sample || sample.length === 0) return "—";
  return sample
    .slice(0, 3)
    .map((v) => (typeof v === "string" ? v : JSON.stringify(v)))
    .join(", ");
}

export const MappingEditor: React.FC<Props> = ({ sourceId }) => {
  const qc = useQueryClient();

  const savedQuery = useQuery({
    queryKey: ["mapping", sourceId],
    queryFn: () => api.getMapping(sourceId),
  });

  const [draft, setDraft] = useState<ColumnMappingConfig | null>(null);
  const [activeSheet, setActiveSheet] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (savedQuery.data?.mapping && !draft) {
      setDraft(cloneMapping(savedQuery.data.mapping));
    }
  }, [savedQuery.data, draft]);

  useEffect(() => {
    if (draft && !activeSheet) {
      const firstSheet = Object.keys(draft.sheets)[0] ?? null;
      setActiveSheet(firstSheet);
    }
  }, [draft, activeSheet]);

  const sheetMapping = draft && activeSheet ? draft.sheets[activeSheet] : null;
  const headerRowsKey = sheetMapping?.header_rows.join(",") ?? "";

  const columnsQuery = useQuery({
    queryKey: [
      "sheetColumns",
      sourceId,
      activeSheet,
      headerRowsKey,
    ],
    queryFn: () =>
      api.listSheetColumns(
        sourceId,
        activeSheet!,
        sheetMapping?.header_rows ?? [0]
      ),
    enabled: Boolean(activeSheet && sheetMapping),
    retry: false,
  });

  const previewQuery = useQuery({
    queryKey: ["sheetPreview", sourceId, activeSheet, headerRowsKey],
    queryFn: () =>
      api.sheetPreview(
        sourceId,
        activeSheet!,
        sheetMapping?.header_rows ?? [0],
        8
      ),
    enabled: Boolean(activeSheet && sheetMapping),
    retry: false,
  });

  const preflightMut = useMutation({
    mutationFn: () => api.preflight(sourceId),
    onSuccess: (res) => {
      if (res.mapping) {
        setDraft(cloneMapping(res.mapping));
        setActiveSheet(Object.keys(res.mapping.sheets)[0] ?? null);
        setMessage("Preflight выполнен. Проверьте и сохраните.");
        void qc.invalidateQueries({ queryKey: ["sheetColumns", sourceId] });
      }
    },
    onError: (err) => {
      setMessage(
        err instanceof ApiError
          ? `Ошибка: ${err.message}`
          : "Не удалось выполнить preflight"
      );
    },
  });

  const saveMut = useMutation({
    mutationFn: (m: ColumnMappingConfig) => api.putMapping(sourceId, m),
    onSuccess: () => {
      setMessage("Сопоставление сохранено.");
      void qc.invalidateQueries({ queryKey: ["mapping", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sources"] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
    },
    onError: (err) => {
      setMessage(
        err instanceof ApiError
          ? `Ошибка: ${err.message}`
          : "Не удалось сохранить mapping"
      );
    },
  });

  const deleteMut = useMutation({
    mutationFn: () => api.deleteMapping(sourceId),
    onSuccess: () => {
      setDraft(null);
      setActiveSheet(null);
      setMessage("Сопоставление удалено.");
      void qc.invalidateQueries({ queryKey: ["mapping", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const analyzeMut = useMutation({
    mutationFn: () => api.analyze(sourceId),
    onSuccess: () => {
      setMessage("Анализ перезапущен.");
      void qc.invalidateQueries({ queryKey: ["result", sourceId] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const sheetKeys = draft ? Object.keys(draft.sheets) : [];

  const columnToRole = useMemo(
    () => (sheetMapping ? mapColumnsToRoles(sheetMapping.roles) : {}),
    [sheetMapping]
  );

  const allColumns: SheetColumnInfo[] = columnsQuery.data?.columns ?? [];

  // Колонки, на которые mapping ссылается, но их нет в текущей выборке
  // (например, preflight использовал другие header_rows).
  const stale = useMemo(() => {
    if (!sheetMapping || !allColumns.length) return [] as string[];
    const known = new Set(allColumns.map((c) => c.name));
    const referenced = Object.values(sheetMapping.roles).flat();
    return referenced.filter((name) => !known.has(name));
  }, [sheetMapping, allColumns]);

  const changeRole = (column: string, role: HiddenGroup | "__none__") => {
    if (!draft || !activeSheet) return;
    const base: Record<string, HiddenGroup | "__none__"> = { ...columnToRole };
    base[column] = role;
    const nextRoles = rolesFromColumnMap(base);
    const next = cloneMapping(draft);
    next.sheets[activeSheet] = { ...next.sheets[activeSheet], roles: nextRoles };
    setDraft(next);
  };

  const changeHeaderRows = (value: string) => {
    if (!draft || !activeSheet) return;
    const parsed = value
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => Number(s))
      .filter((n) => Number.isInteger(n) && n >= 0);
    const next = cloneMapping(draft);
    next.sheets[activeSheet] = {
      ...next.sheets[activeSheet],
      header_rows: parsed.length ? parsed : [0],
    };
    setDraft(next);
  };

  const roleCountsForSheet = useMemo(() => {
    if (!sheetMapping) return [] as Array<[HiddenGroup, number]>;
    return Object.entries(sheetMapping.roles).map(
      ([role, cols]) => [role as HiddenGroup, (cols ?? []).length] as const
    );
  }, [sheetMapping]);

  if (savedQuery.isLoading) {
    return (
      <Card className="px-6 py-8 text-center text-brand-700/70">
        Загрузка сохранённого mapping...
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="px-6 py-5 flex flex-wrap items-center gap-3 justify-between">
        <div>
          <div className="text-base font-semibold text-brand-900">
            Сопоставление колонок
          </div>
          <p className="mt-1 text-sm text-brand-700/80">
            Preflight предлагает стартовое сопоставление на основе заголовков.
            Вы можете изменить роль любой колонки листа и сохранить mapping.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            onClick={() => preflightMut.mutate()}
            disabled={preflightMut.isPending}
          >
            {preflightMut.isPending
              ? "Preflight..."
              : draft
              ? "Пересобрать preflight"
              : "Запустить preflight"}
          </Button>
          <Button
            onClick={() => draft && saveMut.mutate(draft)}
            disabled={!draft || saveMut.isPending}
          >
            {saveMut.isPending ? "Сохраняем..." : "Сохранить"}
          </Button>
          <Button
            onClick={async () => {
              if (!draft) return;
              await saveMut.mutateAsync(draft);
              analyzeMut.mutate();
            }}
            disabled={!draft || saveMut.isPending || analyzeMut.isPending}
          >
            {analyzeMut.isPending
              ? "Анализ..."
              : "Сохранить и запустить анализ"}
          </Button>
          <Button
            variant="ghost"
            onClick={() => deleteMut.mutate()}
            disabled={!savedQuery.data?.mapping || deleteMut.isPending}
          >
            Сбросить
          </Button>
        </div>
      </Card>

      {message && (
        <div className="rounded-md border border-brand-200 bg-brand-50 px-4 py-2 text-sm text-brand-900">
          {message}
        </div>
      )}

      {!draft && (
        <Card className="px-6 py-10 text-center text-brand-700/70">
          Mapping не настроен. Запустите preflight, чтобы получить стартовое
          сопоставление.
        </Card>
      )}

      {draft && (
        <Section
          title="Листы"
          description="Выберите лист, чтобы отредактировать его колонки и строки заголовка."
        >
          <div className="flex flex-wrap gap-2 mb-4">
            {sheetKeys.map((name) => (
              <button
                key={name}
                onClick={() => setActiveSheet(name)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium border ${
                  name === activeSheet
                    ? "bg-brand-100 text-brand-900 border-brand-300"
                    : "bg-white text-brand-800 border-brand-200 hover:bg-brand-50"
                }`}
              >
                {name}
              </button>
            ))}
          </div>

          {sheetMapping && activeSheet && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-3">
                <label className="text-sm text-brand-900">
                  header_rows (через запятую, 0-индекс):
                </label>
                <input
                  value={sheetMapping.header_rows.join(", ")}
                  onChange={(e) => changeHeaderRows(e.target.value)}
                  className="h-9 rounded-md border border-brand-200 px-3 text-sm font-mono w-56"
                />
                <div className="flex gap-2 flex-wrap">
                  {roleCountsForSheet.map(([role, count]) => (
                    <Badge key={role} tone="info">
                      {role}: {count}
                    </Badge>
                  ))}
                </div>
                <div className="ml-auto text-xs text-brand-700/60">
                  {columnsQuery.isLoading
                    ? "Загрузка колонок..."
                    : `Показано колонок: ${allColumns.length}`}
                </div>
              </div>

              {stale.length > 0 && (
                <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-900">
                  В mapping остались {stale.length} колонок, которых нет при
                  текущих header_rows — они будут проигнорированы при анализе
                  (warning <code>mapping.unknown_column</code>).
                </div>
              )}

              {previewQuery.data && (
                <details className="rounded-md border border-brand-100 bg-white">
                  <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-brand-900">
                    Примеры данных (первые {previewQuery.data.preview.length} строк)
                  </summary>
                  <div className="overflow-x-auto max-h-64 overflow-y-auto border-t border-brand-100">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-left text-brand-700/70 bg-brand-50">
                          {previewQuery.data.columns.slice(0, 12).map((c) => (
                            <th
                              key={c}
                              className="py-1.5 px-2 font-medium whitespace-nowrap max-w-[200px] truncate"
                              title={c}
                            >
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewQuery.data.preview.map((row, i) => (
                          <tr key={i} className="border-t border-brand-100 align-top">
                            {previewQuery.data.columns.slice(0, 12).map((c) => {
                              const v = row[c];
                              return (
                                <td
                                  key={c}
                                  className="py-1.5 px-2 font-mono max-w-[200px] truncate"
                                  title={v == null ? "" : String(v)}
                                >
                                  {v == null ? "—" : String(v)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {previewQuery.data.columns.length > 12 && (
                    <div className="px-3 py-1.5 text-[11px] text-brand-700/60">
                      Показаны первые 12 колонок из {previewQuery.data.columns.length}.
                    </div>
                  )}
                </details>
              )}

              <div className="overflow-x-auto rounded-md border border-brand-100">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-brand-700/70 bg-brand-50">
                      <th className="py-2 px-3 font-medium">Колонка (flatten)</th>
                      <th className="py-2 px-3 font-medium w-64">Примеры</th>
                      <th className="py-2 px-3 font-medium w-72">Роль</th>
                    </tr>
                  </thead>
                  <tbody>
                    {columnsQuery.isError && (
                      <tr>
                        <td className="px-3 py-4 text-red-700" colSpan={3}>
                          Не удалось загрузить колонки листа.
                        </td>
                      </tr>
                    )}
                    {!columnsQuery.isLoading &&
                      allColumns.length === 0 &&
                      !columnsQuery.isError && (
                        <tr>
                          <td
                            className="px-3 py-4 text-brand-700/60"
                            colSpan={3}
                          >
                            На этом листе не найдено колонок.
                          </td>
                        </tr>
                      )}
                    {allColumns.map((col) => {
                      const currentRole =
                        columnToRole[col.name] ?? "__none__";
                      return (
                        <tr
                          key={col.name}
                          className="border-t border-brand-100 align-top"
                        >
                          <td className="py-2 px-3 font-mono text-xs">
                            <div>{col.name}</div>
                            <div className="mt-0.5 text-[10px] text-brand-700/60">
                              {col.dtype} · заполнено {col.non_null_count}
                              {col.role_hint && (
                                <>
                                  {" · подсказка: "}
                                  <span className="text-brand-900">
                                    {col.role_hint}
                                  </span>
                                </>
                              )}
                            </div>
                          </td>
                          <td className="py-2 px-3 text-xs text-brand-900/80 break-words">
                            {shortSamples(col.sample_values)}
                          </td>
                          <td className="py-2 px-3">
                            <select
                              value={currentRole}
                              onChange={(e) =>
                                changeRole(
                                  col.name,
                                  e.target.value as HiddenGroup | "__none__"
                                )
                              }
                              className="h-8 w-full rounded-md border border-brand-200 px-2 text-sm"
                            >
                              {ROLE_OPTIONS.map((opt) => (
                                <option key={opt.value} value={opt.value}>
                                  {opt.label}
                                </option>
                              ))}
                            </select>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Section>
      )}
    </div>
  );
};
