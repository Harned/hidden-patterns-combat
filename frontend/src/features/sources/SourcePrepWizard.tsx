import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type {
  CellEdit,
  ColumnMappingConfig,
  HeaderMergeFillResponse,
  HeaderRowsSuggestionResponse,
  SheetMapping,
  SourceSummary,
} from "@/api/types";
import { Badge, Button, Card, Section } from "@/components/ui";
import { SheetGrid, type CellSuggestion, type SheetGridHandle } from "./SheetGrid";

const MappingEditor = React.lazy(() =>
  import("@/features/analysis/MappingEditor").then((m) => ({
    default: m.MappingEditor,
  }))
);

type Step = "sheets" | "prepare" | "finalize";

const STEPS: { id: Step; label: string; description: string }[] = [
  {
    id: "sheets",
    label: "Листы",
    description: "Выберите листы Excel, которые должны участвовать в анализе.",
  },
  {
    id: "prepare",
    label: "Данные и колонки",
    description:
      "Сначала проверьте и при необходимости подчистите данные листа, затем назначьте роли колонкам. Все правки сохраняются в исходный файл.",
  },
  {
    id: "finalize",
    label: "Подтверждение",
    description:
      "Финальная проверка. После подтверждения источник становится доступен для анализа.",
  },
];

export const SourcePrepWizard: React.FC = () => {
  const { sourceId: idParam } = useParams<{ sourceId: string }>();
  const sourceId = Number(idParam);
  const navigate = useNavigate();
  const qc = useQueryClient();

  const [step, setStep] = useState<Step>("sheets");
  const [includedSheets, setIncludedSheets] = useState<Set<string> | null>(
    null
  );
  const [activeSheet, setActiveSheet] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sourceQuery = useQuery<SourceSummary>({
    queryKey: ["source", sourceId],
    queryFn: () => api.getSource(sourceId),
    enabled: Number.isFinite(sourceId) && sourceId > 0,
    retry: false,
  });

  const sheetsQuery = useQuery({
    queryKey: ["sheetNames", sourceId],
    queryFn: () => api.listSheetNames(sourceId),
    enabled: Number.isFinite(sourceId) && sourceId > 0,
  });

  const mappingQuery = useQuery({
    queryKey: ["mapping", sourceId],
    queryFn: () => api.getMapping(sourceId),
    enabled: Number.isFinite(sourceId) && sourceId > 0,
  });

  useEffect(() => {
    if (!sheetsQuery.data || includedSheets) return;
    const all = sheetsQuery.data.sheet_names;
    const fromMapping = mappingQuery.data?.mapping?.sheets;
    const initial = fromMapping
      ? new Set(Object.keys(fromMapping).filter((s) => all.includes(s)))
      : new Set(all);
    setIncludedSheets(initial);
  }, [sheetsQuery.data, mappingQuery.data, includedSheets]);

  const sheetNames = sheetsQuery.data?.sheet_names ?? [];
  const mapping: ColumnMappingConfig | null =
    mappingQuery.data?.mapping ?? null;

  const includedList = useMemo(
    () => sheetNames.filter((s) => includedSheets?.has(s)),
    [sheetNames, includedSheets]
  );

  const mappedSheets = useMemo(
    () => (mapping ? Object.keys(mapping.sheets) : []),
    [mapping]
  );

  useEffect(() => {
    if (mappedSheets.length === 0) {
      if (activeSheet !== null) setActiveSheet(null);
      return;
    }
    if (!activeSheet || !mappedSheets.includes(activeSheet)) {
      setActiveSheet(mappedSheets[0]);
    }
  }, [mappedSheets, activeSheet]);

  const preflightMut = useMutation({
    mutationFn: (sheets: string[]) => api.preflight(sourceId, sheets),
    onSuccess: (res) => {
      if (res.mapping) {
        api
          .putMapping(sourceId, res.mapping)
          .then(() => {
            void qc.invalidateQueries({ queryKey: ["mapping", sourceId] });
            void qc.invalidateQueries({ queryKey: ["sources"] });
          })
          .catch(() => undefined);
      }
    },
    onError: (err) =>
      setError(
        err instanceof ApiError
          ? `Preflight: ${err.message}`
          : "Не удалось выполнить preflight."
      ),
  });

  const finalizeMut = useMutation({
    mutationFn: () => api.finalizeSource(sourceId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["sources"] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
      navigate("/app", { state: { selectSourceId: sourceId } });
    },
    onError: (err) =>
      setError(
        err instanceof ApiError
          ? `Подтверждение: ${err.message}`
          : "Не удалось подтвердить источник."
      ),
  });

  const cancelMut = useMutation({
    mutationFn: () => api.deleteSource(sourceId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["sources"] });
      navigate("/app");
    },
  });

  const goNext = async () => {
    setError(null);
    if (step === "sheets") {
      if (!includedList.length) {
        setError("Выберите хотя бы один лист.");
        return;
      }
      const sameSet =
        mappedSheets.length === includedList.length &&
        includedList.every((s) => mappedSheets.includes(s));
      if (!sameSet) {
        await preflightMut.mutateAsync(includedList).catch(() => undefined);
      }
      setStep("prepare");
      return;
    }
    if (step === "prepare") {
      if (!mapping || Object.keys(mapping.sheets).length === 0) {
        setError("Сохраните column mapping перед переходом дальше.");
        return;
      }
      setStep("finalize");
      return;
    }
    if (step === "finalize") {
      finalizeMut.mutate();
    }
  };

  const goBack = () => {
    setError(null);
    const idx = STEPS.findIndex((s) => s.id === step);
    if (idx > 0) setStep(STEPS[idx - 1].id);
  };

  if (sourceQuery.isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-brand-700/70">
        Загрузка источника...
      </div>
    );
  }
  if (sourceQuery.isError || !sourceQuery.data) {
    return (
      <div className="h-full flex items-center justify-center text-red-700">
        Источник не найден.
      </div>
    );
  }

  const source = sourceQuery.data;
  const isReady = source.preparation_state === "ready";

  return (
    <div className="h-full flex flex-col bg-brand-50/40">
      <header className="flex items-center justify-between px-6 py-4 border-b border-brand-200 bg-white">
        <div className="min-w-0">
          <div className="text-xs text-brand-700/70 uppercase tracking-wide">
            Мастер предобработки
          </div>
          <div className="text-lg font-semibold text-brand-900 truncate">
            {source.original_filename}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isReady ? (
            <Badge tone="success">подтверждён</Badge>
          ) : (
            <Badge tone="warning">черновик</Badge>
          )}
          <Button
            variant="ghost"
            onClick={() => {
              if (
                confirm(
                  "Отменить добавление источника? Файл и черновик будут удалены."
                )
              ) {
                cancelMut.mutate();
              }
            }}
            disabled={cancelMut.isPending}
          >
            Отменить
          </Button>
        </div>
      </header>

      <nav className="flex items-center gap-1 px-6 py-3 border-b border-brand-200 bg-white">
        {STEPS.map((s, idx) => {
          const active = s.id === step;
          const done = STEPS.findIndex((x) => x.id === step) > idx;
          return (
            <React.Fragment key={s.id}>
              <button
                onClick={() => !isReady && setStep(s.id)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                  active
                    ? "bg-brand-100 text-brand-900"
                    : done
                    ? "text-brand-700"
                    : "text-brand-700/60 hover:text-brand-900"
                }`}
                disabled={isReady}
              >
                {idx + 1}. {s.label}
              </button>
              {idx < STEPS.length - 1 && (
                <span className="text-brand-300">›</span>
              )}
            </React.Fragment>
          );
        })}
      </nav>

      <main className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        <div className="text-sm text-brand-700/80">
          {STEPS.find((s) => s.id === step)?.description}
        </div>

        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-800">
            {error}
          </div>
        )}

        {step === "sheets" && (
          <Section title="Листы файла" description="Снимите галочку у листов, которые не должны попасть в анализ.">
            {sheetsQuery.isLoading && (
              <div className="text-sm text-brand-700/70">Чтение списка листов...</div>
            )}
            {sheetsQuery.isError && (
              <div className="text-sm text-red-700">
                Не удалось прочитать список листов.
              </div>
            )}
            {includedSheets && (
              <ul className="space-y-1">
                {sheetNames.map((name) => {
                  const included = includedSheets.has(name);
                  return (
                    <li
                      key={name}
                      className="flex items-center gap-3 rounded-md px-3 py-2 hover:bg-brand-50"
                    >
                      <input
                        type="checkbox"
                        id={`sheet-${name}`}
                        checked={included}
                        onChange={() => {
                          setIncludedSheets((prev) => {
                            const next = new Set(prev);
                            if (next.has(name)) next.delete(name);
                            else next.add(name);
                            return next;
                          });
                        }}
                        className="h-4 w-4"
                      />
                      <label
                        htmlFor={`sheet-${name}`}
                        className="flex-1 text-sm font-medium text-brand-900 cursor-pointer"
                      >
                        {name}
                      </label>
                      {included ? (
                        <Badge tone="success">включён</Badge>
                      ) : (
                        <Badge tone="neutral">исключён</Badge>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </Section>
        )}

        {step === "prepare" && (
          <PrepareStep
            sourceId={sourceId}
            mapping={mapping}
            activeSheet={activeSheet}
            setActiveSheet={setActiveSheet}
          />
        )}

        {step === "finalize" && (
          <Section
            title="Готово к подтверждению"
            description="После подтверждения источник станет доступен для запуска анализа. Правки данных и mapping вы сможете изменить, но это создаст новый запуск анализа."
          >
            <FinalizeSummary mapping={mapping} sheetNames={includedList} />
          </Section>
        )}
      </main>

      <footer className="flex items-center justify-between gap-3 px-6 py-4 border-t border-brand-200 bg-white">
        <Button variant="ghost" onClick={goBack} disabled={step === "sheets"}>
          Назад
        </Button>
        <div className="flex items-center gap-2">
          <Button
            onClick={goNext}
            disabled={
              preflightMut.isPending ||
              finalizeMut.isPending ||
              cancelMut.isPending
            }
          >
            {step === "finalize"
              ? finalizeMut.isPending
                ? "Подтверждаем..."
                : "Подтвердить и добавить"
              : preflightMut.isPending
              ? "Preflight..."
              : "Дальше"}
          </Button>
        </div>
      </footer>
    </div>
  );
};

const PrepareStep: React.FC<{
  sourceId: number;
  mapping: ColumnMappingConfig | null;
  activeSheet: string | null;
  setActiveSheet: (s: string | null) => void;
}> = ({ sourceId, mapping, activeSheet, setActiveSheet }) => {
  const qc = useQueryClient();
  const sheets = mapping ? Object.keys(mapping.sheets) : [];
  const gridRef = useRef<SheetGridHandle>(null);
  const [gridDirty, setGridDirty] = useState(0);

  const sheetMapping: SheetMapping | null =
    activeSheet && mapping ? mapping.sheets[activeSheet] : null;
  const headerRows = sheetMapping?.header_rows ?? [0];
  const athleteRoleColumns = sheetMapping?.roles?.athlete ?? [];
  const hasAthleteRole = athleteRoleColumns.length > 0;
  const hasEpisodeColumn = (sheetMapping?.roles?.episode?.length ?? 0) > 0;

  const emptyCountQuery = useQuery({
    queryKey: ["emptyRows", sourceId, activeSheet, headerRows.join(",")],
    queryFn: () =>
      api.countEmptyRows(sourceId, activeSheet!, headerRows),
    enabled: Boolean(activeSheet && sheetMapping),
    retry: 1,
    staleTime: 0,
  });

  const athleteSuggestionsQuery = useQuery({
    queryKey: [
      "athleteFwdFill",
      sourceId,
      activeSheet,
      headerRows.join(","),
      athleteRoleColumns.join("|"),
    ],
    queryFn: () =>
      api.athleteForwardFillSuggestions(sourceId, activeSheet!, headerRows),
    enabled: Boolean(activeSheet && sheetMapping && hasAthleteRole),
    retry: false,
    staleTime: 0,
  });

  const headerRowsKey = headerRows.join(",");
  const headerSuggestionQuery = useQuery({
    queryKey: ["headerSuggestion", sourceId, activeSheet, headerRowsKey],
    queryFn: () =>
      api.headerRowsSuggestion(
        sourceId,
        activeSheet!,
        sheetMapping?.header_rows ?? [0]
      ),
    enabled: Boolean(activeSheet && sheetMapping),
    retry: false,
    staleTime: 30_000,
  });

  const applyHeaderRowsMut = useMutation({
    mutationFn: async (rows: number[]) => {
      if (!mapping || !activeSheet) {
        throw new Error("Нет активного листа или mapping.");
      }
      const next: ColumnMappingConfig = JSON.parse(JSON.stringify(mapping));
      next.sheets[activeSheet] = {
        ...next.sheets[activeSheet],
        header_rows: [...rows],
      };
      await api.putMapping(sourceId, next);
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["mapping", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sheetColumns", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sheetPreview", sourceId] });
      void qc.invalidateQueries({ queryKey: ["grid", sourceId] });
      void qc.invalidateQueries({ queryKey: ["headerSuggestion", sourceId] });
      void qc.invalidateQueries({ queryKey: ["emptyRows", sourceId] });
      void qc.invalidateQueries({ queryKey: ["athleteFwdFill", sourceId] });
    },
  });

  const headerMergeFillQuery = useQuery({
    queryKey: [
      "headerMergeFill",
      sourceId,
      activeSheet,
      headerRows.join(","),
    ],
    queryFn: () =>
      api.headerMergeFillSuggestions(sourceId, activeSheet!, headerRows),
    enabled: Boolean(activeSheet && sheetMapping),
    retry: false,
    staleTime: 0,
  });

  const suggestionByCell = useMemo<Record<string, CellSuggestion>>(() => {
    const map: Record<string, CellSuggestion> = {};
    // Сначала кладём merge-fill: ячейки строго в зоне шапки, чтобы они
    // не перекрывались athlete-LOCF (последний всегда в data-зоне).
    for (const s of headerMergeFillQuery.data?.suggestions ?? []) {
      map[`${s.row}:${s.col}`] = {
        proposed: s.proposed,
        hint: s.message_ru,
        kind: "header-merge-fill",
      };
    }
    for (const s of athleteSuggestionsQuery.data?.suggestions ?? []) {
      map[`${s.row}:${s.col}`] = {
        proposed: s.proposed,
        hint: s.message_ru,
        kind: "athlete-locf",
      };
    }
    return map;
  }, [athleteSuggestionsQuery.data, headerMergeFillQuery.data]);

  const cleanupMut = useMutation({
    mutationFn: () =>
      api.removeEmptyRows(sourceId, activeSheet!, headerRows),
    onSuccess: (res) => {
      void qc.invalidateQueries({ queryKey: ["grid", sourceId] });
      void qc.invalidateQueries({ queryKey: ["emptyRows", sourceId] });
      void qc.invalidateQueries({ queryKey: ["athleteFwdFill", sourceId] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sources"] });
      alert(`Удалено пустых строк: ${res.deleted}`);
    },
    onError: (err) =>
      alert(
        err instanceof ApiError
          ? `Не удалось очистить лист: ${err.message}`
          : "Не удалось очистить лист."
      ),
  });

  const applyHeaderMergeMut = useMutation({
    mutationFn: async () => {
      const items = headerMergeFillQuery.data?.suggestions ?? [];
      if (items.length === 0) return 0;
      const edits = items.map((s) => ({
        row: s.row,
        col: s.col,
        value: s.proposed,
      }));
      await gridRef.current?.applyEdits(edits);
      return edits.length;
    },
    onSuccess: (count) => {
      void qc.invalidateQueries({ queryKey: ["headerMergeFill", sourceId] });
      void qc.invalidateQueries({ queryKey: ["grid", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sheetColumns", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sheetPreview", sourceId] });
      void qc.invalidateQueries({ queryKey: ["headerSuggestion", sourceId] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
      if (count) alert(`Применено предложений по шапке: ${count}.`);
    },
    onError: (err) =>
      alert(
        err instanceof ApiError
          ? `Не удалось применить предложения по шапке: ${err.message}`
          : "Не удалось применить предложения по шапке."
      ),
  });

  const onApplyHeaderMergeSuggestions = () => {
    const dirty = gridRef.current?.dirtyCount() ?? 0;
    if (dirty > 0) {
      const ok = window.confirm(
        `В сетке есть ${dirty} несохранённых ручных правок. ` +
          "Применить предложения по шапке сейчас? В файл будут записаны только " +
          "значения merged-ячеек шапки, ручные правки останутся в локальном черновике."
      );
      if (!ok) return;
    }
    applyHeaderMergeMut.mutate();
  };

  const applyAthleteMut = useMutation({
    mutationFn: async () => {
      const items = athleteSuggestionsQuery.data?.suggestions ?? [];
      if (items.length === 0) return 0;
      const edits = items.map((s) => ({
        row: s.row,
        col: s.col,
        value: s.proposed,
      }));
      await gridRef.current?.applyEdits(edits);
      return edits.length;
    },
    onSuccess: (count) => {
      void qc.invalidateQueries({ queryKey: ["athleteFwdFill", sourceId] });
      void qc.invalidateQueries({ queryKey: ["grid", sourceId] });
      void qc.invalidateQueries({ queryKey: ["emptyRows", sourceId] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
      if (count) alert(`Применено предложений ФИО: ${count}.`);
    },
    onError: (err) =>
      alert(
        err instanceof ApiError
          ? `Не удалось применить предложения: ${err.message}`
          : "Не удалось применить предложения."
      ),
  });

  const onApplyAthleteSuggestions = () => {
    const dirty = gridRef.current?.dirtyCount() ?? 0;
    if (dirty > 0) {
      const ok = window.confirm(
        `В сетке есть ${dirty} несохранённых ручных правок. ` +
          "Применить предложения ФИО сейчас? В файл будут записаны только предложения, " +
          "ручные правки останутся в локальном черновике."
      );
      if (!ok) return;
    }
    applyAthleteMut.mutate();
  };

  if (!mapping || sheets.length === 0) {
    return (
      <Card className="px-6 py-8 text-center text-brand-700/70">
        Сначала выберите листы и подождите, пока preflight предложит начальное
        сопоставление.
      </Card>
    );
  }

  // Только при последнем успешном ответе: иначе при сетевой ошибке RQ
  // оставляет прежний { count } и isSuccess=false, и нельзя показывать
  // устаревшее «обнаружено N пустых».
  const emptyCount = emptyCountQuery.isSuccess
    ? (emptyCountQuery.data?.count ?? 0)
    : 0;
  const athleteData = athleteSuggestionsQuery.data;
  const athleteCount = athleteData?.suggestions.length ?? 0;

  return (
    <div className="space-y-6">
      <BatchSuggestionsPanel
        sourceId={sourceId}
        mapping={mapping}
        activeSheet={activeSheet}
        setActiveSheet={setActiveSheet}
      />

      <Section
        title="Подготовка листа"
        description="Сверху — данные листа (правки сохраняются в исходный файл). Снизу — назначение ролей колонкам этого же листа. Заголовочные строки выделены и недоступны для редактирования."
      >
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {sheets.map((name) => (
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

        <div className="rounded-md border border-brand-100 bg-white px-4 py-3 mb-4 space-y-2 text-sm text-brand-800">
          <div className="font-semibold text-brand-900">
            Что обычно стоит проверить руками
          </div>
          <ul className="list-disc pl-5 space-y-1 text-brand-700/90">
            <li>
              Полностью пустые строки между данными — они мешают подсчётам и
              разметке (для них есть кнопка ниже).
            </li>
            <li>
              Дублирующиеся «шапки» внутри данных (когда заголовок повторён
              посреди таблицы) — удалите такие строки вручную или через
              редактор сетки.
            </li>
            <li>
              «Сводные» строки итогов в данных (всё число — итог, а не запись
              эпизода) — их тоже стоит убрать перед анализом.
            </li>
          </ul>
        </div>

        <div className="flex flex-wrap items-center gap-2 mb-3">
          {emptyCountQuery.isLoading ? (
            <span className="text-xs text-brand-700/60">
              Проверяем пустые строки… на больших листах это может занять
              несколько секунд.
            </span>
          ) : emptyCountQuery.isError ? (
            <div className="flex flex-wrap items-center gap-2 text-xs text-red-800">
              <span>
                Не удалось проверить пустые строки:{" "}
                {emptyCountQuery.error instanceof Error
                  ? emptyCountQuery.error.message
                  : "ошибка сети"}
                .
              </span>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => void emptyCountQuery.refetch()}
                disabled={emptyCountQuery.isFetching}
              >
                {emptyCountQuery.isFetching ? "Запрос…" : "Повторить"}
              </Button>
            </div>
          ) : emptyCount > 0 ? (
            <div className="flex flex-wrap items-center gap-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
              <span>
                Обнаружено{" "}
                <b>
                  {emptyCount} полностью пуст
                  {pluralEnding(emptyCount, "ая", "ые", "ых")} строк
                  {pluralEnding(emptyCount, "а", "и", "")}
                </b>{" "}
                под заголовком — рекомендуем удалить.
              </span>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => activeSheet && cleanupMut.mutate()}
                disabled={!activeSheet || cleanupMut.isPending}
              >
                {cleanupMut.isPending
                  ? "Удаляем..."
                  : `Удалить ${emptyCount} строк`}
              </Button>
            </div>
          ) : (
            <span className="text-xs text-emerald-700">
              Полностью пустых строк под заголовком не найдено.
            </span>
          )}

          <span className="ml-auto" />

          <Button
            size="sm"
            variant="ghost"
            onClick={() => activeSheet && cleanupMut.mutate()}
            disabled={!activeSheet || cleanupMut.isPending}
            title="Удалить data-строки, в которых все ячейки пустые. Заголовочные строки не трогаются."
          >
            {cleanupMut.isPending
              ? "Чистим..."
              : "Удалить полностью пустые строки"}
          </Button>
        </div>

        <AthleteForwardFillBanner
          hasAthleteRole={hasAthleteRole}
          hasEpisodeColumn={hasEpisodeColumn}
          isLoading={athleteSuggestionsQuery.isLoading}
          isError={athleteSuggestionsQuery.isError}
          warning={athleteData?.warning ?? null}
          athleteColumn={athleteData?.athlete_column ?? null}
          count={athleteCount}
          isApplying={applyAthleteMut.isPending}
          onApply={onApplyAthleteSuggestions}
        />

        {activeSheet && sheetMapping && (
          <>
            <HeaderRowsPrepareBanner
              isLoading={headerSuggestionQuery.isLoading}
              isError={headerSuggestionQuery.isError}
              data={headerSuggestionQuery.data}
              isApplying={applyHeaderRowsMut.isPending}
              onApply={(rows) => applyHeaderRowsMut.mutate(rows)}
            />
            <HeaderMergeFillBanner
              isLoading={headerMergeFillQuery.isLoading}
              isError={headerMergeFillQuery.isError}
              data={headerMergeFillQuery.data}
              isApplying={applyHeaderMergeMut.isPending}
              onApply={onApplyHeaderMergeSuggestions}
            />
            <SheetGrid
              ref={gridRef}
              sourceId={sourceId}
              sheetName={activeSheet}
              headerRows={sheetMapping.header_rows}
              suggestionByCell={suggestionByCell}
              onDirtyChange={setGridDirty}
            />
          </>
        )}
        {gridDirty > 0 && (
          <div className="mt-2 text-xs text-brand-700/60">
            В сетке есть несохранённые ручные правки ({gridDirty}). Не забудьте
            нажать «Сохранить изменения», иначе они не попадут в файл.
          </div>
        )}
      </Section>

      <Section
        title="Сопоставление колонок"
        description="Назначьте роли колонкам активного листа. Изменения header_rows ниже синхронизируются с сеткой данных выше."
      >
        <Suspense
          fallback={
            <Card className="px-6 py-8 text-center text-brand-700/70">
              Загрузка редактора mapping...
            </Card>
          }
        >
          <MappingEditor
            sourceId={sourceId}
            controlledActiveSheet={activeSheet}
            onActiveSheetChange={setActiveSheet}
            hideSheetTabs
            hideHeader
          />
        </Suspense>
      </Section>
    </div>
  );
};

// ---------------------------------------------------------------------------
// BatchSuggestionsPanel
// ---------------------------------------------------------------------------

type BatchSheetStatus = {
  sheet: string;
  headerRows: number[];
  emptyCount: number | null;
  athleteCount: number | null;
  hasAthleteRole: boolean;
  headerRowsMismatch: boolean;
  suggestedHeaderRows: number[];
  mergeFillCount: number | null;
  athleteEdits: CellEdit[];
  mergeEdits: CellEdit[];
  loading: boolean;
};

const BatchSuggestionsPanel: React.FC<{
  sourceId: number;
  mapping: ColumnMappingConfig | null;
  activeSheet: string | null;
  setActiveSheet: (s: string) => void;
}> = ({ sourceId, mapping, activeSheet, setActiveSheet }) => {
  const qc = useQueryClient();
  const sheets = useMemo(
    () => (mapping ? Object.keys(mapping.sheets) : []),
    [mapping]
  );
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [applying, setApplying] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Parallel queries: one slot = [emptyRows, athleteFwd, headerSuggestion, mergeFill]
  const emptyRowsQueries = useQueries({
    queries: sheets.map((sheet) => {
      const hrs = mapping?.sheets[sheet]?.header_rows ?? [0];
      return {
        queryKey: ["emptyRows", sourceId, sheet, hrs.join(",")],
        queryFn: () => api.countEmptyRows(sourceId, sheet, hrs),
        enabled: Boolean(mapping),
        staleTime: 0,
        retry: 0,
      };
    }),
  });

  const athleteQueries = useQueries({
    queries: sheets.map((sheet) => {
      const sm = mapping?.sheets[sheet];
      const hrs = sm?.header_rows ?? [0];
      const hasAthlete = (sm?.roles?.athlete?.length ?? 0) > 0;
      return {
        queryKey: [
          "athleteFwdFill",
          sourceId,
          sheet,
          hrs.join(","),
          (sm?.roles?.athlete ?? []).join("|"),
        ],
        queryFn: () => api.athleteForwardFillSuggestions(sourceId, sheet, hrs),
        enabled: Boolean(mapping && hasAthlete),
        staleTime: 0,
        retry: 0,
      };
    }),
  });

  const headerSuggestionQueries = useQueries({
    queries: sheets.map((sheet) => {
      const hrs = mapping?.sheets[sheet]?.header_rows ?? [0];
      return {
        queryKey: ["headerSuggestion", sourceId, sheet, hrs.join(",")],
        queryFn: () => api.headerRowsSuggestion(sourceId, sheet, hrs),
        enabled: Boolean(mapping),
        staleTime: 30_000,
        retry: 0,
      };
    }),
  });

  const mergeFillQueries = useQueries({
    queries: sheets.map((sheet) => {
      const hrs = mapping?.sheets[sheet]?.header_rows ?? [0];
      return {
        queryKey: ["headerMergeFill", sourceId, sheet, hrs.join(",")],
        queryFn: () => api.headerMergeFillSuggestions(sourceId, sheet, hrs),
        enabled: Boolean(mapping),
        staleTime: 0,
        retry: 0,
      };
    }),
  });

  const statuses: BatchSheetStatus[] = sheets.map((sheet, i) => {
    const sm = mapping?.sheets[sheet];
    const hrs = sm?.header_rows ?? [0];
    const hasAthlete = (sm?.roles?.athlete?.length ?? 0) > 0;
    const emptyQ = emptyRowsQueries[i];
    const athleteQ = athleteQueries[i];
    const headerQ = headerSuggestionQueries[i];
    const mergeQ = mergeFillQueries[i];
    const loading =
      emptyQ.isLoading || athleteQ.isLoading || headerQ.isLoading || mergeQ.isLoading;
    const athleteSuggestions = athleteQ.data?.suggestions ?? [];
    const mergeSuggestions = mergeQ.data?.suggestions ?? [];
    return {
      sheet,
      headerRows: hrs,
      emptyCount: emptyQ.isSuccess ? (emptyQ.data?.count ?? 0) : null,
      athleteCount: athleteQ.isSuccess ? athleteSuggestions.length : null,
      hasAthleteRole: hasAthlete,
      headerRowsMismatch: headerQ.isSuccess
        ? !(headerQ.data?.matches_current ?? true)
        : false,
      suggestedHeaderRows: headerQ.data?.suggested_header_rows ?? [],
      mergeFillCount: mergeQ.isSuccess ? mergeSuggestions.length : null,
      athleteEdits: athleteSuggestions.map((s) => ({
        row: s.row,
        col: s.col,
        value: s.proposed as CellEdit["value"],
      })),
      mergeEdits: mergeSuggestions.map((s) => ({
        row: s.row,
        col: s.col,
        value: s.proposed as CellEdit["value"],
      })),
      loading,
    };
  });

  const hasAnyIssue = statuses.some(
    (s) =>
      (s.emptyCount ?? 0) > 0 ||
      (s.athleteCount ?? 0) > 0 ||
      s.headerRowsMismatch ||
      (s.mergeFillCount ?? 0) > 0
  );

  const allLoading = statuses.some((s) => s.loading);

  const toggleSheet = (sheet: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(sheet)) next.delete(sheet);
      else next.add(sheet);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === sheets.length) setSelected(new Set());
    else setSelected(new Set(sheets));
  };

  const invalidateAll = () => {
    void qc.invalidateQueries({ queryKey: ["emptyRows", sourceId] });
    void qc.invalidateQueries({ queryKey: ["athleteFwdFill", sourceId] });
    void qc.invalidateQueries({ queryKey: ["headerMergeFill", sourceId] });
    void qc.invalidateQueries({ queryKey: ["headerSuggestion", sourceId] });
    void qc.invalidateQueries({ queryKey: ["grid", sourceId] });
    void qc.invalidateQueries({ queryKey: ["mapping", sourceId] });
    void qc.invalidateQueries({ queryKey: ["sheetColumns", sourceId] });
    void qc.invalidateQueries({ queryKey: ["source", sourceId] });
  };

  const applyEmptyRows = async () => {
    const targets = statuses.filter(
      (s) => selected.has(s.sheet) && (s.emptyCount ?? 0) > 0
    );
    if (!targets.length) return;
    setApplying("empty");
    const errs: Record<string, string> = {};
    for (const s of targets) {
      try {
        await api.removeEmptyRows(sourceId, s.sheet, s.headerRows);
      } catch (e) {
        errs[s.sheet] = e instanceof Error ? e.message : "ошибка";
      }
    }
    setErrors(errs);
    setApplying(null);
    invalidateAll();
  };

  const applyAthlete = async () => {
    const targets = statuses.filter(
      (s) =>
        selected.has(s.sheet) && s.hasAthleteRole && (s.athleteCount ?? 0) > 0
    );
    if (!targets.length) return;
    setApplying("athlete");
    const errs: Record<string, string> = {};
    for (const s of targets) {
      try {
        if (s.athleteEdits.length > 0) {
          await api.applySheetGridEdits(sourceId, s.sheet, s.athleteEdits);
        }
      } catch (e) {
        errs[s.sheet] = e instanceof Error ? e.message : "ошибка";
      }
    }
    setErrors(errs);
    setApplying(null);
    invalidateAll();
  };

  const applyMergeFill = async () => {
    const targets = statuses.filter(
      (s) => selected.has(s.sheet) && (s.mergeFillCount ?? 0) > 0
    );
    if (!targets.length) return;
    setApplying("merge");
    const errs: Record<string, string> = {};
    for (const s of targets) {
      try {
        if (s.mergeEdits.length > 0) {
          await api.applySheetGridEdits(sourceId, s.sheet, s.mergeEdits);
        }
      } catch (e) {
        errs[s.sheet] = e instanceof Error ? e.message : "ошибка";
      }
    }
    setErrors(errs);
    setApplying(null);
    invalidateAll();
  };

  const applyHeaderRows = async () => {
    const targets = statuses.filter(
      (s) => selected.has(s.sheet) && s.headerRowsMismatch && s.suggestedHeaderRows.length > 0
    );
    if (!targets.length || !mapping) return;
    setApplying("headerRows");
    const next: ColumnMappingConfig = JSON.parse(JSON.stringify(mapping));
    const errs: Record<string, string> = {};
    for (const s of targets) {
      try {
        next.sheets[s.sheet] = {
          ...next.sheets[s.sheet],
          header_rows: [...s.suggestedHeaderRows],
        };
      } catch (e) {
        errs[s.sheet] = e instanceof Error ? e.message : "ошибка";
      }
    }
    try {
      await api.putMapping(sourceId, next);
    } catch (e) {
      for (const s of targets) {
        errs[s.sheet] = e instanceof Error ? e.message : "ошибка";
      }
    }
    setErrors(errs);
    setApplying(null);
    invalidateAll();
  };

  const selectedStatuses = statuses.filter((s) => selected.has(s.sheet));
  const canApplyEmpty = selectedStatuses.some((s) => (s.emptyCount ?? 0) > 0);
  const canApplyAthlete = selectedStatuses.some(
    (s) => s.hasAthleteRole && (s.athleteCount ?? 0) > 0
  );
  const canApplyMerge = selectedStatuses.some((s) => (s.mergeFillCount ?? 0) > 0);
  const canApplyHeaderRows = selectedStatuses.some((s) => s.headerRowsMismatch);

  if (sheets.length <= 1) return null;

  return (
    <Section
      title="Сводка по всем листам"
      description="Рекомендации для каждого листа. Выберите листы и применяйте действия сразу к нескольким."
    >
      {Object.entries(errors).length > 0 && (
        <div className="mb-3 space-y-1">
          {Object.entries(errors).map(([sheet, msg]) => (
            <div
              key={sheet}
              className="rounded bg-red-50 border border-red-200 px-3 py-1 text-xs text-red-800"
            >
              {sheet}: {msg}
            </div>
          ))}
        </div>
      )}

      <div className="overflow-x-auto rounded-md border border-brand-100 bg-white">
        <table className="min-w-full text-sm">
          <thead className="bg-brand-50 text-xs text-brand-700 uppercase">
            <tr>
              <th className="px-3 py-2 text-left w-8">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={selected.size === sheets.length && sheets.length > 0}
                  onChange={toggleAll}
                  title="Выбрать все"
                />
              </th>
              <th className="px-3 py-2 text-left">Лист</th>
              <th className="px-3 py-2 text-center" title="Пустые строки">Пустые стр.</th>
              <th className="px-3 py-2 text-center" title="Пропущенные ФИО">ФИО</th>
              <th className="px-3 py-2 text-center" title="Строки заголовка">Шапка</th>
              <th className="px-3 py-2 text-center" title="Merged-ячейки в шапке">Merged</th>
            </tr>
          </thead>
          <tbody>
            {statuses.map((s, idx) => (
              <tr
                key={s.sheet}
                className={`cursor-pointer ${
                  idx % 2 === 0 ? "bg-white" : "bg-brand-50/40"
                } ${s.sheet === activeSheet ? "ring-1 ring-inset ring-brand-400" : ""}`}
              >
                <td className="px-3 py-2">
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={selected.has(s.sheet)}
                    onChange={() => toggleSheet(s.sheet)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </td>
                <td
                  className="px-3 py-2 font-medium text-brand-900 hover:text-brand-600"
                  onClick={() => setActiveSheet(s.sheet)}
                >
                  {s.sheet}
                </td>
                <td className="px-3 py-2 text-center">
                  {s.loading ? (
                    <span className="text-brand-300">…</span>
                  ) : s.emptyCount === null ? (
                    <span className="text-brand-300">—</span>
                  ) : s.emptyCount > 0 ? (
                    <span className="inline-flex items-center justify-center rounded-full bg-amber-100 text-amber-800 text-xs px-2 py-0.5 font-medium">
                      {s.emptyCount}
                    </span>
                  ) : (
                    <span className="text-emerald-600 text-xs">✓</span>
                  )}
                </td>
                <td className="px-3 py-2 text-center">
                  {!s.hasAthleteRole ? (
                    <span className="text-brand-300 text-xs">нет роли</span>
                  ) : s.loading ? (
                    <span className="text-brand-300">…</span>
                  ) : s.athleteCount === null ? (
                    <span className="text-brand-300">—</span>
                  ) : s.athleteCount > 0 ? (
                    <span className="inline-flex items-center justify-center rounded-full bg-violet-100 text-violet-800 text-xs px-2 py-0.5 font-medium">
                      {s.athleteCount}
                    </span>
                  ) : (
                    <span className="text-emerald-600 text-xs">✓</span>
                  )}
                </td>
                <td className="px-3 py-2 text-center">
                  {s.loading ? (
                    <span className="text-brand-300">…</span>
                  ) : s.headerRowsMismatch ? (
                    <span
                      className="inline-flex items-center justify-center rounded-full bg-amber-100 text-amber-800 text-xs px-2 py-0.5 font-medium"
                      title={`Рекомендованы: [${s.suggestedHeaderRows.join(", ")}]`}
                    >
                      !
                    </span>
                  ) : (
                    <span className="text-emerald-600 text-xs">✓</span>
                  )}
                </td>
                <td className="px-3 py-2 text-center">
                  {s.loading ? (
                    <span className="text-brand-300">…</span>
                  ) : s.mergeFillCount === null ? (
                    <span className="text-brand-300">—</span>
                  ) : s.mergeFillCount > 0 ? (
                    <span className="inline-flex items-center justify-center rounded-full bg-amber-100 text-amber-800 text-xs px-2 py-0.5 font-medium">
                      {s.mergeFillCount}
                    </span>
                  ) : (
                    <span className="text-emerald-600 text-xs">✓</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {(selected.size > 0 || hasAnyIssue) && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {selected.size > 0 && (
            <span className="text-xs text-brand-700/70">
              Выбрано листов: {selected.size}
            </span>
          )}
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void applyEmptyRows()}
            disabled={!canApplyEmpty || applying !== null || allLoading}
            title="Удалить пустые строки на выбранных листах"
          >
            {applying === "empty" ? "Удаляем…" : "Удалить пустые строки"}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void applyAthlete()}
            disabled={!canApplyAthlete || applying !== null || allLoading}
            title="Заполнить пропущенные ФИО вниз на выбранных листах"
          >
            {applying === "athlete" ? "Применяем…" : "Заполнить ФИО"}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void applyMergeFill()}
            disabled={!canApplyMerge || applying !== null || allLoading}
            title="Материализовать merged-ячейки в шапке на выбранных листах"
          >
            {applying === "merge" ? "Применяем…" : "Merged-шапка"}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void applyHeaderRows()}
            disabled={!canApplyHeaderRows || applying !== null || allLoading}
            title="Применить рекомендованные строки заголовка на выбранных листах"
          >
            {applying === "headerRows" ? "Сохраняем…" : "Применить строки шапки"}
          </Button>
        </div>
      )}
    </Section>
  );
};

const HeaderRowsPrepareBanner: React.FC<{
  isLoading: boolean;
  isError: boolean;
  data: HeaderRowsSuggestionResponse | undefined;
  isApplying: boolean;
  onApply: (rows: number[]) => void;
}> = ({ isLoading, isError, data, isApplying, onApply }) => {
  if (isLoading) {
    return (
      <div className="mb-4 text-xs text-brand-700/60">
        Проверяем многоуровневую шапку (какие строки объединить в заголовок)…
      </div>
    );
  }
  if (isError) {
    return (
      <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
        Не удалось запросить подсказку по строкам шапки. Укажите{" "}
        <b>строки заголовка</b> вручную в разделе «Сопоставление колонок» ниже
        (нумерация с 0).
      </div>
    );
  }
  if (!data) return null;
  const { suggested_header_rows: suggested, matches_current: matches } = data;
  if (matches) {
    return (
      <div className="mb-4 rounded-md border border-teal-200 bg-teal-50/90 px-3 py-2 text-sm text-teal-950">
        <div className="font-semibold">Многострочная шапка</div>
        <p className="mt-1 text-xs text-teal-900/90">
          Для читаемых имён колонок (по смыслу — как подсказки по ФИО выше) из
          файла собирается заголовок из строк{" "}
          <code className="font-mono text-[11px]">[{suggested.join(", ")}]</code>{" "}
          (нумерация с 0). Сейчас эта настройка <b>совпадает</b> с эвристикой.
          Развёрнутое превью flatten-имён — внизу, под полем «Строки заголовка».
        </p>
      </div>
    );
  }
  return (
    <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-950">
      <div className="font-semibold">Многострочная шапка</div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-amber-900/95">
        <span>
          Рекомендованы строки:{" "}
          <code className="font-mono">[{suggested.join(", ")}]</code>. В mapping
          сейчас:{" "}
          <code className="font-mono">
            [{(data.current_header_rows ?? []).join(", ")}]
          </code>
          .
        </span>
        <Button
          size="sm"
          variant="secondary"
          onClick={() => onApply(suggested)}
          disabled={isApplying}
        >
          {isApplying ? "Сохраняем…" : "Применить в mapping"}
        </Button>
      </div>
    </div>
  );
};

const HeaderMergeFillBanner: React.FC<{
  isLoading: boolean;
  isError: boolean;
  data: HeaderMergeFillResponse | undefined;
  isApplying: boolean;
  onApply: () => void;
}> = ({ isLoading, isError, data, isApplying, onApply }) => {
  if (isLoading) {
    return (
      <div className="mb-4 text-xs text-brand-700/60">
        Проверяем merged-ячейки шапки…
      </div>
    );
  }
  if (isError) {
    return (
      <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
        Не удалось получить предложения по merged-шапке.
      </div>
    );
  }
  if (!data) return null;
  const count = data.suggestions.length;
  if (count === 0) {
    if (data.warning) {
      return (
        <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          {data.warning}
        </div>
      );
    }
    return (
      <div className="mb-4 text-xs text-emerald-700">
        Незаполненных ячеек в merged-шапке не найдено.
      </div>
    );
  }
  return (
    <div className="mb-4 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-950">
      <div className="flex flex-wrap items-center gap-3">
        <span>
          В шапке есть <b>{count}</b> объединённых ячеек, у которых «дочерние»
          поля пусты — после ручного редактирования сетки flatten-имена колонок
          могут потерять верхний уровень. Можем материализовать значение
          родительской ячейки во все объединённые позиции.
        </span>
        <Button
          size="sm"
          variant="secondary"
          onClick={onApply}
          disabled={isApplying}
          title="Скопировать значение master-ячейки merged-диапазона во все его подчинённые ячейки. Запишется в файл одним пакетом."
        >
          {isApplying ? "Применяем..." : `Применить ${count} предложений`}
        </Button>
      </div>
      <div className="mt-1 text-xs text-amber-900/80">
        Затрагиваются только ячейки внутри объединённых диапазонов, целиком
        лежащих в строках заголовка. Содержимое самих master-ячеек не меняется.
      </div>
    </div>
  );
};

const AthleteForwardFillBanner: React.FC<{
  hasAthleteRole: boolean;
  hasEpisodeColumn: boolean;
  isLoading: boolean;
  isError: boolean;
  warning: string | null;
  athleteColumn: string | null;
  count: number;
  isApplying: boolean;
  onApply: () => void;
}> = ({
  hasAthleteRole,
  hasEpisodeColumn,
  isLoading,
  isError,
  warning,
  athleteColumn,
  count,
  isApplying,
  onApply,
}) => {
  if (!hasAthleteRole) {
    return (
      <div className="mb-4 rounded-md border border-brand-200 bg-white px-3 py-2 text-xs text-brand-700/70">
        Чтобы получить предложения по заполнению ФИО, назначьте роль{" "}
        <b>«спортсмен»</b> хотя бы одной колонке листа в блоке «Сопоставление
        колонок» ниже.
      </div>
    );
  }
  if (isLoading) {
    return (
      <div className="mb-4 text-xs text-brand-700/60">
        Ищем пропущенные ФИО...
      </div>
    );
  }
  if (isError) {
    return (
      <div className="mb-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
        Не удалось получить предложения по ФИО. Попробуйте обновить страницу.
      </div>
    );
  }
  if (count === 0) {
    return (
      <div className="mb-4 text-xs text-emerald-700">
        Пропущенных ФИО не найдено
        {athleteColumn ? ` (по колонке «${athleteColumn}»)` : ""}.
        {warning && (
          <span className="ml-2 text-amber-700/90">{warning}</span>
        )}
      </div>
    );
  }
  return (
    <div className="mb-4 rounded-md border border-violet-200 bg-violet-50 px-3 py-2 text-sm text-violet-900">
      <div className="flex flex-wrap items-center gap-3">
        <span>
          Обнаружено{" "}
          <b>
            {count} пропущенн{pluralEnding(count, "ое", "ых", "ых")} ФИО
          </b>
          {athleteColumn ? ` в колонке «${athleteColumn}»` : ""}. Ниже в сетке
          такие ячейки подсвечены — наведите курсор, чтобы увидеть, какое
          значение будет проставлено.
        </span>
        <Button
          size="sm"
          variant="secondary"
          onClick={onApply}
          disabled={isApplying}
          title="Скопировать последнее непустое ФИО выше во все подсвеченные ячейки. Запишется в файл одним пакетом, без отдельной кнопки «Сохранить»."
        >
          {isApplying ? "Применяем..." : `Применить ${count} предложений`}
        </Button>
      </div>
      <div className="mt-1 text-xs text-violet-800/80">
        Это <b>копирование вниз</b> ближайшего ФИО сверху, а не «угадывание»
        спортсмена по эпизоду. Если в файле строки перемешаны — отмените
        применение и поправьте ФИО вручную.
        {hasEpisodeColumn && (
          <>
            {" "}
            Подсветка и подстановка — только для строк, где в колонке
            с ролью «эпизод» есть значение, и не для полностью пустых строк.
          </>
        )}
      </div>
      {warning && (
        <div className="mt-1 text-xs text-amber-700/90">{warning}</div>
      )}
    </div>
  );
};

const FinalizeSummary: React.FC<{
  mapping: ColumnMappingConfig | null;
  sheetNames: string[];
}> = ({ mapping, sheetNames }) => {
  if (!mapping) {
    return (
      <div className="text-sm text-red-700">
        Mapping не сохранён — вернитесь на шаг «Данные и колонки».
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="text-sm text-brand-900">
        Листов в анализе:{" "}
        <b>{Object.keys(mapping.sheets).length}</b>
        {sheetNames.length !== Object.keys(mapping.sheets).length && (
          <span className="ml-2 text-amber-700">
            (выбор листов и mapping расходятся — повторите preflight)
          </span>
        )}
      </div>
      <div className="space-y-2">
        {Object.entries(mapping.sheets).map(([name, m]) => {
          const total = Object.values(m.roles).reduce(
            (acc, cols) => acc + (cols?.length ?? 0),
            0
          );
          return (
            <div
              key={name}
              className="rounded-md border border-brand-100 bg-white px-3 py-2 text-sm"
            >
              <div className="font-medium text-brand-900">{name}</div>
              <div className="text-xs text-brand-700/70">
                header_rows: [{m.header_rows.join(", ")}], колонок с ролями:{" "}
                {total}
              </div>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {Object.entries(m.roles).map(([role, cols]) => (
                  <Badge key={role} tone="info">
                    {role}: {(cols ?? []).length}
                  </Badge>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

function pluralEnding(n: number, one: string, few: string, many: string): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}
