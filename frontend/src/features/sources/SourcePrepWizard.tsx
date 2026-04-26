import React, { Suspense, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type {
  ColumnMappingConfig,
  SheetMapping,
  SourceSummary,
} from "@/api/types";
import { Badge, Button, Card, Section } from "@/components/ui";
import { SheetGrid } from "./SheetGrid";

const MappingEditor = React.lazy(() =>
  import("@/features/analysis/MappingEditor").then((m) => ({
    default: m.MappingEditor,
  }))
);

type Step = "sheets" | "mapping" | "edit" | "finalize";

const STEPS: { id: Step; label: string; description: string }[] = [
  {
    id: "sheets",
    label: "Листы",
    description: "Выберите листы Excel, которые должны участвовать в анализе.",
  },
  {
    id: "mapping",
    label: "Сопоставление колонок",
    description:
      "Назначьте роли колонкам выбранных листов. Preflight предлагает стартовое сопоставление.",
  },
  {
    id: "edit",
    label: "Редактирование данных",
    description:
      "Удаление пустых строк и точечная правка ячеек. Все правки сохраняются в исходный файл.",
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
      // Запускаем preflight только если у нас ещё нет mapping для выбранных листов.
      const knownSheets = mapping ? Object.keys(mapping.sheets) : [];
      const sameSet =
        knownSheets.length === includedList.length &&
        includedList.every((s) => knownSheets.includes(s));
      if (!sameSet) {
        await preflightMut.mutateAsync(includedList).catch(() => undefined);
      }
      setStep("mapping");
      return;
    }
    if (step === "mapping") {
      if (!mapping || Object.keys(mapping.sheets).length === 0) {
        setError("Сохраните column mapping перед переходом дальше.");
        return;
      }
      setStep("edit");
      return;
    }
    if (step === "edit") {
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

        {step === "mapping" && (
          <Suspense
            fallback={
              <Card className="px-6 py-8 text-center text-brand-700/70">
                Загрузка редактора mapping...
              </Card>
            }
          >
            <MappingEditor sourceId={sourceId} />
          </Suspense>
        )}

        {step === "edit" && (
          <SheetEditingStep sourceId={sourceId} mapping={mapping} />
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

const SheetEditingStep: React.FC<{
  sourceId: number;
  mapping: ColumnMappingConfig | null;
}> = ({ sourceId, mapping }) => {
  const qc = useQueryClient();
  const sheets = mapping ? Object.keys(mapping.sheets) : [];
  const [active, setActive] = useState<string | null>(sheets[0] ?? null);

  useEffect(() => {
    if (!active && sheets.length > 0) setActive(sheets[0]);
  }, [active, sheets]);

  const sheetMapping: SheetMapping | null =
    active && mapping ? mapping.sheets[active] : null;

  const cleanupMut = useMutation({
    mutationFn: () =>
      api.removeEmptyRows(sourceId, active!, sheetMapping?.header_rows ?? [0]),
    onSuccess: (res) => {
      void qc.invalidateQueries({ queryKey: ["grid", sourceId] });
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

  if (!mapping || sheets.length === 0) {
    return (
      <Card className="px-6 py-8 text-center text-brand-700/70">
        Сначала выполните column mapping.
      </Card>
    );
  }

  return (
    <Section
      title="Редактирование листа"
      description="Изменения сохраняются в файл источника. Заголовочные строки выделены и недоступны для редактирования."
    >
      <div className="flex flex-wrap items-center gap-2 mb-3">
        {sheets.map((name) => (
          <button
            key={name}
            onClick={() => setActive(name)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium border ${
              name === active
                ? "bg-brand-100 text-brand-900 border-brand-300"
                : "bg-white text-brand-800 border-brand-200 hover:bg-brand-50"
            }`}
          >
            {name}
          </button>
        ))}
        <span className="ml-auto" />
        <Button
          variant="secondary"
          onClick={() => active && cleanupMut.mutate()}
          disabled={!active || cleanupMut.isPending}
        >
          {cleanupMut.isPending
            ? "Чистим..."
            : "Удалить полностью пустые строки"}
        </Button>
      </div>
      {active && sheetMapping && (
        <SheetGrid
          sourceId={sourceId}
          sheetName={active}
          headerRows={sheetMapping.header_rows}
        />
      )}
    </Section>
  );
};

const FinalizeSummary: React.FC<{
  mapping: ColumnMappingConfig | null;
  sheetNames: string[];
}> = ({ mapping, sheetNames }) => {
  if (!mapping) {
    return (
      <div className="text-sm text-red-700">
        Mapping не сохранён — вернитесь на шаг «Сопоставление колонок».
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
