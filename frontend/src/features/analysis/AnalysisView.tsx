import React, { Suspense, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type { AnalysisRunFull, HMMMode, SourceSummary } from "@/api/types";
import { Button, Card, Section } from "@/components/ui";
import { StatusBadge } from "./StatusBadge";
import { WarningsList } from "./WarningsList";
import { AuditTable } from "./AuditTable";
import { DetectedColumns } from "./DetectedColumns";
import { ChartsGrid } from "./ChartsGrid";
import { ZapChannelsCard } from "./ZapChannelsCard";

// Code-split: редактор mapping и HMM-вьюха грузятся только когда
// пользователь открывает соответствующие секции интерфейса.
const MappingEditor = React.lazy(() =>
  import("./MappingEditor").then((m) => ({ default: m.MappingEditor }))
);
const HMMView = React.lazy(() =>
  import("./HMMView").then((m) => ({ default: m.HMMView }))
);

const SectionFallback: React.FC<{ label: string }> = ({ label }) => (
  <Card className="px-6 py-8 text-center text-brand-700/70">{label}</Card>
);

type Tab = "result" | "mapping";

export const AnalysisView: React.FC<{ sourceId: number }> = ({ sourceId }) => {
  const qc = useQueryClient();
  const [tab, setTab] = useState<Tab>("result");
  const [hmmMode, setHmmMode] = useState<HMMMode>("auto");

  const sourceQuery = useQuery<SourceSummary>({
    queryKey: ["source", sourceId],
    queryFn: () => api.getSource(sourceId),
  });

  const resultQuery = useQuery<AnalysisRunFull, ApiError>({
    queryKey: ["result", sourceId],
    queryFn: () => api.latestResult(sourceId),
    retry: false,
  });

  const runAnalyze = useMutation({
    mutationFn: () => api.analyze(sourceId, hmmMode),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["result", sourceId] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const source = sourceQuery.data;
  const run = resultQuery.data;
  const result = run?.result;

  return (
    <div className="max-w-6xl w-full mx-auto px-6 py-6 space-y-6">
      <Card className="px-6 py-5 flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="text-xs text-brand-700/70 uppercase tracking-wide">
            Источник
          </div>
          <div className="text-lg font-semibold text-brand-900 truncate">
            {source?.original_filename ?? "..."}
          </div>
          {run && (
            <div className="mt-2 flex items-center gap-2 flex-wrap">
              <StatusBadge status={run.status} />
              <span className="text-xs text-brand-700/60">
                Последний запуск:{" "}
                {new Date(run.created_at).toLocaleString("ru-RU")}
              </span>
              {source?.has_mapping && (
                <span className="text-xs text-emerald-700">
                  (применён column mapping)
                </span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <select
            value={hmmMode}
            onChange={(e) => setHmmMode(e.target.value as HMMMode)}
            className="h-10 rounded-md border border-brand-200 px-2 text-sm"
            title="Режим HMM"
          >
            <option value="auto">HMM: auto (BIC)</option>
            <option value="detailed">HMM: 7-state (detailed)</option>
            <option value="basic">HMM: 3-state (basic)</option>
            <option value="off">HMM: off</option>
          </select>
          <Button
            onClick={() => runAnalyze.mutate()}
            disabled={runAnalyze.isPending}
          >
            {runAnalyze.isPending
              ? "Анализируем..."
              : run
              ? "Перезапустить анализ"
              : "Запустить анализ"}
          </Button>
        </div>
      </Card>

      <div className="flex items-center gap-2 border-b border-brand-200">
        {(
          [
            ["result", "Результат"],
            ["mapping", "Сопоставление колонок"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            onClick={() => setTab(value)}
            className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 ${
              tab === value
                ? "border-brand-500 text-brand-900"
                : "border-transparent text-brand-700/70 hover:text-brand-900"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {runAnalyze.isError && (
        <div className="rounded-md bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-800">
          Не удалось выполнить анализ:{" "}
          {runAnalyze.error instanceof Error
            ? runAnalyze.error.message
            : "ошибка"}
        </div>
      )}

      {tab === "mapping" && (
        <Suspense fallback={<SectionFallback label="Загрузка редактора mapping..." />}>
          <MappingEditor sourceId={sourceId} />
        </Suspense>
      )}

      {tab === "result" && (
        <>
          {resultQuery.isLoading && (
            <Card className="px-6 py-10 text-center text-brand-700/70">
              Загрузка результата...
            </Card>
          )}

          {resultQuery.isError && resultQuery.error.status === 404 && (
            <Card className="px-6 py-10 text-center">
              <div className="text-base font-medium text-brand-900">
                Для этого источника анализ ещё не запускался.
              </div>
              <div className="mt-2 text-sm text-brand-700/70">
                Нажмите «Запустить анализ», чтобы получить honest audit. Если
                структура файла нестандартная, сначала настройте column mapping
                на вкладке «Сопоставление колонок».
              </div>
            </Card>
          )}

          {result && (
            <>
              <Section
                title="Отчёт"
                description="Краткая сводка с честным статусом. Никаких диагностических утверждений без оснований."
              >
                <pre className="whitespace-pre-wrap text-sm text-brand-900/90 font-sans">
                  {result.report}
                </pre>
                {result.basic_statistics.notes.length > 0 && (
                  <ul className="mt-4 list-disc ml-5 text-sm text-brand-900/80 space-y-0.5">
                    {result.basic_statistics.notes.map((n, i) => (
                      <li key={i}>{n}</li>
                    ))}
                  </ul>
                )}
                {Object.keys(result.basic_statistics.hidden_group_totals).length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {Object.entries(result.basic_statistics.hidden_group_totals).map(
                      ([g, total]) => (
                        <span
                          key={g}
                          className="rounded-md bg-brand-50 border border-brand-200 px-2.5 py-1 text-xs text-brand-900"
                        >
                          {g}: <b>{total}</b>
                        </span>
                      )
                    )}
                  </div>
                )}
              </Section>

              {result.hmm && result.status === "hmm_ready" && (
                <Suspense
                  fallback={<SectionFallback label="Загрузка HMM-диагностики..." />}
                >
                  <HMMView hmm={result.hmm} />
                </Suspense>
              )}
              <ZapChannelsCard baseline={result.basic_statistics} />
              <AuditTable audit={result.data_audit} />
              <DetectedColumns detection={result.detected_columns} />
              <ChartsGrid charts={result.charts} />
              <WarningsList warnings={result.warnings} errors={result.errors} />

              <p className="text-xs text-brand-700/60 text-center">
                Алгоритм: v{result.algo_version}. Инварианты: observations =
                ЗАП; скрытые состояния — маневрирование → КФВ → ВУП.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
};
