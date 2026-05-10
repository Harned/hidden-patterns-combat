import React, { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { MarkovAthleteResult, MarkovResult } from "@/api/types";
import { Button, Card } from "@/components/ui";

const STATE_LABELS_RU: Record<string, string> = {
  manoeuvring: "Манёвр",
  grip: "КФВ",
  off_balance: "ВУП",
  technical_action: "ЗАП",
  pause: "Пауза",
};

const STYLE_LABELS_RU: Record<string, string> = {
  endurance: "Выносливый",
  speed_power: "Скоростно-силовой",
  burnout: "Провал по дистанции",
  unclassified: "—",
};

const STYLE_COLORS: Record<string, string> = {
  endurance: "bg-blue-100 text-blue-800",
  speed_power: "bg-amber-100 text-amber-800",
  burnout: "bg-red-100 text-red-800",
  unclassified: "bg-gray-100 text-gray-600",
};

// Heat-map colour for transition probability (0→white, 1→brand-700)
function heatColor(value: number): string {
  const pct = Math.round(value * 100);
  if (pct >= 80) return "bg-brand-600 text-white";
  if (pct >= 60) return "bg-brand-400 text-white";
  if (pct >= 40) return "bg-brand-300 text-brand-900";
  if (pct >= 20) return "bg-brand-200 text-brand-800";
  if (pct >= 5) return "bg-brand-100 text-brand-700";
  return "bg-white text-brand-400";
}

const TransitionMatrix: React.FC<{ result: MarkovAthleteResult }> = ({ result }) => {
  const { transition_matrix, state_labels } = result;
  if (!transition_matrix || transition_matrix.length === 0) return null;

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse min-w-full">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left font-medium text-brand-600 border border-brand-100">
              из ↓ / в →
            </th>
            {state_labels.map((s) => (
              <th
                key={s}
                className="px-2 py-1 font-medium text-brand-700 border border-brand-100 text-center"
              >
                {STATE_LABELS_RU[s] ?? s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {transition_matrix.map((row, i) => (
            <tr key={state_labels[i]}>
              <td className="px-2 py-1 font-medium text-brand-700 border border-brand-100 whitespace-nowrap">
                {STATE_LABELS_RU[state_labels[i]] ?? state_labels[i]}
              </td>
              {row.map((val, j) => (
                <td
                  key={j}
                  className={`px-2 py-1 border border-brand-100 text-center tabular-nums ${heatColor(val)}`}
                >
                  {val.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const StationaryBar: React.FC<{ result: MarkovAthleteResult }> = ({ result }) => {
  const { stationary, state_labels } = result;
  const colors = ["bg-blue-500", "bg-amber-500", "bg-emerald-500", "bg-purple-500", "bg-gray-300"];
  return (
    <div className="space-y-1">
      {state_labels.map((s, i) => {
        const pct = Math.round((stationary[s] ?? 0) * 100);
        return (
          <div key={s} className="flex items-center gap-2">
            <span className="text-xs text-brand-600 w-16 shrink-0">
              {STATE_LABELS_RU[s] ?? s}
            </span>
            <div className="flex-1 bg-brand-50 rounded h-4 overflow-hidden">
              <div
                className={`h-full ${colors[i % colors.length]}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-xs tabular-nums text-brand-700 w-10 text-right">
              {pct}%
            </span>
          </div>
        );
      })}
    </div>
  );
};

const MetricsRow: React.FC<{ result: MarkovAthleteResult }> = ({ result }) => {
  const m = result.episode_metrics;
  const styleKey = result.style ?? "unclassified";
  return (
    <div className="flex flex-wrap gap-3 text-xs text-brand-700">
      <span>
        <span className="font-medium">Эпизодов:</span> {m.episode_count}
      </span>
      <span>
        <span className="font-medium">Поединков:</span> {m.bout_count}
      </span>
      {m.action_density != null && (
        <span>
          <span className="font-medium">Плотность:</span>{" "}
          {m.action_density.toFixed(2)} акт/эп
        </span>
      )}
      {m.action_rate_per_second != null && (
        <span>
          <span className="font-medium">Темп:</span>{" "}
          {m.action_rate_per_second.toFixed(3)}/с
        </span>
      )}
      {m.non_technical_share != null && (
        <span>
          <span className="font-medium">Без ЗАП:</span>{" "}
          {Math.round(m.non_technical_share * 100)}%
        </span>
      )}
      {m.activity_evenness != null && (
        <span>
          <span className="font-medium">Равномерность:</span>{" "}
          {m.activity_evenness.toFixed(3)}
        </span>
      )}
      <span
        className={`inline-flex items-center rounded-full px-2 py-0.5 font-medium ${STYLE_COLORS[styleKey] ?? STYLE_COLORS.unclassified}`}
      >
        {STYLE_LABELS_RU[styleKey] ?? styleKey}
      </span>
    </div>
  );
};

const AthleteCard: React.FC<{ result: MarkovAthleteResult }> = ({ result }) => {
  const [expanded, setExpanded] = useState(false);
  return (
    <Card className="px-4 py-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <button
          className="text-sm font-medium text-brand-900 hover:text-brand-600 text-left"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "▼" : "▶"} {result.athlete}
        </button>
        <MetricsRow result={result} />
      </div>
      {expanded && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-brand-100">
          <div>
            <div className="text-xs font-medium text-brand-600 mb-1">
              Матрица переходов
            </div>
            <TransitionMatrix result={result} />
          </div>
          <div>
            <div className="text-xs font-medium text-brand-600 mb-1">
              Стационарное распределение π
            </div>
            <StationaryBar result={result} />
          </div>
        </div>
      )}
    </Card>
  );
};

export const MarkovView: React.FC<{ sourceId: number }> = ({ sourceId }) => {
  const qc = useQueryClient();
  const [data, setData] = useState<MarkovResult | null>(null);
  const [filter, setFilter] = useState("");

  const mutation = useMutation({
    mutationFn: () => api.runMarkov(sourceId),
    onSuccess: (result) => {
      setData(result);
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
    },
  });

  const filtered = data
    ? data.athletes.filter((a) =>
        a.athlete.toLowerCase().includes(filter.toLowerCase())
      )
    : [];

  return (
    <div className="space-y-4">
      <Card className="px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <div className="text-sm font-medium text-brand-900">
              5-state Observable Markov Chain (TASK_SPEC_011)
            </div>
            <div className="text-xs text-brand-700/70">
              Индивидуальные Марков-профили спортсменов: матрица переходов A,
              стационарное распределение π, episode-метрики и описательный
              стиль. Состояния:{" "}
              <span className="font-medium">
                маневрирование → КФВ → ВУП → ЗАП → пауза
              </span>
              . Все выводы вероятностные.
            </div>
            {data && (
              <div className="text-xs text-brand-700/60">
                Обработано спортсменов: {data.summary.athletes_rendered} /{" "}
                {data.summary.athletes_total}
                {data.summary.skipped_athletes.length > 0 && (
                  <span className="ml-2 text-amber-600">
                    (пропущено: {data.summary.skipped_athletes.join(", ")})
                  </span>
                )}
              </div>
            )}
          </div>
          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
          >
            {mutation.isPending
              ? "Строим модели..."
              : data
              ? "Пересчитать"
              : "Построить модели"}
          </Button>
        </div>

        {mutation.isError && (
          <div className="mt-3 rounded-md bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-800">
            {mutation.error instanceof Error
              ? mutation.error.message
              : "Ошибка при построении моделей"}
          </div>
        )}

        {data &&
          [
            ...data.summary.config_warnings,
            ...data.summary.column_validation_warnings,
            ...data.summary.split_warnings,
          ]
            .slice(0, 5)
            .map((w, i) => (
              <div
                key={i}
                className="mt-2 rounded bg-amber-50 border border-amber-200 px-3 py-1.5 text-xs text-amber-800"
              >
                {w.code}: {w.message}
              </div>
            ))}
      </Card>

      {data && data.athletes.length > 0 && (
        <>
          <div className="flex items-center gap-3">
            <input
              type="text"
              placeholder="Фильтр по ФИО..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="h-8 flex-1 max-w-xs rounded-md border border-brand-200 px-3 text-sm"
            />
            <span className="text-xs text-brand-700/60">
              {filtered.length} из {data.athletes.length}
            </span>
          </div>
          <div className="space-y-2">
            {filtered.map((a) => (
              <AthleteCard key={a.athlete} result={a} />
            ))}
          </div>
        </>
      )}

      {data && data.athletes.length === 0 && (
        <Card className="px-6 py-8 text-center text-brand-700/60 text-sm">
          Нет данных по спортсменам. Проверьте state_groups.yaml и column
          mapping.
        </Card>
      )}
    </div>
  );
};
