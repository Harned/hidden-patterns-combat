import React from "react";
import type { HMMResult } from "@/api/types";
import { Badge, Section } from "@/components/ui";
import { ViterbiTimeline } from "./ViterbiTimeline";

const fmt = (v: number, digits = 3) => Number.isFinite(v) ? v.toFixed(digits) : "—";

const TransitionHeatmap: React.FC<{ hmm: HMMResult }> = ({ hmm }) => {
  const labels = hmm.parameters.state_labels;
  const A = hmm.parameters.transition_matrix;
  return (
    <div className="overflow-x-auto">
      <table className="border border-brand-100 rounded-md text-sm">
        <thead>
          <tr>
            <th className="px-3 py-2 bg-brand-50 text-brand-700/70 text-left">
              из ↓ / в →
            </th>
            {labels.map((l) => (
              <th key={l} className="px-3 py-2 bg-brand-50 text-brand-700/70">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {A.map((row, i) => (
            <tr key={labels[i]} className="border-t border-brand-100">
              <th className="px-3 py-2 text-left font-medium text-brand-900">
                {labels[i]}
              </th>
              {row.map((v, j) => {
                const intensity = Math.max(0, Math.min(1, v));
                const bg = `rgba(79, 91, 213, ${0.1 + intensity * 0.7})`;
                const text = intensity > 0.5 ? "white" : "#1a1f4b";
                return (
                  <td
                    key={j}
                    className="px-3 py-2 text-center font-mono text-xs"
                    style={{ background: bg, color: text }}
                  >
                    {v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const StateDistribution: React.FC<{ hmm: HMMResult }> = ({ hmm }) => {
  const entries = Object.entries(hmm.state_distribution);
  return (
    <div className="space-y-2">
      {entries.map(([state, share]) => {
        const pct = Math.round(share * 1000) / 10;
        return (
          <div key={state} className="flex items-center gap-3">
            <div className="w-40 text-sm">{state}</div>
            <div className="flex-1 bg-brand-100 rounded h-3 overflow-hidden">
              <div
                className="bg-brand-500 h-3"
                style={{ width: `${Math.max(2, pct)}%` }}
              />
            </div>
            <div className="w-16 text-right text-sm font-mono">{pct}%</div>
          </div>
        );
      })}
    </div>
  );
};

const TrajectoriesTable: React.FC<{ hmm: HMMResult }> = ({ hmm }) => {
  const rows = hmm.trajectories.slice(0, 30);
  if (rows.length === 0) return null;
  return (
    <div className="overflow-x-auto rounded-md border border-brand-100">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-brand-700/70 bg-brand-50">
            <th className="py-2 px-3 font-medium">Лист</th>
            <th className="py-2 px-3 font-medium">Эпизод</th>
            <th className="py-2 px-3 font-medium">Длина</th>
            <th className="py-2 px-3 font-medium">Наблюдения</th>
            <th className="py-2 px-3 font-medium">Скрытая траектория</th>
            <th className="py-2 px-3 font-medium">log L</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((tr, i) => (
            <tr key={i} className="border-t border-brand-100 align-top">
              <td className="py-2 px-3">{tr.sheet}</td>
              <td className="py-2 px-3">{tr.episode_index}</td>
              <td className="py-2 px-3">{tr.length}</td>
              <td className="py-2 px-3 font-mono text-xs max-w-xs break-words">
                {tr.observation_tokens.join(" → ")}
              </td>
              <td className="py-2 px-3 font-mono text-xs">
                {tr.state_path.join(" → ")}
              </td>
              <td className="py-2 px-3 font-mono text-xs">
                {fmt(tr.log_likelihood, 2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {hmm.trajectories.length > rows.length && (
        <div className="text-xs text-brand-700/60 p-2">
          Показаны первые {rows.length} из {hmm.trajectories.length} траекторий.
        </div>
      )}
    </div>
  );
};

export const HMMView: React.FC<{ hmm: HMMResult }> = ({ hmm }) => (
  <Section
    title={
      hmm.parameters.variant === "detailed_7state"
        ? "HMM-диагностика (7 состояний: маневры / захваты / хваты / обхваты / прихваты / упоры / ВУП)"
        : "HMM-диагностика (3 состояния: маневрирование / КФВ / ВУП)"
    }
    description="Все выводы вероятностные и ограничены данными."
  >
    <div className="flex flex-wrap gap-2 mb-5">
      <Badge tone={hmm.parameters.variant === "detailed_7state" ? "info" : "neutral"}>
        variant: {hmm.parameters.variant}
      </Badge>
      <Badge tone={hmm.parameters.converged ? "success" : "warning"}>
        converged: {String(hmm.parameters.converged)}
      </Badge>
      <Badge tone="info">iter: {hmm.parameters.n_iter}</Badge>
      <Badge tone="info">
        log L = {fmt(hmm.parameters.log_likelihood, 2)}
      </Badge>
      {hmm.parameters.bic !== null && (
        <Badge tone="info">BIC = {fmt(hmm.parameters.bic, 2)}</Badge>
      )}
      <Badge tone="neutral">seed: {hmm.parameters.random_seed}</Badge>
      <Badge tone="neutral">
        состояний: {hmm.parameters.n_states}
      </Badge>
      <Badge tone="neutral">
        алфавит: {hmm.parameters.n_observations}
      </Badge>
      <Badge tone={hmm.sanity["transition_dominance_ok"] ? "success" : "warning"}>
        sanity: {hmm.sanity["transition_dominance_ok"] ? "ok" : "weak"}
      </Badge>
    </div>

    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
      <div>
        <div className="text-sm font-medium text-brand-900 mb-2">
          Матрица переходов
        </div>
        <TransitionHeatmap hmm={hmm} />
      </div>
      <div>
        <div className="text-sm font-medium text-brand-900 mb-2">
          Распределение времени по состояниям
        </div>
        <StateDistribution hmm={hmm} />
      </div>
    </div>

    <div className="mt-6">
      <div className="text-sm font-medium text-brand-900 mb-2">
        Интерпретация
      </div>
      <pre className="whitespace-pre-wrap text-sm text-brand-900/90 font-sans bg-brand-50 border border-brand-100 rounded-md p-3">
        {hmm.interpretation}
      </pre>
    </div>

    <div className="mt-6">
      <div className="text-sm font-medium text-brand-900 mb-2">
        Скрытые траектории по эпизодам (табличный вид)
      </div>
      <TrajectoriesTable hmm={hmm} />
    </div>

    <div className="mt-6">
      <ViterbiTimeline hmm={hmm} />
    </div>
  </Section>
);
