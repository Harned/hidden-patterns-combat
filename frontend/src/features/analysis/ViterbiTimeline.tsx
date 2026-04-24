import React, { useMemo, useState } from "react";
import type { HMMResult, HMMTrajectory } from "@/api/types";
import { Badge } from "@/components/ui";

// Фиксированная палитра — имена состояний из домена. Детальная модель
// (7 состояний) покрывается полностью, basic попадает по именам.
const STATE_COLORS: Record<string, string> = {
  маневрирование: "#5e9dff",
  КФВ: "#f59e0b",
  ВУП: "#10b981",
  маневры: "#5e9dff",
  захваты: "#f59e0b",
  хваты: "#d97706",
  обхваты: "#ef4444",
  прихваты: "#a855f7",
  упоры: "#ec4899",
};

const DEFAULT_COLOR = "#94a3b8";

interface Props {
  hmm: HMMResult;
  maxTrajectories?: number;
}

export const ViterbiTimeline: React.FC<Props> = ({ hmm, maxTrajectories = 20 }) => {
  const trajectories: HMMTrajectory[] = hmm.trajectories.slice(0, maxTrajectories);
  const states = hmm.parameters.state_labels;
  const [expanded, setExpanded] = useState<number | null>(null);

  const legend = useMemo(
    () =>
      states.map((s) => ({
        name: s,
        color: STATE_COLORS[s] ?? DEFAULT_COLOR,
      })),
    [states]
  );

  if (trajectories.length === 0) return null;

  return (
    <div>
      <div className="text-sm font-medium text-brand-900 mb-1">
        Viterbi: скрытая траектория по эпизодам
      </div>
      <p className="text-xs text-brand-700/70 mb-4">
        Каждая строка — один эпизод. Цветные блоки — наиболее вероятные
        состояния на каждом шаге observation.
      </p>
      <div className="flex flex-wrap gap-2 mb-4">
        {legend.map((l) => (
          <div key={l.name} className="flex items-center gap-2 text-xs">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{ background: l.color }}
            />
            <span className="text-brand-900">{l.name}</span>
          </div>
        ))}
      </div>

      <div className="space-y-2">
        {trajectories.map((tr, idx) => {
          const isExpanded = expanded === idx;
          return (
            <div
              key={idx}
              className="rounded-md border border-brand-100 bg-white"
            >
              <div className="flex items-center gap-3 px-3 py-2 text-xs">
                <Badge tone="neutral">{tr.sheet}</Badge>
                <span className="text-brand-700/70">эпизод #{tr.episode_index}</span>
                <span className="text-brand-700/70">len={tr.length}</span>
                <span className="text-brand-700/70">
                  logL={tr.log_likelihood.toFixed(2)}
                </span>
                <button
                  onClick={() => setExpanded(isExpanded ? null : idx)}
                  className="ml-auto text-brand-600 hover:text-brand-700"
                >
                  {isExpanded ? "свернуть" : "показать наблюдения"}
                </button>
              </div>
              <div className="flex gap-0.5 px-3 pb-3">
                {tr.state_path.map((state, i) => (
                  <div
                    key={i}
                    title={`${state} ← ${tr.observation_tokens[i]}`}
                    className="h-5 flex-1 rounded-sm"
                    style={{
                      background: STATE_COLORS[state] ?? DEFAULT_COLOR,
                      minWidth: 8,
                    }}
                  />
                ))}
              </div>
              {isExpanded && (
                <div className="border-t border-brand-100 px-3 py-2 text-xs font-mono text-brand-900/90">
                  <div>
                    <span className="text-brand-700/60">obs:</span>{" "}
                    {tr.observation_tokens.join(" → ")}
                  </div>
                  <div className="mt-1">
                    <span className="text-brand-700/60">states:</span>{" "}
                    {tr.state_path.join(" → ")}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {hmm.trajectories.length > trajectories.length && (
        <p className="mt-3 text-xs text-brand-700/60">
          Показаны первые {trajectories.length} из {hmm.trajectories.length} эпизодов.
        </p>
      )}
    </div>
  );
};
