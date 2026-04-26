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
const NOOP_COLOR = "#cbd5e1";

interface Props {
  hmm: HMMResult;
  maxTrajectories?: number;
}

export const ViterbiTimeline: React.FC<Props> = ({ hmm, maxTrajectories = 20 }) => {
  const trajectories: HMMTrajectory[] = hmm.trajectories.slice(0, maxTrajectories);
  const states = hmm.parameters.state_labels;
  const [expanded, setExpanded] = useState<number | null>(null);
  const [showPosterior, setShowPosterior] = useState<boolean>(false);

  const legend = useMemo(
    () =>
      states.map((s) => ({
        name: s,
        color: STATE_COLORS[s] ?? DEFAULT_COLOR,
      })),
    [states]
  );

  const anyPosterior = useMemo(
    () => trajectories.some((t) => Array.isArray(t.state_posterior)),
    [trajectories]
  );

  if (trajectories.length === 0) return null;

  return (
    <div>
      <div className="flex items-center gap-3 mb-1">
        <div className="text-sm font-medium text-brand-900">
          Viterbi: скрытая траектория по эпизодам
        </div>
        {anyPosterior && (
          <label className="flex items-center gap-2 text-xs text-brand-700">
            <input
              type="checkbox"
              checked={showPosterior}
              onChange={(e) => setShowPosterior(e.target.checked)}
              className="accent-brand-600"
            />
            показать posterior γ под каждым эпизодом
          </label>
        )}
      </div>
      <p className="text-xs text-brand-700/70 mb-4">
        Каждая строка — один эпизод. Цветные блоки — наиболее вероятные
        состояния на каждом шаге observation. Серые блоки — шаги без ЗАП
        (наблюдение = noop, состояние восстановлено по приору).
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
        <div className="flex items-center gap-2 text-xs">
          <span
            className="inline-block w-3 h-3 rounded-sm"
            style={{ background: NOOP_COLOR }}
          />
          <span className="text-brand-900">noop (без ЗАП)</span>
        </div>
      </div>

      <div className="space-y-2">
        {trajectories.map((tr, idx) => {
          const isExpanded = expanded === idx;
          const dim = !tr.has_zap;
          return (
            <div
              key={idx}
              className="rounded-md border border-brand-100 bg-white"
              style={dim ? { opacity: 0.6 } : undefined}
            >
              <div className="flex items-center gap-3 px-3 py-2 text-xs flex-wrap">
                <Badge tone="neutral">{tr.sheet}</Badge>
                <span className="text-brand-700/70">эпизод #{tr.episode_index}</span>
                <span className="text-brand-700/70">len={tr.length}</span>
                <span className="text-brand-700/70">
                  logL={tr.log_likelihood.toFixed(2)}
                </span>
                {tr.confidence !== null && (
                  <Badge tone={tr.confidence >= 0.7 ? "success" : "warning"}>
                    confidence {(tr.confidence * 100).toFixed(0)}%
                  </Badge>
                )}
                {!tr.has_zap && (
                  <Badge tone="warning">без ZAP — Viterbi по приору</Badge>
                )}
                <button
                  onClick={() => setExpanded(isExpanded ? null : idx)}
                  className="ml-auto text-brand-600 hover:text-brand-700"
                >
                  {isExpanded ? "свернуть" : "показать наблюдения"}
                </button>
              </div>
              <div className="flex gap-0.5 px-3 pb-3">
                {tr.state_path.map((state, i) => {
                  const isNoop = tr.observation_tokens[i] === "_noop_";
                  return (
                    <div
                      key={i}
                      title={`${state} ← ${tr.observation_tokens[i]}`}
                      className="h-5 flex-1 rounded-sm"
                      style={{
                        background: isNoop
                          ? NOOP_COLOR
                          : STATE_COLORS[state] ?? DEFAULT_COLOR,
                        minWidth: 8,
                      }}
                    />
                  );
                })}
              </div>
              {showPosterior && Array.isArray(tr.state_posterior) && (
                <div className="px-3 pb-3">
                  <div className="text-[11px] text-brand-700/60 mb-1">
                    posterior γ_t (стэк по состояниям)
                  </div>
                  <div className="flex gap-0.5">
                    {tr.state_posterior.map((row, i) => (
                      <div
                        key={i}
                        className="h-3 flex-1 flex flex-col-reverse rounded-sm overflow-hidden"
                        style={{ minWidth: 8 }}
                        title={row
                          .map((p, k) => `${states[k]}: ${(p * 100).toFixed(0)}%`)
                          .join("\n")}
                      >
                        {row.map((p, k) => (
                          <div
                            key={k}
                            style={{
                              height: `${Math.max(0, p * 100)}%`,
                              background: STATE_COLORS[states[k]] ?? DEFAULT_COLOR,
                            }}
                          />
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}
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
