import React, { useMemo, useState } from "react";
import type { MarkovAthleteResult, TrainerAthleteSummary } from "@/api/types";
import { Button, Card, Section } from "@/components/ui";

type SortDir = "desc" | "asc";

function dominantState(stationary: Record<string, number>): string | null {
  const entries = Object.entries(stationary);
  if (entries.length === 0) return null;
  return entries.reduce((best, cur) => (cur[1] > best[1] ? cur : best))[0];
}

export const TrainerSummary: React.FC<{
  summary: TrainerAthleteSummary | null;
  markovAthletes?: MarkovAthleteResult[] | null;
  onOpenMarkov?: () => void;
}> = ({ summary, markovAthletes, onOpenMarkov }) => {
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const markovMap = useMemo(() => {
    if (!markovAthletes) return null;
    const m = new Map<string, MarkovAthleteResult>();
    for (const a of markovAthletes) m.set(a.athlete, a);
    return m;
  }, [markovAthletes]);

  const rows = useMemo(() => {
    if (!summary) return [];
    const copy = [...summary.athletes];
    copy.sort((a, b) => {
      const diff = a.episode_count - b.episode_count;
      const byCount = sortDir === "desc" ? -diff : diff;
      if (byCount !== 0) return byCount;
      return a.athlete.localeCompare(b.athlete, "ru");
    });
    return copy;
  }, [summary, sortDir]);

  if (!summary) {
    return (
      <Card className="px-6 py-10 text-center text-brand-700/70">
        Сводка по спортсменам недоступна: похоже, анализ выполнялся без
        column mapping. Настройте роли «спортсмен» и «эпизод» в редакторе
        и перезапустите анализ.
      </Card>
    );
  }

  const hasData = rows.length > 0;
  const showMarkov = markovMap !== null && markovMap.size > 0;

  return (
    <Section
      title="Сводка по спортсменам"
      description="Описательная панель для тренера: кто фигурирует в выгрузке и сколько уникальных эпизодов записано на каждого. Это не диагностика и не оценка стиля — только перечень и счётчики."
    >
      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm text-brand-900/80">
        <span className="rounded-md bg-brand-50 border border-brand-200 px-2.5 py-1">
          Спортсменов: <b>{summary.total_athletes}</b>
        </span>
        <span className="rounded-md bg-brand-50 border border-brand-200 px-2.5 py-1">
          Эпизодов суммарно: <b>{summary.total_episodes}</b>
        </span>
        {showMarkov && onOpenMarkov && (
          <Button size="sm" variant="ghost" onClick={onOpenMarkov}>
            Марков-профили →
          </Button>
        )}
      </div>

      {summary.notes.length > 0 && (
        <ul className="mb-4 list-disc ml-5 text-sm text-brand-900/80 space-y-0.5">
          {summary.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}

      {hasData ? (
        <div className="overflow-x-auto rounded-md border border-brand-100 bg-white">
          <table className="min-w-full text-sm">
            <thead className="bg-brand-50 text-brand-900">
              <tr>
                <th className="px-3 py-2 text-left font-semibold">#</th>
                <th className="px-3 py-2 text-left font-semibold">ФИО</th>
                <th
                  className="px-3 py-2 text-right font-semibold cursor-pointer select-none"
                  onClick={() =>
                    setSortDir((d) => (d === "desc" ? "asc" : "desc"))
                  }
                  title="Кликните, чтобы переключить направление сортировки"
                >
                  Эпизоды {sortDir === "desc" ? "↓" : "↑"}
                </th>
                {showMarkov && (
                  <>
                    <th className="px-3 py-2 text-left font-semibold" title="Стиль ведения схватки по марковской модели">
                      Стиль
                    </th>
                    <th className="px-3 py-2 text-left font-semibold" title="Доминирующее состояние (argmax стационарного распределения π)">
                      π-лидер
                    </th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => {
                const mk = markovMap?.get(row.athlete);
                const dom = mk ? dominantState(mk.stationary) : null;
                return (
                  <tr
                    key={row.athlete}
                    className={idx % 2 === 0 ? "bg-white" : "bg-brand-50/40"}
                  >
                    <td className="px-3 py-2 text-brand-700/70">{idx + 1}</td>
                    <td className="px-3 py-2 text-brand-900">{row.athlete}</td>
                    <td className="px-3 py-2 text-right font-medium text-brand-900">
                      {row.episode_count}
                    </td>
                    {showMarkov && (
                      <>
                        <td className="px-3 py-2">
                          {mk?.style ? (
                            <span className="inline-flex items-center rounded-full bg-brand-100 text-brand-800 text-xs px-2 py-0.5">
                              {mk.style}
                            </span>
                          ) : (
                            <span className="text-brand-300 text-xs">—</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-xs text-brand-700">
                          {dom ?? <span className="text-brand-300">—</span>}
                        </td>
                      </>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <Card className="px-6 py-8 text-center text-brand-700/70">
          В выгрузке нет ни одной строки, где заполнены и ФИО, и номер
          эпизода — отображать нечего.
        </Card>
      )}
    </Section>
  );
};
