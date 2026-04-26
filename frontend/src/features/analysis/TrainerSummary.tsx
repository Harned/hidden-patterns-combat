import React, { useMemo, useState } from "react";
import type { TrainerAthleteSummary } from "@/api/types";
import { Card, Section } from "@/components/ui";

type SortDir = "desc" | "asc";

export const TrainerSummary: React.FC<{
  summary: TrainerAthleteSummary | null;
}> = ({ summary }) => {
  const [sortDir, setSortDir] = useState<SortDir>("desc");

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
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr
                  key={row.athlete}
                  className={idx % 2 === 0 ? "bg-white" : "bg-brand-50/40"}
                >
                  <td className="px-3 py-2 text-brand-700/70">{idx + 1}</td>
                  <td className="px-3 py-2 text-brand-900">{row.athlete}</td>
                  <td className="px-3 py-2 text-right font-medium text-brand-900">
                    {row.episode_count}
                  </td>
                </tr>
              ))}
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
