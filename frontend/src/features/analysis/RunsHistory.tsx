import React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { AnalysisRunSummary } from "@/api/types";
import { Badge, Card } from "@/components/ui";

const STATE_TONE: Record<
  AnalysisRunSummary["state"],
  "neutral" | "info" | "success" | "warning" | "danger"
> = {
  pending: "neutral",
  running: "info",
  done: "success",
  failed: "danger",
};

interface Props {
  sourceId: number;
  selectedRunId?: number | null;
  onSelect?: (run: AnalysisRunSummary) => void;
}

const fmtDate = (value: string | null) =>
  value ? new Date(value).toLocaleString("ru-RU") : "—";

export const RunsHistory: React.FC<Props> = ({
  sourceId,
  selectedRunId,
  onSelect,
}) => {
  const query = useQuery<AnalysisRunSummary[]>({
    queryKey: ["runs", sourceId],
    queryFn: () => api.listRuns(sourceId),
    refetchInterval: 2000,
  });

  const runs = query.data ?? [];

  if (query.isLoading) {
    return (
      <Card className="px-6 py-6 text-sm text-brand-700/70 text-center">
        Загрузка истории запусков...
      </Card>
    );
  }

  if (runs.length === 0) {
    return (
      <Card className="px-6 py-6 text-sm text-brand-700/70 text-center">
        Запусков пока нет.
      </Card>
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border border-brand-100 bg-white">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-brand-700/70 bg-brand-50">
            <th className="py-2 px-3 font-medium">#</th>
            <th className="py-2 px-3 font-medium">Состояние</th>
            <th className="py-2 px-3 font-medium">Статус</th>
            <th className="py-2 px-3 font-medium">Режим</th>
            <th className="py-2 px-3 font-medium">Создан</th>
            <th className="py-2 px-3 font-medium">Завершён</th>
            <th className="py-2 px-3 font-medium">Ошибка</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const active = selectedRunId === r.id;
            return (
              <tr
                key={r.id}
                onClick={() => onSelect?.(r)}
                className={`border-t border-brand-100 cursor-pointer ${
                  active ? "bg-brand-100/60" : "hover:bg-brand-50"
                }`}
              >
                <td className="py-2 px-3 font-mono">{r.id}</td>
                <td className="py-2 px-3">
                  <Badge tone={STATE_TONE[r.state]}>{r.state}</Badge>
                </td>
                <td className="py-2 px-3">{r.status || "—"}</td>
                <td className="py-2 px-3 font-mono text-xs">{r.hmm_mode}</td>
                <td className="py-2 px-3 text-xs">{fmtDate(r.created_at)}</td>
                <td className="py-2 px-3 text-xs">{fmtDate(r.finished_at)}</td>
                <td className="py-2 px-3 text-xs text-red-700">
                  {r.error ? r.error.slice(0, 80) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
