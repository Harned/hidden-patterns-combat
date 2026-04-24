import React from "react";
import type { ColumnDetectionReport } from "@/api/types";
import { Badge, Section } from "@/components/ui";

export const DetectedColumns: React.FC<{ detection: ColumnDetectionReport }> = ({
  detection,
}) => (
  <Section
    title="Сопоставление колонок"
    description="Эвристика по заголовкам и содержимому. Любой кандидат требует ручного подтверждения перед содержательным анализом."
  >
    <div className="flex flex-wrap gap-2 mb-4">
      <div className="text-xs text-brand-700/70">Уверенно распознано:</div>
      {detection.detected_groups.length === 0 ? (
        <Badge tone="warning">ничего</Badge>
      ) : (
        detection.detected_groups.map((g) => (
          <Badge key={g} tone="info">
            {g}
          </Badge>
        ))
      )}
    </div>

    <div className="flex flex-wrap gap-2 mb-5">
      <div className="text-xs text-brand-700/70">Не распознано:</div>
      {detection.missing_groups.length === 0 ? (
        <Badge tone="success">нет</Badge>
      ) : (
        detection.missing_groups.map((g) => (
          <Badge key={g} tone="warning">
            {g}
          </Badge>
        ))
      )}
    </div>

    {detection.candidates.length > 0 && (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-brand-700/70 border-b border-brand-100">
              <th className="py-2 pr-4 font-medium">Группа</th>
              <th className="py-2 pr-4 font-medium">Лист</th>
              <th className="py-2 pr-4 font-medium">Колонка</th>
              <th className="py-2 pr-4 font-medium">Score</th>
              <th className="py-2 pr-4 font-medium">Обоснование</th>
            </tr>
          </thead>
          <tbody>
            {detection.candidates.slice(0, 30).map((c, i) => (
              <tr key={i} className="border-b border-brand-50 align-top">
                <td className="py-2 pr-4">{c.group}</td>
                <td className="py-2 pr-4">{c.sheet}</td>
                <td className="py-2 pr-4 font-mono text-xs">{c.column}</td>
                <td className="py-2 pr-4">
                  {c.score >= 0.7 ? (
                    <Badge tone="success">{c.score.toFixed(2)}</Badge>
                  ) : (
                    <Badge tone="warning">{c.score.toFixed(2)}</Badge>
                  )}
                </td>
                <td className="py-2 pr-4 text-xs text-brand-900/80">{c.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {detection.candidates.length > 30 && (
          <div className="text-xs text-brand-700/60 mt-2">
            ...и ещё {detection.candidates.length - 30}
          </div>
        )}
      </div>
    )}

    {detection.assumptions.length > 0 && (
      <div className="mt-5 rounded-md bg-brand-50 border border-brand-100 px-4 py-3">
        <div className="text-xs font-semibold text-brand-900 mb-1">Предположения</div>
        <ul className="list-disc ml-5 text-xs text-brand-900/80 space-y-0.5">
          {detection.assumptions.map((a, i) => (
            <li key={i}>{a}</li>
          ))}
        </ul>
      </div>
    )}
  </Section>
);
