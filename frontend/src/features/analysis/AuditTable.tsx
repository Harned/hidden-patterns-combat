import React from "react";
import type { AuditReport } from "@/api/types";
import { Section } from "@/components/ui";

export const AuditTable: React.FC<{ audit: AuditReport }> = ({ audit }) => (
  <Section
    title="Аудит файла"
    description={`Всего строк: ${audit.total_rows}. Доля пропусков: ${(
      audit.overall_null_ratio * 100
    ).toFixed(1)}%.`}
  >
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-brand-700/70 border-b border-brand-100">
            <th className="py-2 pr-4 font-medium">Лист</th>
            <th className="py-2 pr-4 font-medium">Строки</th>
            <th className="py-2 pr-4 font-medium">Колонки</th>
            <th className="py-2 pr-4 font-medium">Подозрительное</th>
          </tr>
        </thead>
        <tbody>
          {audit.sheets.map((s) => (
            <tr key={s.name} className="border-b border-brand-50 align-top">
              <td className="py-2 pr-4 font-medium">{s.name}</td>
              <td className="py-2 pr-4">{s.n_rows}</td>
              <td className="py-2 pr-4">{s.n_cols}</td>
              <td className="py-2 pr-4">
                {s.suspicious.length === 0 ? (
                  <span className="text-brand-700/60">—</span>
                ) : (
                  <ul className="list-disc ml-5 space-y-0.5">
                    {s.suspicious.slice(0, 5).map((x, i) => (
                      <li key={i} className="text-xs">
                        {x}
                      </li>
                    ))}
                    {s.suspicious.length > 5 && (
                      <li className="text-xs opacity-60">
                        ...и ещё {s.suspicious.length - 5}
                      </li>
                    )}
                  </ul>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </Section>
);
