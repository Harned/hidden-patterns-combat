import React from "react";
import type { WarningItem, WarningSeverity } from "@/api/types";
import { Section } from "@/components/ui";
import { translateWarning } from "@/i18n/warnings";

const severityStyles: Record<WarningSeverity, string> = {
  info: "border-sky-200 bg-sky-50 text-sky-900",
  warning: "border-amber-200 bg-amber-50 text-amber-900",
  error: "border-red-200 bg-red-50 text-red-900",
};

const severityLabels: Record<WarningSeverity, string> = {
  info: "info",
  warning: "warning",
  error: "error",
};

function groupByCode(items: WarningItem[]): Record<string, WarningItem[]> {
  const grouped: Record<string, WarningItem[]> = {};
  for (const it of items) {
    (grouped[it.code] ??= []).push(it);
  }
  return grouped;
}

export const WarningsList: React.FC<{
  warnings: WarningItem[];
  errors?: WarningItem[];
}> = ({ warnings, errors = [] }) => {
  if (warnings.length === 0 && errors.length === 0) {
    return null;
  }
  const all = [...errors, ...warnings];
  const grouped = groupByCode(all);

  return (
    <Section
      title="Предупреждения и ошибки"
      description="Группировка по коду события. Все замечания алгоритма собраны честно; при недостатке данных алгоритм предупреждает, а не додумывает."
    >
      <div className="space-y-3">
        {Object.entries(grouped).map(([code, items]) => {
          const severity = items[0].severity;
          const translation = translateWarning(code);
          const title = translation?.short ?? items[0].message;
          return (
            <details
              key={code}
              className={`rounded-md border px-4 py-3 ${severityStyles[severity]}`}
            >
              <summary className="cursor-pointer text-sm font-medium flex items-center gap-2 flex-wrap">
                <span className="uppercase text-[10px] tracking-wide">
                  {severityLabels[severity]}
                </span>
                <span>{title}</span>
                <span className="font-mono text-[11px] opacity-70">{code}</span>
                <span className="text-xs opacity-70">×{items.length}</span>
              </summary>
              <ul className="mt-2 space-y-1 text-sm">
                {items.slice(0, 30).map((w, i) => (
                  <li key={i} className="leading-snug">
                    {w.message}
                  </li>
                ))}
                {items.length > 30 && (
                  <li className="opacity-70 text-xs">
                    ...и ещё {items.length - 30}
                  </li>
                )}
              </ul>
            </details>
          );
        })}
      </div>
    </Section>
  );
};
