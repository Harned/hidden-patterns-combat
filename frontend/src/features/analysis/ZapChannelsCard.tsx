import React from "react";
import type { BaselineReport } from "@/api/types";
import { Badge, Section } from "@/components/ui";

interface Props {
  baseline: BaselineReport;
}

/**
 * Сводка ЗАП-событий по каналам. Показывается, только когда
 * `zap_events_by_channel` содержит ненулевые значения.
 */
export const ZapChannelsCard: React.FC<Props> = ({ baseline }) => {
  const channels = Object.entries(baseline.zap_events_by_channel).filter(
    ([, v]) => v > 0
  );
  if (channels.length === 0) return null;

  channels.sort((a, b) => b[1] - a[1]);
  const totalEvents = channels.reduce((acc, [, v]) => acc + v, 0);

  const kinds = baseline.zap_column_kinds;
  const kindCounts: Record<string, number> = {};
  for (const kind of Object.values(kinds)) {
    kindCounts[kind] = (kindCounts[kind] ?? 0) + 1;
  }

  return (
    <Section
      title="ЗАП-события по каналам"
      description={`Агрегация по доменным наблюдениям. Всего ${totalEvents} событий в ${channels.length} каналах.`}
    >
      <div className="flex flex-wrap gap-2 mb-3">
        {Object.entries(kindCounts).map(([kind, count]) => (
          <Badge key={kind} tone={kind === "empty" ? "neutral" : "info"}>
            {kind}: {count}
          </Badge>
        ))}
      </div>

      <div className="overflow-x-auto rounded-md border border-brand-100">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-brand-700/70 bg-brand-50">
              <th className="py-2 px-3 font-medium">Канал</th>
              <th className="py-2 px-3 font-medium w-32">События</th>
            </tr>
          </thead>
          <tbody>
            {channels.map(([name, count]) => (
              <tr key={name} className="border-t border-brand-100">
                <td className="py-2 px-3 font-mono text-xs">{name}</td>
                <td className="py-2 px-3">{count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-brand-700/60">
        Для binary/count колонок «событие» = эпизод со значением больше нуля.
        Поле <code>zap_total_triggers</code> в результате содержит сумму
        значений (может быть больше числа событий).
      </p>
    </Section>
  );
};
