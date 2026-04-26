import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartData } from "@/api/types";
import { Section } from "@/components/ui";

function toBarData(chart: ChartData): { label: string; value: number }[] {
  // Для bar у нас x — метки, y — значения; для hbar — наоборот (x = значения, y = метки).
  if (chart.kind === "hbar") {
    return (chart.y as string[]).map((label, i) => ({
      label: String(label),
      value: Number((chart.x as (number | string)[])[i]) || 0,
    }));
  }
  return (chart.x as string[]).map((label, i) => ({
    label: String(label),
    value: Number((chart.y as (number | string)[])[i]) || 0,
  }));
}

const ChartCard: React.FC<{ chart: ChartData }> = ({ chart }) => {
  const data = toBarData(chart);
  const layout = chart.kind === "hbar" ? "vertical" : "horizontal";
  const shownCount =
    typeof chart.meta?.shown === "number" ? (chart.meta.shown as number) : data.length;

  return (
    <div className="rounded-md border border-brand-100 bg-white p-4">
      <div className="text-sm font-medium text-brand-900 mb-2">{chart.title}</div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout={layout} margin={{ top: 8, right: 16, bottom: 8, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef0f8" />
            {layout === "horizontal" ? (
              <>
                <XAxis dataKey="label" tick={{ fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
              </>
            ) : (
              <>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={160} />
              </>
            )}
            <Tooltip />
            <Bar dataKey="value" fill="#4f5bd5" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-[11px] text-brand-700/60">
        Показано элементов: {shownCount}
      </div>
    </div>
  );
};

function getChartSheet(chart: ChartData): string | null {
  const sheet = chart.meta?.sheet;
  if (typeof sheet === "string" && sheet.length > 0) return sheet;
  return null;
}

export const ChartsGrid: React.FC<{
  charts: ChartData[];
  sheetOrder?: string[];
}> = ({ charts, sheetOrder }) => {
  if (charts.length === 0) return null;

  const globalCharts: ChartData[] = [];
  const bySheet = new Map<string, ChartData[]>();
  for (const chart of charts) {
    const sheet = getChartSheet(chart);
    if (sheet === null) {
      globalCharts.push(chart);
      continue;
    }
    const bucket = bySheet.get(sheet);
    if (bucket) bucket.push(chart);
    else bySheet.set(sheet, [chart]);
  }

  const orderedSheetNames = (() => {
    const known = new Set(bySheet.keys());
    const ordered: string[] = [];
    if (sheetOrder) {
      for (const name of sheetOrder) {
        if (known.has(name)) {
          ordered.push(name);
          known.delete(name);
        }
      }
    }
    const remaining = Array.from(known).sort((a, b) =>
      a.localeCompare(b, "ru"),
    );
    return [...ordered, ...remaining];
  })();

  return (
    <Section
      title="Графики"
      description="Сводные графики по всему анализу и отдельные группы на каждый лист (пропуски и т.п.). Для ЗАП-распределений показываются только уверенные кандидаты."
    >
      <div className="space-y-6">
        {globalCharts.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-brand-900 mb-3">Сводка</h4>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {globalCharts.map((c) => (
                <ChartCard key={c.id} chart={c} />
              ))}
            </div>
          </div>
        )}
        {orderedSheetNames.map((sheetName) => {
          const sheetCharts = bySheet.get(sheetName) ?? [];
          if (sheetCharts.length === 0) return null;
          return (
            <div key={`sheet-${sheetName}`}>
              <h4 className="text-sm font-semibold text-brand-900 mb-3">
                Лист «{sheetName}»
              </h4>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                {sheetCharts.map((c) => (
                  <ChartCard key={c.id} chart={c} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </Section>
  );
};
