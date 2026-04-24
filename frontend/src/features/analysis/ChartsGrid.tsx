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

export const ChartsGrid: React.FC<{ charts: ChartData[] }> = ({ charts }) => {
  if (charts.length === 0) return null;
  return (
    <Section
      title="Графики"
      description="Распределения и пропуски, собранные напрямую из AnalysisResult. Для ЗАП-распределений показываются только уверенные кандидаты."
    >
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {charts.map((c) => (
          <ChartCard key={c.id} chart={c} />
        ))}
      </div>
    </Section>
  );
};
