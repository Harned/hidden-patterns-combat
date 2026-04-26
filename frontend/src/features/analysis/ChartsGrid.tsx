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

function extractMatrix(chart: ChartData): number[][] | null {
  const series = chart.series?.[0];
  if (!series) return null;
  const matrix = (series as { matrix?: unknown }).matrix;
  if (!Array.isArray(matrix)) return null;
  const rows: number[][] = [];
  for (const row of matrix) {
    if (!Array.isArray(row)) return null;
    rows.push(row.map((v) => Number(v) || 0));
  }
  return rows;
}

const HeatmapCard: React.FC<{ chart: ChartData }> = ({ chart }) => {
  const matrix = extractMatrix(chart);
  const xLabels = (chart.x as unknown[]).map((v) => String(v));
  const yLabels = (chart.y as unknown[]).map((v) => String(v));

  if (!matrix || matrix.length === 0) {
    return (
      <div className="rounded-md border border-brand-100 bg-white p-4">
        <div className="text-sm font-medium text-brand-900 mb-2">{chart.title}</div>
        <div className="text-xs text-brand-700/60">
          Нет данных для отображения матрицы.
        </div>
      </div>
    );
  }

  const flat = matrix.flat();
  const maxValue = flat.length ? Math.max(...flat) : 0;
  const minValue = flat.length ? Math.min(...flat) : 0;
  const range = maxValue - minValue || 1;

  const cellColor = (v: number) => {
    // линейная интерполяция от слабо-серого до основного бренда (#4f5bd5)
    const t = (v - minValue) / range;
    const r = Math.round(238 + (79 - 238) * t);
    const g = Math.round(240 + (91 - 240) * t);
    const b = Math.round(248 + (213 - 248) * t);
    return `rgb(${r},${g},${b})`;
  };

  const textColor = (v: number) => {
    const t = (v - minValue) / range;
    return t > 0.55 ? "#ffffff" : "#1f2937";
  };

  // Динамический шаблон колонок: первый — подпись строки, остальные — значения.
  const gridTemplate = `minmax(120px, 0.7fr) repeat(${xLabels.length}, minmax(0, 1fr))`;

  return (
    <div className="rounded-md border border-brand-100 bg-white p-4">
      <div className="text-sm font-medium text-brand-900 mb-2">{chart.title}</div>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-1" style={{ gridTemplateColumns: gridTemplate }}>
          <div />
          {xLabels.map((lbl) => (
            <div
              key={`x-${lbl}`}
              className="text-[11px] text-brand-700 text-center px-1 py-0.5 truncate"
              title={lbl}
            >
              {lbl}
            </div>
          ))}
          {matrix.map((row, ri) => (
            <React.Fragment key={`row-${ri}`}>
              <div
                className="text-[11px] text-brand-700 px-1 py-0.5 truncate"
                title={yLabels[ri] ?? `row ${ri}`}
              >
                {yLabels[ri] ?? `row ${ri}`}
              </div>
              {row.map((v, ci) => (
                <div
                  key={`cell-${ri}-${ci}`}
                  className="text-[11px] text-center px-1 py-2 rounded-sm tabular-nums"
                  style={{ backgroundColor: cellColor(v), color: textColor(v) }}
                  title={`${yLabels[ri] ?? ri} → ${xLabels[ci] ?? ci}: ${v.toFixed(3)}`}
                >
                  {v.toFixed(2)}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
      <div className="mt-2 text-[11px] text-brand-700/60">
        min={minValue.toFixed(3)}, max={maxValue.toFixed(3)}.
        Строки → столбцы — вероятности перехода/эмиссии.
      </div>
    </div>
  );
};

const ChartCard: React.FC<{ chart: ChartData }> = ({ chart }) => {
  if (chart.kind === "heatmap") {
    return <HeatmapCard chart={chart} />;
  }
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
