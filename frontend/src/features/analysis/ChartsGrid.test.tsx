import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ChartsGrid } from "./ChartsGrid";
import type { ChartData } from "@/api/types";

describe("ChartsGrid", () => {
  it("renders heatmap matrix values, not bar placeholders", () => {
    const chart: ChartData = {
      id: "hmm_transition_matrix",
      title: "Матрица переходов HMM",
      kind: "heatmap",
      x: ["маневрирование", "КФВ", "ВУП"],
      y: ["маневрирование", "КФВ", "ВУП"],
      series: [
        {
          matrix: [
            [0.6, 0.3, 0.1],
            [0.1, 0.7, 0.2],
            [0.05, 0.15, 0.8],
          ],
        },
      ],
      meta: {},
    };
    render(<ChartsGrid charts={[chart]} />);

    // Заголовок отрисован.
    expect(screen.getByText("Матрица переходов HMM")).toBeInTheDocument();

    // Хотя бы одно числовое значение из матрицы попало в DOM.
    expect(screen.getByText("0.60")).toBeInTheDocument();
    expect(screen.getByText("0.80")).toBeInTheDocument();

    // Подписи строк/столбцов из x/y отрисованы (как минимум одна,
    // дубликаты по обеим осям не считаем).
    expect(screen.getAllByText("КФВ").length).toBeGreaterThan(0);
  });

  it("renders empty heatmap message when matrix is missing", () => {
    const chart: ChartData = {
      id: "hmm_transition_matrix",
      title: "Матрица переходов HMM",
      kind: "heatmap",
      x: [],
      y: [],
      series: [],
      meta: {},
    };
    render(<ChartsGrid charts={[chart]} />);
    expect(
      screen.getByText("Нет данных для отображения матрицы.")
    ).toBeInTheDocument();
  });
});
