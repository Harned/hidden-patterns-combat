import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type { CellEdit, GridCellValue, SheetGridFragment } from "@/api/types";
import { Button } from "@/components/ui";

export interface CellSuggestion {
  proposed: string;
  hint: string;
  /** Опциональный код причины — для будущих типов подсказок. */
  kind?: string;
}

interface Props {
  sourceId: number;
  sheetName: string;
  headerRows: number[];
  /** Сколько строк рендерим в окне; виртуализация — оконная пагинация. */
  pageSize?: number;
  onMutated?: () => void;
  /**
   * Карта предложений по ячейкам в формате `${row}:${col}` → подсказка.
   * Используется для подсветки и tooltip; значение в input не подменяется
   * до явного применения через `applySuggestions` или ручного ввода.
   */
  suggestionByCell?: Record<string, CellSuggestion>;
  /** Уведомление о наличии локальных несохранённых правок. */
  onDirtyChange?: (count: number) => void;
}

export interface SheetGridHandle {
  /**
   * Применить набор предложений (или иных правок) одним PUT, минуя локальный
   * draft. Используется кнопкой «Применить предложения» из мастера, чтобы
   * не требовать второго клика «Сохранить» в сетке.
   */
  applyEdits: (edits: CellEdit[]) => Promise<void>;
  /** Сбросить локальные правки без записи в файл. */
  discardLocal: () => void;
  /** Текущее количество локальных несохранённых правок. */
  dirtyCount: () => number;
}

interface PendingEdit {
  row: number;
  col: number;
  value: GridCellValue;
}

const COL_WIDTHS = "minmax(64px, 80px) repeat(var(--cols), minmax(140px, 1fr))";
const ROW_HEIGHT = 36;

const formatValue = (v: GridCellValue): string => {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
};

const parseValue = (raw: string): GridCellValue => {
  if (raw === "") return null;
  if (/^-?\d+$/.test(raw)) {
    const n = Number(raw);
    return Number.isSafeInteger(n) ? n : raw;
  }
  if (/^-?\d*\.\d+$/.test(raw)) {
    const n = Number(raw);
    return Number.isFinite(n) ? n : raw;
  }
  return raw;
};

/**
 * Виртуализированная Excel-подобная сетка. Ввод ведётся **локально** —
 * никаких автозапросов на сервер на каждый символ; сохранение в файл
 * выполняется по явной кнопке «Сохранить изменения». Внешний код может
 * применить пачку правок (например, предложения ФИО) через `ref.applyEdits`,
 * который пишет в файл одним запросом без участия локального draft.
 */
export const SheetGrid = forwardRef<SheetGridHandle, Props>(function SheetGrid(
  {
    sourceId,
    sheetName,
    headerRows,
    pageSize = 200,
    onMutated,
    suggestionByCell,
    onDirtyChange,
  },
  ref
) {
  const qc = useQueryClient();
  const [page, setPage] = useState(0);
  const [pending, setPending] = useState<Record<string, PendingEdit>>({});
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const startRow = page * pageSize + 1;

  const gridQuery = useQuery({
    queryKey: ["grid", sourceId, sheetName, startRow, pageSize],
    queryFn: () =>
      api.readSheetGrid(sourceId, sheetName, {
        startRow,
        startCol: 1,
        nRows: pageSize,
        nCols: 50,
      }),
    retry: false,
    staleTime: 0,
  });

  const fragment: SheetGridFragment | undefined = gridQuery.data;
  const totalRows = fragment?.total_rows ?? 0;
  const totalCols = fragment?.total_cols ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));
  const headerRowSet = useMemo(() => new Set(headerRows), [headerRows]);

  const dirtyCount = Object.keys(pending).length;

  useEffect(() => {
    onDirtyChange?.(dirtyCount);
  }, [dirtyCount, onDirtyChange]);

  const saveMut = useMutation({
    mutationFn: async (edits: CellEdit[]) =>
      api.applySheetGridEdits(sourceId, sheetName, edits),
    onSuccess: (_data, edits) => {
      setStatusMsg(
        edits.length === 1
          ? "Сохранена 1 ячейка."
          : `Сохранено ячеек: ${edits.length}.`
      );
      void qc.invalidateQueries({ queryKey: ["grid", sourceId, sheetName] });
      void qc.invalidateQueries({ queryKey: ["emptyRows", sourceId] });
      void qc.invalidateQueries({ queryKey: ["source", sourceId] });
      void qc.invalidateQueries({ queryKey: ["sources"] });
      onMutated?.();
    },
    onError: (err) =>
      setStatusMsg(
        err instanceof ApiError
          ? `Ошибка сохранения: ${err.message}`
          : "Не удалось сохранить правки."
      ),
  });

  const onCellChange = (row: number, col: number, raw: string) => {
    setPending((prev) => ({
      ...prev,
      [`${row}:${col}`]: { row, col, value: parseValue(raw) },
    }));
    setStatusMsg(null);
  };

  const flushNow = async () => {
    if (dirtyCount === 0) return;
    const edits = Object.values(pending);
    try {
      await saveMut.mutateAsync(edits);
      setPending({});
    } catch {
      // ошибка уже отображена через onError
    }
  };

  const discardLocal = () => {
    if (dirtyCount === 0) return;
    setPending({});
    setStatusMsg("Локальные правки отменены.");
  };

  const tryChangePage = (next: number) => {
    if (dirtyCount > 0) {
      const ok = window.confirm(
        `У вас ${dirtyCount} несохранённых правок. Перейти на другую страницу окна без сохранения?`
      );
      if (!ok) return;
      setPending({});
    }
    setPage(next);
  };

  // Сбрасываем локальный draft при смене листа.
  useEffect(() => {
    setPending({});
    setStatusMsg(null);
    setPage(0);
  }, [sheetName]);

  const pendingRef = useRef(pending);
  useEffect(() => {
    pendingRef.current = pending;
  }, [pending]);

  useImperativeHandle(
    ref,
    () => ({
      applyEdits: async (edits: CellEdit[]) => {
        if (edits.length === 0) return;
        await saveMut.mutateAsync(edits);
        // Из applyEdits локальный draft не трогаем — это другой источник правок.
      },
      discardLocal,
      dirtyCount: () => Object.keys(pendingRef.current).length,
    }),
    [saveMut]
  );

  if (gridQuery.isLoading) {
    return (
      <div className="px-4 py-8 text-center text-brand-700/70">
        Загрузка фрагмента листа...
      </div>
    );
  }
  if (gridQuery.isError) {
    return (
      <div className="px-4 py-8 text-center text-red-700">
        Не удалось загрузить лист «{sheetName}».
      </div>
    );
  }
  if (!fragment) return null;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3 text-xs text-brand-700/70">
        <span>
          Строк всего: <b className="text-brand-900">{totalRows}</b>, колонок:{" "}
          <b className="text-brand-900">{totalCols}</b>.
        </span>
        <span>
          Окно: {startRow}…{Math.min(startRow + pageSize - 1, totalRows)}.
        </span>
        <span className="ml-auto flex items-center gap-2">
          {dirtyCount > 0 && (
            <>
              <span className="rounded-md bg-emerald-100 px-2 py-0.5 text-emerald-900">
                Несохранённых правок: {dirtyCount}
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={discardLocal}
                disabled={saveMut.isPending}
                title="Откатить все локальные изменения окна и не записывать их в файл."
              >
                Отменить
              </Button>
            </>
          )}
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void flushNow()}
            disabled={dirtyCount === 0 || saveMut.isPending}
            title="Записать локальные изменения в Excel-файл одним пакетом."
          >
            {saveMut.isPending ? "Сохраняем..." : "Сохранить изменения"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={page === 0}
            onClick={() => tryChangePage(Math.max(0, page - 1))}
          >
            ← Предыдущая
          </Button>
          <span>
            {page + 1} / {totalPages}
          </span>
          <Button
            size="sm"
            variant="ghost"
            disabled={page + 1 >= totalPages}
            onClick={() => tryChangePage(page + 1)}
          >
            Следующая →
          </Button>
        </span>
      </div>

      {statusMsg && (
        <div className="text-xs text-brand-700/70">{statusMsg}</div>
      )}

      <div
        className="overflow-auto rounded-md border border-brand-200 bg-white"
        style={{ maxHeight: "60vh" }}
      >
        <div
          className="grid text-xs font-mono"
          style={
            {
              gridTemplateColumns: COL_WIDTHS,
              ["--cols" as string]: fragment.n_cols,
            } as React.CSSProperties
          }
        >
          <div className="sticky top-0 z-10 bg-brand-50 border-b border-brand-200 px-2 py-1.5 text-brand-700/70 text-right">
            #
          </div>
          {Array.from({ length: fragment.n_cols }).map((_, c) => (
            <div
              key={`h-${c}`}
              className="sticky top-0 z-10 bg-brand-50 border-b border-brand-200 px-2 py-1.5 text-brand-700/70"
            >
              {colLabel(c + 1)}
            </div>
          ))}

          {fragment.cells.map((row, rIdx) => {
            const absoluteRow = startRow + rIdx;
            const isHeader = headerRowSet.has(absoluteRow - 1);
            return (
              <React.Fragment key={`r-${absoluteRow}`}>
                <div
                  className={`sticky left-0 z-[1] border-b border-brand-100 px-2 py-1.5 text-right text-brand-700/70 ${
                    isHeader ? "bg-amber-50" : "bg-brand-50/60"
                  }`}
                  style={{ height: ROW_HEIGHT }}
                >
                  {absoluteRow}
                </div>
                {row.map((value, cIdx) => {
                  const col = cIdx + 1;
                  const key = `${absoluteRow}:${col}`;
                  const pendingEdit = pending[key];
                  const suggestion = suggestionByCell?.[key];
                  const display = pendingEdit
                    ? formatValue(pendingEdit.value)
                    : formatValue(value);
                  // Подсветка приоритет: локальная правка > предложение.
                  const cellClass = pendingEdit
                    ? "bg-emerald-50"
                    : suggestion
                    ? "bg-violet-50"
                    : "";
                  const titleParts: string[] = [];
                  if (suggestion) {
                    titleParts.push(`Предлагаем: ${suggestion.proposed}`);
                    if (suggestion.hint) titleParts.push(suggestion.hint);
                  }
                  return (
                    <input
                      key={key}
                      value={display}
                      onChange={(e) =>
                        onCellChange(absoluteRow, col, e.target.value)
                      }
                      readOnly={isHeader}
                      title={titleParts.join(" — ") || undefined}
                      className={`border-b border-r border-brand-100 px-2 py-1 text-brand-900 outline-none focus:bg-brand-50 ${
                        isHeader ? "bg-amber-50/60 font-semibold" : ""
                      } ${cellClass}`}
                      style={{ height: ROW_HEIGHT }}
                    />
                  );
                })}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
});

/** Excel-style column label: 1 → A, 27 → AA. */
function colLabel(n: number): string {
  let s = "";
  let x = n;
  while (x > 0) {
    const rem = (x - 1) % 26;
    s = String.fromCharCode(65 + rem) + s;
    x = Math.floor((x - 1) / 26);
  }
  return s;
}
