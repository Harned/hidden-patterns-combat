import React, { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type { CellEdit, GridCellValue, SheetGridFragment } from "@/api/types";
import { Button } from "@/components/ui";

interface Props {
  sourceId: number;
  sheetName: string;
  headerRows: number[];
  /** Сколько строк рендерим в окне; виртуализация — оконная пагинация. */
  pageSize?: number;
  onMutated?: () => void;
}

interface PendingEdit {
  row: number;
  col: number;
  value: GridCellValue;
}

const COL_WIDTHS = "minmax(64px, 80px) repeat(var(--cols), minmax(140px, 1fr))";
const ROW_HEIGHT = 36;
const SAVE_DEBOUNCE_MS = 600;

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

/** Виртуализированная Excel-подобная сетка: строки листа загружаются окном
 *  через `GET /grid`, правки буферизуются и сохраняются debounced PUT. */
export const SheetGrid: React.FC<Props> = ({
  sourceId,
  sheetName,
  headerRows,
  pageSize = 200,
  onMutated,
}) => {
  const qc = useQueryClient();
  const [page, setPage] = useState(0);
  const [pending, setPending] = useState<Record<string, PendingEdit>>({});
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const saveTimer = useRef<number | null>(null);

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

  const saveMut = useMutation({
    mutationFn: async (edits: CellEdit[]) =>
      api.applySheetGridEdits(sourceId, sheetName, edits),
    onSuccess: () => {
      setPending({});
      setStatusMsg("Правки сохранены.");
      void qc.invalidateQueries({ queryKey: ["grid", sourceId, sheetName] });
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

  useEffect(
    () => () => {
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
    },
    []
  );

  const scheduleSave = (next: Record<string, PendingEdit>) => {
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    if (Object.keys(next).length === 0) return;
    saveTimer.current = window.setTimeout(() => {
      saveMut.mutate(Object.values(next));
    }, SAVE_DEBOUNCE_MS);
  };

  const onCellChange = (row: number, col: number, raw: string) => {
    const next = { ...pending, [`${row}:${col}`]: { row, col, value: parseValue(raw) } };
    setPending(next);
    setStatusMsg("Изменения буферизованы, скоро сохраним...");
    scheduleSave(next);
  };

  const flushNow = () => {
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    if (Object.keys(pending).length === 0) return;
    saveMut.mutate(Object.values(pending));
  };

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
          {Object.keys(pending).length > 0 && (
            <Button size="sm" variant="secondary" onClick={flushNow}>
              Сохранить сейчас ({Object.keys(pending).length})
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
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
            onClick={() => setPage((p) => p + 1)}
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
                  const display = pendingEdit
                    ? formatValue(pendingEdit.value)
                    : formatValue(value);
                  return (
                    <input
                      key={key}
                      value={display}
                      onChange={(e) =>
                        onCellChange(absoluteRow, col, e.target.value)
                      }
                      onBlur={flushNow}
                      readOnly={isHeader}
                      className={`border-b border-r border-brand-100 px-2 py-1 text-brand-900 outline-none focus:bg-brand-50 ${
                        isHeader ? "bg-amber-50/60 font-semibold" : ""
                      } ${pendingEdit ? "bg-emerald-50" : ""}`}
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
};

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
