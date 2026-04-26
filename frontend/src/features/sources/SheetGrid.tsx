import React, {
  forwardRef,
  memo,
  useCallback,
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

interface ActiveCell {
  row: number;
  col: number;
  isHeader: boolean;
  key: string;
  initialValue: string;
}

type RegisterCellRef = (key: string, el: HTMLInputElement | null) => void;

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

interface GridCellProps {
  row: number;
  col: number;
  initial: string;
  pendingValue: string | undefined;
  isHeader: boolean;
  cellClass: string;
  title: string | undefined;
  isActive: boolean;
  onCommit: (row: number, col: number, raw: string) => void;
  onFocusCell: (
    row: number,
    col: number,
    raw: string,
    isHeader: boolean
  ) => void;
  onTypeMirror: (row: number, col: number, raw: string) => void;
  registerCellRef: RegisterCellRef;
}

/**
 * Ячейка-инпут со своим неуправляемым DOM-значением. Парент **не**
 * перерисовывается на каждый ввод символа: браузер обновляет инпут сам,
 * а зеркалирование в строку формул выполняется через DOM-ref.
 *
 * Коммит в `pending` происходит на `blur` или явном применении.
 */
const GridCell = memo(function GridCell(props: GridCellProps) {
  const {
    row,
    col,
    initial,
    pendingValue,
    isHeader,
    cellClass,
    title,
    isActive,
    onCommit,
    onFocusCell,
    onTypeMirror,
    registerCellRef,
  } = props;

  const inputRef = useRef<HTMLInputElement | null>(null);
  const focusedRef = useRef(false);
  const externalValue = pendingValue ?? initial;

  // Внешняя ресинхронизация (refetch фрагмента, applyEdits): обновляем DOM
  // только если ячейка не в фокусе — иначе мы перебили бы ввод пользователя.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    if (focusedRef.current) return;
    if (el.value !== externalValue) {
      el.value = externalValue;
    }
  }, [externalValue]);

  const setRef = useCallback(
    (el: HTMLInputElement | null) => {
      inputRef.current = el;
      const k = `${row}:${col}`;
      registerCellRef(k, el);
    },
    [row, col, registerCellRef]
  );

  return (
    <input
      ref={setRef}
      defaultValue={externalValue}
      readOnly={isHeader}
      title={title}
      onFocus={(e) => {
        focusedRef.current = true;
        onFocusCell(row, col, e.currentTarget.value, isHeader);
      }}
      onInput={(e) =>
        onTypeMirror(row, col, (e.currentTarget as HTMLInputElement).value)
      }
      onBlur={(e) => {
        focusedRef.current = false;
        onCommit(row, col, e.currentTarget.value);
      }}
      className={`border-b border-r border-brand-100 px-2 py-1 text-brand-900 outline-none focus:bg-brand-50 ${
        isHeader ? "bg-amber-50/60 font-semibold" : ""
      } ${cellClass} ${isActive ? "ring-2 ring-inset ring-brand-300" : ""}`}
      style={{ height: ROW_HEIGHT }}
    />
  );
});

/**
 * Виртуализированная Excel-подобная сетка. Ввод ведётся **локально** —
 * никаких автозапросов на сервер на каждый символ; сохранение в файл
 * выполняется по явной кнопке «Сохранить изменения». Внешний код может
 * применить пачку правок (например, предложения ФИО) через `ref.applyEdits`,
 * который пишет в файл одним запросом без участия локального draft.
 *
 * Перфоманс: ячейки — uncontrolled inputs в `React.memo`; типинг идёт по
 * браузерному DOM-пути, парент перерисовывается только на focus/blur.
 * Над сеткой — «строка формул» с адресом и полным текстом активной ячейки.
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
  const [active, setActive] = useState<ActiveCell | null>(null);

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

  const pendingRef = useRef(pending);
  useEffect(() => {
    pendingRef.current = pending;
  }, [pending]);

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

  // -------------------- активная ячейка + строка формул --------------------

  const barRef = useRef<HTMLInputElement | null>(null);
  const cellRefMap = useRef(new Map<string, HTMLInputElement>());
  const activeKeyRef = useRef<string | null>(null);

  const registerCellRef = useCallback<RegisterCellRef>((key, el) => {
    if (el) cellRefMap.current.set(key, el);
    else cellRefMap.current.delete(key);
  }, []);

  const onFocusCell = useCallback(
    (row: number, col: number, raw: string, isHeader: boolean) => {
      const k = `${row}:${col}`;
      // Зеркалим текущее значение ячейки в строку формул на DOM-уровне
      // (без setState — никаких ререндеров на каждый focus-keystroke).
      if (barRef.current) barRef.current.value = raw;
      if (activeKeyRef.current !== k) {
        activeKeyRef.current = k;
        setActive({ row, col, isHeader, key: k, initialValue: raw });
      }
    },
    []
  );

  const onTypeMirror = useCallback(
    (row: number, col: number, raw: string) => {
      if (
        activeKeyRef.current === `${row}:${col}` &&
        barRef.current &&
        document.activeElement !== barRef.current
      ) {
        barRef.current.value = raw;
      }
    },
    []
  );

  const onCommitCell = useCallback(
    (row: number, col: number, raw: string) => {
      const k = `${row}:${col}`;
      const value = parseValue(raw);
      setPending((prev) => ({ ...prev, [k]: { row, col, value } }));
      setStatusMsg(null);
    },
    []
  );

  const onBarInput = useCallback((raw: string) => {
    const ac = activeKeyRef.current;
    if (!ac) return;
    const cell = cellRefMap.current.get(ac);
    if (cell && document.activeElement !== cell) {
      cell.value = raw;
    }
  }, []);

  const onBarBlur = useCallback(
    (raw: string) => {
      const ac = active;
      if (!ac || ac.isHeader) return;
      // Перед коммитом синхронизируем DOM ячейки на случай, если
      // пользователь редактировал только в строке формул.
      const cell = cellRefMap.current.get(ac.key);
      if (cell && cell.value !== raw) cell.value = raw;
      onCommitCell(ac.row, ac.col, raw);
    },
    [active, onCommitCell]
  );

  // -------------------- save / discard / pagination --------------------

  const flushNow = useCallback(async () => {
    // Принудительно «выпустить» текущий ввод в pending (blur активного input).
    const ae =
      typeof document !== "undefined"
        ? (document.activeElement as HTMLElement | null)
        : null;
    if (ae && typeof ae.blur === "function") ae.blur();
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => resolve())
    );
    const edits = Object.values(pendingRef.current);
    if (edits.length === 0) return;
    try {
      await saveMut.mutateAsync(edits);
      setPending({});
    } catch {
      // ошибка уже отображена через onError
    }
  }, [saveMut]);

  const discardLocal = useCallback(() => {
    if (Object.keys(pendingRef.current).length === 0) return;
    setPending({});
    setStatusMsg("Локальные правки отменены.");
  }, []);

  const tryChangePage = (next: number) => {
    if (Object.keys(pendingRef.current).length > 0) {
      const ok = window.confirm(
        `У вас ${
          Object.keys(pendingRef.current).length
        } несохранённых правок. Перейти на другую страницу окна без сохранения?`
      );
      if (!ok) return;
      setPending({});
    }
    setPage(next);
    activeKeyRef.current = null;
    setActive(null);
  };

  // Сбрасываем локальный draft при смене листа.
  useEffect(() => {
    setPending({});
    setStatusMsg(null);
    setPage(0);
    activeKeyRef.current = null;
    setActive(null);
  }, [sheetName]);

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
    [saveMut, discardLocal]
  );

  // Заранее формируем отформатированные строки pending — стабильные по
  // значению для непротронутых ячеек, чтобы memo пропускал их.
  const pendingStrByKey = useMemo(() => {
    const m: Record<string, string> = {};
    for (const k in pending) m[k] = formatValue(pending[k].value);
    return m;
  }, [pending]);

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
            disabled={saveMut.isPending}
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

      {/* Строка формул как в Excel: адрес активной ячейки + полное значение. */}
      <div
        className="flex items-stretch gap-2 rounded-md border border-brand-200 bg-white px-2 py-1 text-xs"
        title="Полное значение активной ячейки. Правки синхронизируются с сеткой."
      >
        <span className="self-center min-w-[72px] rounded bg-brand-50 px-2 py-1 text-center font-mono text-brand-900">
          {active ? `${colLabel(active.col)}${active.row}` : "—"}
        </span>
        <span className="self-center hidden text-brand-700/60 sm:inline">
          {active
            ? active.isHeader
              ? `Строка ${active.row} (заголовок, только просмотр)`
              : `Строка ${active.row}`
            : "Кликните по ячейке, чтобы редактировать полное значение"}
        </span>
        <input
          ref={barRef}
          key={active?.key ?? "empty"}
          defaultValue={active?.initialValue ?? ""}
          placeholder={active ? "" : "Полное значение появится здесь"}
          readOnly={!active || active.isHeader}
          disabled={!active}
          onInput={(e) =>
            onBarInput((e.currentTarget as HTMLInputElement).value)
          }
          onBlur={(e) => onBarBlur(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === "Escape") {
              (e.currentTarget as HTMLInputElement).blur();
            }
          }}
          className="flex-1 rounded border border-brand-100 bg-white px-2 py-1 font-mono text-brand-900 outline-none focus:border-brand-300 disabled:bg-brand-50/60"
        />
      </div>

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
                  const k = `${absoluteRow}:${col}`;
                  const pendingStr = pendingStrByKey[k];
                  const initialStr = formatValue(value);
                  const suggestion = suggestionByCell?.[k];
                  const cellClass =
                    pendingStr !== undefined
                      ? "bg-emerald-50"
                      : suggestion
                      ? suggestion.kind === "header-merge-fill"
                        ? "bg-amber-50"
                        : "bg-violet-50"
                      : "";
                  const titleParts: string[] = [];
                  if (suggestion) {
                    titleParts.push(`Предлагаем: ${suggestion.proposed}`);
                    if (suggestion.hint) titleParts.push(suggestion.hint);
                  }
                  const title = titleParts.join(" — ") || undefined;
                  const isActive = active?.key === k;
                  return (
                    <GridCell
                      key={k}
                      row={absoluteRow}
                      col={col}
                      initial={initialStr}
                      pendingValue={pendingStr}
                      isHeader={isHeader}
                      cellClass={cellClass}
                      title={title}
                      isActive={isActive}
                      onCommit={onCommitCell}
                      onFocusCell={onFocusCell}
                      onTypeMirror={onTypeMirror}
                      registerCellRef={registerCellRef}
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
