/**
 * Русскоязычные подписи для warning-кодов, приходящих от algo/backend.
 * Если кода нет в карте — UI показывает сообщение, пришедшее с бэкенда
 * (там оно уже на русском, но здесь мы можем переопределить для более
 * сжатой формулировки в сводках).
 */

export interface WarningSpec {
  short: string;
  tone: "info" | "warning" | "error";
}

export const WARNING_SPECS: Record<string, WarningSpec> = {
  "audit.possible_multirow_header": {
    short: "Похоже на многострочный заголовок",
    tone: "warning",
  },
  "audit.many_missing": { short: "Много пропусков", tone: "warning" },
  "audit.empty_file": { short: "Файл пустой", tone: "error" },
  "audit.empty_sheet": { short: "Пустой лист", tone: "info" },
  "audit.suspicious": { short: "Подозрительные данные", tone: "info" },
  "detection.no_strong_zap": {
    short: "Нет уверенных ЗАП-колонок",
    tone: "warning",
  },
  "detection.weak_candidates": {
    short: "Слабые кандидаты колонок",
    tone: "info",
  },
  "detection.missing_required_groups": {
    short: "Не распознаны обязательные группы",
    tone: "warning",
  },
  "status.needs_column_mapping": {
    short: "Требуется column mapping",
    tone: "warning",
  },
  "mapping.unknown_column": {
    short: "Колонки mapping не найдены",
    tone: "warning",
  },
  "mapping.groups_without_data": {
    short: "Группы без данных",
    tone: "warning",
  },
  "mapping.empty_config": { short: "Пустой mapping", tone: "warning" },
  "mapping.no_zap_values": {
    short: "Нет валидных ЗАП-значений",
    tone: "warning",
  },
  "hmm.disabled": { short: "HMM отключена", tone: "info" },
  "hmm.guards_failed": { short: "HMM: guard'ы не пройдены", tone: "warning" },
  "hmm.fit_failed": {
    short: "HMM: обучение не удалось",
    tone: "warning",
  },
  "hmm.sanity_failed": { short: "HMM: sanity-check не пройден", tone: "warning" },
  "loading.excel_error": { short: "Ошибка чтения Excel", tone: "error" },
  "loading.unexpected_error": {
    short: "Неожиданная ошибка чтения",
    tone: "error",
  },
};

export function translateWarning(code: string): WarningSpec | null {
  return WARNING_SPECS[code] ?? null;
}
