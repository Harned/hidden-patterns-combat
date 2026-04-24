# TASK_SPEC_003_COLUMN_MAPPING

## Текущая задача

Добавить поддержку ручного и полуавтоматического сопоставления колонок
(`column mapping`) для Excel-источников, структура которых не распознаётся
автоматически (в т.ч. многострочные заголовки).

После выполнения задачи реальный файл `docs/Оценка СД содержание.xlsx`
должен анализироваться в статусе `baseline_only` с настоящими
распределениями по всем четырём предметным группам.

## Предметные инварианты (без изменений)

- `observations = ЗАП`.
- Скрытые состояния описывают соревновательную деятельность:
  `маневрирование -> КФВ -> ВУП -> ЗАП`.
- Никакой HMM-диагностики в этой задаче не добавляется. Mapping — это
  подготовка к полноценному diagnostic-режиму, а не он сам.
- Алгоритм остаётся независимым модулем; backend только сохраняет
  `mapping_config` и прокидывает его в `analyze_source`.
- При неподтверждённом или отсутствующем mapping алгоритм продолжает
  возвращать честный `audit_only` / `needs_column_mapping`, а не
  «додумывать» структуру.

## Что должно появиться

1. **algo**:
   - `ColumnMappingConfig` / `SheetMapping` в схеме.
   - Поддержка многострочных заголовков (`header_rows=[0,1,2]`) с
     детерминированным flatten.
   - `preflight_mapping(source_path, hints=None)` — возвращает
     предполагаемый `ColumnMappingConfig` на основании flatten-заголовков
     и эвристики.
   - `analyze_source(path, config=AnalyzeConfig(mapping=...))` — если
     mapping задан и листов покрыто достаточно, возвращает `baseline_only`
     с распределениями по всем 4 группам.
   - Расширенный `BaselineReport`:
     `hidden_group_value_counts: dict[HiddenGroup, dict[str, dict[value, count]]]`
     и `time_statistics: dict[str, TimeStats]`.
   - Обновлённые chart-ready JSON для каждой группы.
2. **backend**:
   - На уровне БД: у `Source` появляется `mapping_config` (JSON, nullable).
   - `POST /api/sources/{id}/preflight` — запускает `preflight_mapping`
     и возвращает предлагаемый config.
   - `GET /api/sources/{id}/mapping`.
   - `PUT /api/sources/{id}/mapping` — сохранение config (полная
     валидация на уровне pydantic).
   - `DELETE /api/sources/{id}/mapping` — сбросить mapping.
   - `POST /api/sources/{id}/analyze` — если есть сохранённый mapping,
     использует его.
3. **frontend**:
   - В AnalysisView — вкладки «Результат» / «Сопоставление колонок».
   - В вкладке mapping: запуск preflight, таблица «flatten-колонка → роль»
     (dropdown на каждую колонку), настройка `header_rows` на лист, кнопки
     «Сохранить» и «Сохранить и запустить анализ», сброс.
   - Русскоязычный UI.
4. **тесты**:
   - algo: multi-row header → flatten, preflight auto-detects хотя бы
     один `zap`-кандидат на реальном-подобном listsamples; mapping →
     настоящий baseline для всех 4 групп; инвариант «HMM-поля отсутствуют»
     сохраняется.
   - backend: preflight, save/get/delete mapping, изоляция между
     пользователями, анализ с сохранённым mapping возвращает
     `baseline_only`.
5. **e2e на реальном Excel**:
   - register → upload → preflight → сохранение авто-mapping →
     analyze → `status = baseline_only` с непустыми распределениями по
     `ЗАП`, `маневрирование`, `КФВ`, `ВУП`.

## Формат `ColumnMappingConfig`

Pydantic-схема (упрощённо):

```python
class SheetMapping(BaseModel):
    header_rows: list[int] = [0]
    data_start_row: int | None = None  # для явной отладки, по умолчанию = max(header_rows)+1
    roles: dict[HiddenGroup, list[str]] = {}
    # ключ — доменная роль; значение — список flatten-имён колонок на листе

class ColumnMappingConfig(BaseModel):
    sheets: dict[str, SheetMapping] = {}   # ключ — имя листа
    version: str = "1"
```

Flatten-имя колонки — строка вида ``"level0 | level1 | level2"`` с
очисткой от `Unnamed: N` и пустых уровней.

## Правила и ограничения

- `mapping_config` — ответственность пользователя. Алгоритм доверяет ему.
- Если mapping ссылается на колонку, которой нет на листе, возвращается
  `warning` с `code=mapping.unknown_column`, а значение игнорируется.
- Если после применения mapping ни одна `zap`-колонка не дала валидных
  значений — статус `needs_column_mapping` и соответствующие warnings
  (не `baseline_only`).
- Детекция кандидатов (`detection.py`) переиспользуется для preflight,
  но над flatten-заголовками.
- HMM-поля в `AnalysisResult` по-прежнему отсутствуют; `status == hmm_ready`
  недостижим в этой задаче.

## Definition of Done

Выполнение всех:

1. Тесты algo и backend зелёные, ruff clean, tsc clean.
2. На реальном `docs/Оценка СД содержание.xlsx` полный цикл
   `preflight → save mapping → analyze` возвращает `status = baseline_only`
   с непустыми распределениями для `ЗАП` и хотя бы одной из групп
   `маневрирование / КФВ / ВУП`.
3. UI позволяет пользователю запустить preflight, увидеть и изменить
   mapping, сохранить его и перезапустить анализ.
4. Инварианты сохранены: observations = ЗАП, `fighter_style` не
   используется, HMM-поля в результате отсутствуют.
5. README обновлены, ограничения зафиксированы.
