# TASK_SPEC_011_INDIVIDUAL_MARKOV

## Текущая задача

Построить **индивидуальную наблюдаемую 5-state Marков-цепь** для каждого
финалиста Чемпионата России по самбо 2025 (10 весовых категорий ×
места 1–3, всего ~30 спортсменов). Это **основной** диагностический
выход магистерской работы.

После выполнения задачи команда `make individual-models` за один прогон
формирует ~30 индивидуальных HTML-отчётов из единственного файла
`docs/Оценка СД содержание.xlsx` без обращения к HMM.

## Предметные инварианты

(см. `DOMAIN_SPEC.md` § «Уровни моделирования»)

- Состояния — пять, из набора `manoeuvring / grip / off_balance /
  technical_action / pause`. Имена в коде — строго английские
  идентификаторы; русские подписи — только в YAML-маппинге и в шаблоне
  отчёта.
- Режим `single` — по умолчанию. Приоритет:
  `technical_action > off_balance > grip > manoeuvring > pause`.
- Режим `multi` — по флагу. Эпизод раскрывается в последовательность
  непустых групп в порядке `manoeuvring → grip → off_balance →
  technical_action`.
- Никакого `fighter_style` в основном выводе.
- HMM не используется (см. `TASK_SPEC_004/005` — under review).

## Источник данных

- `docs/Оценка СД содержание.xlsx`, лист `Общее`.
- Препроцессинг 3-уровневой шапки — **переиспользовать** существующий
  пайплайн (`algo/hpc_algo/mapping.py` flatten).
- Эпизод записан **двумя строками**, по одной на участника.
  Связка по `№ эпизода` + `Время эпизода, с.` + `Время паузы, с.`.
- Маркер конца поединка: пустое `Время паузы` у последнего эпизода
  поединка + одна-две полностью пустых строки.
- Кодировка признаковых столбцов: `1` — действие, `2` — дважды,
  пусто/`0` — нет. Защита `>2 → log&skip` (warning
  `data.value_out_of_range`, без падения).

## Группы состояний

Маппинг «состояние → список flatten-колонок» хранится в **внешнем YAML**:
`config/state_groups.yaml`. Группы описаны в `DATA_SPEC.md` § «Группы
признаковых столбцов».

`pause` не имеет колонок: попадает, если эпизод после применения
`>2 → log&skip` оказался полностью пуст или это строка-разделитель
поединков.

## Артефакты

Для каждого спортсмена:

1. `episodes_df` — упорядоченная по времени последовательность эпизодов
   с одной из 5 меток состояния (или последовательностью, в `multi`).
2. `transition_matrix A` — матрица 5×5, **сумма по строке = 1 ± 1e-6**
   (явный test). Если состояние не наблюдалось как «откуда» —
   соответствующая строка заполняется равномерно с warning
   (`mc.unobserved_row`).
3. `stationary_distribution π` — собственный вектор `A^T` с
   `eigenvalue == 1`. Если матрица периодическая или не эргодична —
   warning `mc.non_ergodic`, π считается через time-averaging
   (`(1/T) Σ_{t} state_t`) как fallback.
4. `episode_metrics` — см. `TASK_SPEC_013` (входит в тот же отчёт).
5. `interpretation_text` — детерминированный шаблонный текст, без LLM:
   - доминирующее состояние (по π);
   - наиболее вероятный переход из доминирующего;
   - доля `technical_action` и `pause`;
   - стиль управления эпизодом (`endurance / speed_power / burnout`)
     по `TASK_SPEC_013`, **только как описание**.
6. `report.html` — самодостаточный (inline CSS, без CDN), включающий:
   тепловую карту `A`, бар-чарт `π`, таблицу метрик, блок интерпретации,
   список warnings.

## Что должно появиться

1. **algo/hpc_algo/markov_individual.py** —
   - `build_episode_sequence(df, state_groups, mode='single')`;
   - `fit_individual_markov(sequence) -> MarkovIndividualResult`;
   - возвращает `A`, `π`, `state_visit_counts`, `transition_counts`,
     `warnings`.
2. **algo/hpc_algo/state_groups.py** — загрузчик YAML и валидация
   (все колонки YAML действительно встречаются после flatten;
   иначе warning).
3. **algo/hpc_algo/episode_split.py** — разбиение листа `Общее` на
   поединки и эпизоды по правилу 2 строк + маркера конца поединка.
4. **algo/hpc_algo/schema.py** —
   - `EpisodeRecord(athlete, weight_class, place, bout_id,
     episode_idx, episode_duration, pause_duration, score, state)`;
   - `MarkovIndividualResult(states, transition_matrix, stationary,
     visit_counts, warnings, mode)`.
5. **scripts/build_individual_models.py** — CLI: читает Excel,
   фильтрует по списку финалистов, генерирует отчёты в
   `reports/individual/<athlete>.html`.
6. **Makefile** — таргет `individual-models` запускает скрипт за один
   прогон.
7. **algo/tests/test_markov_individual.py** — минимум:
   - синтетический эпизодный поток → корректные `A`, `π`;
   - инвариант сумм строк `1 ± 1e-6`;
   - режим `multi` корректно раскрывает эпизод;
   - `>2` в признаковом столбце → warning, пайплайн не падает;
   - неэргодичная цепочка → fallback на time-averaging + warning.

## Конфиг

`config/state_groups.yaml` (формат — рекомендация):

```yaml
version: 1
states:
  manoeuvring:
    columns: ["ПС | направление 1", "ПС | направление 2", "..."]
  grip:
    columns: ["Захваты | О", "Захваты | ОО", "..."]
  off_balance:
    columns: ["ВУП-Р", "ВУП-Т", "..."]
  technical_action:
    columns: ["Броски | Руками", "Удержание", "Болевой | На руку", "..."]
mode: single  # или multi
priority: ["technical_action", "off_balance", "grip", "manoeuvring", "pause"]
```

## Что НЕ делает этот шаг

- Не добавляет HMM (см. `TASK_SPEC_004/005`).
- Не строит агрегатную модель (см. `TASK_SPEC_012`).
- Не публикует шкалу координационной сложности 1–5.
- Не делает финальный вывод по гипотезе «формулы победы».
- Не меняет существующий пайплайн flatten-шапки.

## Definition of Done

1. `make individual-models` за один прогон выдаёт ~30 HTML-отчётов
   в `reports/individual/`.
2. Для каждой матрицы `A` сумма строк = 1 ± 1e-6 (отдельный pytest).
3. ≥ 5 pytest-тестов на `markov_individual` + `episode_split` зелёные.
4. README обновлён: цель, источник, команды запуска, ограничения.
5. Никаких русских строк в логике состояний; русские подписи —
   только в YAML и в шаблоне HTML.
6. Защита `>2 → log&skip` покрыта warning'ом и тестом.
7. Никаких HMM-полей в результате.
