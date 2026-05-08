# TASK_SPEC_012_AGGREGATE_DIVERGENCE

## Текущая задача

Построить **агрегатную** наблюдаемую 5-state Marков-цепь по призёрам
1–3 в каждой из 10 весовых категорий ЧР-2025, реализовать функцию
дивергенции `divergence(individual, aggregate)` и ранжирование
финалистов внутри категории.

Это **второй** приоритет магистерской поставки.
Рабочая гипотеза — «средняя формула победы среди призёров» —
**не отвергнута**; решение по гипотезе принимается отдельно
после анализа результатов.

## Зависимости

- `TASK_SPEC_011_INDIVIDUAL_MARKOV.md` (индивидуальные модели и
  `state_groups.yaml`).
- `DOMAIN_SPEC.md` § «Уровни моделирования».

## Предметные инварианты

- Агрегатная модель строится **только** по призёрам 1–3 каждой
  весовой категории. Усреднение по всем участникам категории —
  запрещено (`AGENT_RULES.md` § Запрещено).
- 10 категорий → 10 агрегатных моделей.
- Алфавит состояний — тот же, что в `TASK_SPEC_011`.
- Никакого `fighter_style` в основном выводе.

## Источник данных

Тот же: `docs/Оценка СД содержание.xlsx`, лист `Общее`. Список
призёров определяется по столбцам `Весовая категория` (или
эквивалент) и `Место` (или метка финалиста), которые читаются из
`base columns` (`DATA_SPEC.md`).

## Артефакты

Для каждой весовой категории `wc`:

1. `aggregate_sequence_wc` — конкатенация эпизодных
   последовательностей всех призёров категории (с учётом границ
   спортсменов, чтобы не индуцировать ложные переходы между
   разными атлетами; реализация — построчное накопление
   `transition_counts`, без склейки последовательностей).
2. `A_aggregate_wc` — матрица 5×5 с тем же инвариантом сумм строк.
3. `π_aggregate_wc` — стационарное распределение.
4. `divergence(individual, aggregate)`:
   - `kl_transitions = Σ_i π_aggregate[i] · KL(A_individual[i,:] ‖ A_aggregate[i,:])`
     (среднее KL-row, взвешенное по агрегатной π);
   - `l1_stationary  = ‖π_individual − π_aggregate‖₁`;
   - оба — неотрицательные; чем меньше, тем ближе индивидуал к
     «средней формуле победы среди призёров».
5. `ranking_wc` — отсортированный по возрастанию композитной метрики
   `α · l1_stationary + (1 − α) · kl_transitions` список финалистов
   категории с местами 1–3 и (если есть) дополнительных финалистов.
   `α` — параметр конфигурации, по умолчанию `0.5`.
6. `report_wc.html` — агрегат тепловой карты, π, ранжирование,
   warnings.

## Что должно появиться

1. **algo/hpc_algo/markov_aggregate.py** —
   - `fit_aggregate_markov(individual_results: list, weight_class) -> MarkovAggregateResult`;
   - использует `transition_counts` индивидуалов (не склеивая
     последовательности) для построения агрегатной `A`.
2. **algo/hpc_algo/compare.py** —
   - `divergence(individual: MarkovIndividualResult, aggregate: MarkovAggregateResult) -> Divergence`;
   - `rank_within_class(category_individuals, aggregate, alpha=0.5) -> list`.
3. **algo/hpc_algo/schema.py** —
   - `MarkovAggregateResult(weight_class, members, transition_matrix,
     stationary, members_count, warnings)`;
   - `Divergence(kl_transitions, l1_stationary, composite, alpha)`.
4. **scripts/build_aggregate_models.py** — CLI: для каждой
   весовой категории строит агрегат, ранжирует финалистов,
   генерирует `reports/aggregate/<weight_class>.html`.
5. **Makefile** — таргет `aggregate-models` (зависит от
   `individual-models`).
6. **algo/tests/test_markov_aggregate.py** + `test_compare.py` —
   - агрегат строится только по списку призёров (тест: чужой
     спортсмен в категории не должен попадать);
   - дивергенция: `divergence(x, x) == 0`, симметрия для L1,
     **асимметрия** для KL (это намеренно);
   - ранжирование стабильно при перестановках входа.

## Что НЕ делает этот шаг

- Не делает финальный вывод по гипотезе «формулы победы»
  (отдельная сессия эксперта).
- Не использует HMM.
- Не строит ансамбли по нескольким турнирам.

## Definition of Done

1. `make aggregate-models` за один прогон выдаёт 10 HTML-отчётов
   в `reports/aggregate/`.
2. Каждая `A_aggregate_wc`: сумма строк = 1 ± 1e-6 (pytest).
3. `divergence(x, x) == 0` (pytest).
4. Ранжирование возвращает финалистов категории по возрастанию
   композитной метрики; стабильно к порядку входа (pytest).
5. Никаких HMM-полей в результате.
6. README обновлён.
