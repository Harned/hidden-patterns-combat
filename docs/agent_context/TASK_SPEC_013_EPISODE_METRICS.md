# TASK_SPEC_013_EPISODE_METRICS

## Текущая задача

Реализовать **метрики управления эпизодом** и описательную классификацию
стиля по порогам из YAML. Это **третий** приоритет магистерской поставки.

Метрики — описательное расширение, не диагноз и не замена матрицы
переходов. Они отображаются в индивидуальном HTML-отчёте
(`TASK_SPEC_011`) и используются в текстовой интерпретации.

## Зависимости

- `TASK_SPEC_011_INDIVIDUAL_MARKOV.md` (последовательность эпизодов,
  длительности, состояния).

## Предметные инварианты

- Никакого `fighter_style` в основном выводе.
- `classify_style` — описательная метка `endurance | speed_power | burnout`,
  получается из явных порогов в YAML, не из HMM-эмиссий и не из
  интуиции.
- Пороги — внешний YAML-конфиг `config/style_thresholds.yaml`,
  валидация при загрузке.

## Метрики

Для каждого спортсмена:

1. `episode_count` — число эпизодов в его поединках.
2. `episode_duration_stats`:
   - `mean`, `std`, `p25`, `p75` по `Время эпизода, с.`.
3. `action_density` — среднее число активаций (значений `1` или `2`)
   на эпизод, по всем признаковым столбцам, после `>2 → skip`.
4. `non_technical_share` — доля эпизодов, чьё состояние ≠
   `technical_action`.
5. `activity_evenness` — выравненность активности по эпизодам
   поединка; реализация по умолчанию: нормированная энтропия
   распределения `action_density` по эпизодам, в `[0, 1]`.
6. `classify_style(metrics) -> 'endurance' | 'speed_power' | 'burnout'`
   — пороговый классификатор. Точные правила задаются YAML,
   например (иллюстративно):
   ```yaml
   thresholds:
     endurance:
       min_episode_count: 8
       min_activity_evenness: 0.6
     speed_power:
       max_episode_count: 6
       min_action_density: 1.5
     burnout:
       min_action_density_first_half: 1.5
       max_action_density_second_half: 0.5
   ```
   Если ни одно правило не сработало — `classify_style` возвращает
   `unclassified` + warning `style.no_rule_matched` (без падения).

## Что должно появиться

1. **algo/hpc_algo/episode_metrics.py** —
   - `compute_episode_metrics(episodes, raw_features_df) -> EpisodeMetrics`;
   - `classify_style(metrics, thresholds) -> StyleLabel`.
2. **algo/hpc_algo/schema.py** — `EpisodeMetrics`, `StyleLabel`.
3. **config/style_thresholds.yaml** — пороги (плейсхолдер,
   точные значения уточняются эпизод-аналитиком).
4. **algo/tests/test_episode_metrics.py** —
   - метрики корректны на синтетике;
   - классификатор детерминирован по порогам;
   - неподходящий случай → `unclassified` + warning;
   - инвариант: метрики ничего не пишут в матрицу переходов
     (только читают эпизодный поток).

## Что НЕ делает этот шаг

- Не делает основной вывод вокруг стиля бойца.
- Не подменяет матрицу переходов.
- Не выводит шкалу координационной сложности 1–5.
- Не использует HMM.

## Definition of Done

1. Все метрики посчитаны и попадают в HTML-отчёт `TASK_SPEC_011`.
2. `classify_style` детерминирован, пороги читаются из YAML.
3. Случай «ни одно правило не сработало» возвращает
   `unclassified` + warning, не падает.
4. ≥ 1 pytest на `compute_episode_metrics` и ≥ 1 на
   `classify_style` зелёные.
5. README обновлён, ограничения зафиксированы (метрики —
   описательное расширение, не замена MC).
