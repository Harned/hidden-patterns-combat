# TASK_SPEC_004_HMM

> **Status: under review — depends on resolution of Conflict C1**
>
> После встречи 02.05.2026 основной поставкой магистерской работы стала
> 5-state наблюдаемая Marков-цепь по эпизодам
> (`TASK_SPEC_011_INDIVIDUAL_MARKOV.md`). Существующая HMM-постановка
> (этот файл и `TASK_SPEC_005`) методологически не отвергнута, но
> отложена. Все инварианты ниже остаются в силе, если по итогам
> Уровня 1 будет решено вернуться к HMM. До этого решения новые
> работы по HMM не начинать.

## Текущая задача

Добавить диагностический HMM-режим processing module. Задача — по
подтверждённому column mapping и достаточно плотным данным восстановить
наиболее вероятную **скрытую причинную траекторию**, которая приводит к
наблюдаемым ЗАП-событиям.

HMM включается **только** при выполнении всех guard'ов качества
(см. ниже). Если хоть один guard не прошёл — `status` остаётся
`baseline_only`, HMM не запускается, пользователь получает честный
warning с указанием, чего не хватает.

## Предметные инварианты

(см. `DOMAIN_SPEC.md`, не переговариваемы)

* observations = `ЗАП` (включая категорические метки и per-channel
  бинарные/count-события, см. `TASK_SPEC_003_1`);
* hidden states описывают соревновательную деятельность:
  `маневрирование -> КФВ -> ВУП -> ЗАП`;
* в MVP используется **3-state** модель: `S1=маневрирование`,
  `S2=КФВ`, `S3=ВУП`. Детализированная (7-state) — отдельный TASK позже;
* имена состояний — строго предметные; никакого `fighter_style`.

## Схема наблюдений

Из `BaselineReport` уже есть:

* categorical ЗАП — value_counts;
* binary/count ЗАП — per-channel events (`zap_events_by_channel`).

В HMM-ветке **observation** — событие в эпизоде. Одна «последовательность»
= один эпизод. Алфавит наблюдений `V` состоит из:

1. channel_label'ов binary/count-колонок (`Удержание`, `На руку`,
   `На ногу`, `ЗАП-Р`, …);
2. уникальных значений categorical ЗАП-колонок;
3. специального токена `_noop_` (эпизод без ЗАП-события).

Для каждого эпизода строится последовательность токенов в порядке
возникновения. Если в эпизоде не зарегистрировано ни одного события —
используется однотокенная последовательность `[_noop_]` (это сохраняет
обучающий сигнал «большинство эпизодов завершается без ЗАП»).

## Доменные ограничения на параметры HMM

* **состояний 3**, имена фиксированные (`маневрирование`, `КФВ`, `ВУП`);
* начальное распределение `π` инициализируется предметно: `маневрирование`
  ≥ 0.8, остальные малы;
* матрица переходов `A` инициализируется как трёхдиагональная «движение
  по цепочке вперёд» (`маневрирование -> КФВ -> ВУП -> ЗАП-наблюдение`),
  чтобы не получить состояния-синонимы после обучения;
* обучение — Baum-Welch (EM) с **фиксированным seed**, ограниченным
  числом итераций и явной проверкой сходимости;
* после обучения проводится **sanity check**: диагональные/под-диагональные
  элементы `A` должны преобладать (смысл — состояния всё ещё лежат на
  цепочке). Если нет — `status` НЕ повышается до `hmm_ready`.

## Guard'ы качества данных

HMM запускается только если **все** выполнено:

1. `applied_mapping` не пуст; mapping содержит роль `ЗАП` хотя бы для
   одного листа.
2. Суммарное число эпизодов (`sum(episodes_per_sheet.values())`) ≥
   `MIN_EPISODES` (по умолчанию 30).
3. Суммарное число ЗАП-событий (`sum(zap_events_by_channel.values()) +
   sum(zap_value_counts per column)`) ≥ `MIN_ZAP_EVENTS` (по умолчанию 20).
4. В алфавите наблюдений ≥ 2 различных токенов (иначе эмиссия тривиальна).
5. После обучения sanity check матрицы переходов пройден.

Если любое условие нарушено, HMM не запускается, возвращается
`status=baseline_only` + warning `hmm.guards_failed` с указанием
конкретного нарушенного guard'а.

## Что должно появиться

1. **algo/hpc_algo/hmm.py** — сборка последовательностей, HMM-обучение,
   Viterbi, gamma, guards, sanity check. Реализация через `hmmlearn`
   (`CategoricalHMM`) с явным `random_state`. Если `hmmlearn`
   недоступен — guard блокирует HMM и возвращается warning
   `hmm.dependency_missing`.
2. **algo/hpc_algo/schema.py** — новые типы:
   * `HMMParameters` (A, B, pi, n_states, n_observations, labels);
   * `HMMTrajectory` (per-episode Viterbi path + log-likelihood);
   * `HMMResult` (parameters, trajectories по листам, gamma summary,
     transition heatmap, interpretation, seed, n_iter);
   * `AnalysisResult.hmm: HMMResult | None` — новое поле.
3. **algo/hpc_algo/api.py** — `AnalyzeConfig.enable_hmm` (по умолчанию
   True), `AnalyzeConfig.hmm_seed`, `AnalyzeConfig.hmm_min_*` (параметры
   guard'ов). `status = hmm_ready` выставляется только при успешном
   обучении и проверках; `applied_mapping` и baseline остаются в результате.
4. **algo/hpc_algo/baseline.py / charts** — chart-ready JSON для
   transition matrix (heatmap) и state distribution (bar).
5. **algo tests**:
   * sequence builder корректно строит последовательности по эпизодам;
   * guard'ы блокируют HMM на тонких данных;
   * HMM работает на synthetic fixture с ≥30 эпизодами и 20 ЗАП-событиями;
   * инварианты: 3 состояния с доменными именами; `observations = ЗАП`;
     HMM-поля присутствуют ТОЛЬКО при `status == hmm_ready`;
   * воспроизводимость: один и тот же seed → одинаковый `log_likelihood`.
6. **backend**: HMM-параметры (seed, enable) пробрасываются при analyze;
   e2e-тест на плотной фикстуре доходит до `hmm_ready` и возвращает
   HMMResult в payload.
7. **frontend**: отдельные компоненты `HMMOverview`, `TransitionHeatmap`,
   `ViterbiSequences`; рендерятся только при `status == hmm_ready`;
   при `baseline_only` явно показывается, какой guard не прошёл.

## Что НЕ делает этот шаг

* Не реализует 7-state детализированную модель (маневры / захваты /
  хваты / обхваты / прихваты / упоры / ВУП) — это отдельный TASK.
* Не добавляет интерактивный выбор числа состояний / алгоритма.
* Не меняет формат `ColumnMappingConfig`.
* Не запускает HMM на thin-data реального Excel без предварительного
  увеличения объёма данных — если guard'ы не проходят, UI честно
  показывает, почему.

## Definition of Done

1. На synthetic fixture с достаточным числом эпизодов и ЗАП-событий
   полный цикл `analyze_source(..., enable_hmm=True)` возвращает
   `status = hmm_ready` с заполненными `hmm.parameters` / `hmm.trajectories`.
2. На synthetic thin-data fixture (2 эпизода) guard блокирует HMM,
   возвращается `baseline_only` + warning `hmm.guards_failed`.
3. Одинаковый seed даёт одинаковый `log_likelihood` (инвариант
   воспроизводимости).
4. Состояния именованы строго `маневрирование` / `КФВ` / `ВУП`.
5. `fighter_style` нигде не используется.
6. Тесты algo + backend зелёные; ruff/tsc clean.
7. README и `docs/agent_context/README.md` обновлены.
