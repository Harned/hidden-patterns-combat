# Roadmap

## Done (current baseline)

1. Разделены режимы `research` и `inverse-diagnostic` без поломки старого CLI.
2. Добавлен строгий observation layer с config-driven mapping (`observation_mapping_v1.json`).
3. Добавлена каноническая таблица эпизодов (`1 row = 1 episode`) с traceability и train eligibility.
4. Реализован inverse diagnostic pipeline: observed sequence -> Viterbi -> posterior -> report.
5. Добавлена weak supervision логика (semantic init + post-hoc relabeling S1/S2/S3).
6. Добавлены unit/integration тесты для mapping/filtering/inverse e2e/backward compatibility.

## Next

1. Добавить валидацию observation mapping по версиям датасетов (v2/v3) и протокол миграции.
2. Добавить отдельные train/infer датасеты и стабильную offline-оценку качества inverse режима.
3. Ввести калибровку confidence и мониторинг доли `unknown` по спортсменам/весам.
4. Добавить semi-supervised режим с разметкой экспертных эпизодов.
5. Расширить LPR-отчет до сравнений по периодам и контекстным сценариям схватки.
