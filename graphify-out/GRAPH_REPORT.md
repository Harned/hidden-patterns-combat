# Graph Report - hidden-patterns-combat  (2026-05-09)

## Corpus Check
- 151 files · ~79,044 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1383 nodes · 2305 edges · 119 communities (109 shown, 10 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 456 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8e1f82e5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 104|Community 104]]

## God Nodes (most connected - your core abstractions)
1. `analyze_source()` - 55 edges
2. `AnalyzeConfig` - 52 edges
3. `preflight_mapping()` - 32 edges
4. `register_verified()` - 25 edges
5. `_get_owned_source_or_404()` - 21 edges
6. `load_excel()` - 19 edges
7. `HMMRunConfig` - 19 edges
8. `forward_fill_athlete_suggestions()` - 18 edges
9. `SourceError` - 17 edges
10. `useAuth()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `run_preflight()` --calls--> `preflight_mapping()`  [INFERRED]
  backend/app/analysis/service.py → algo/hpc_algo/api.py
- `athlete_forward_fill_suggestions()` --calls--> `forward_fill_athlete_suggestions()`  [INFERRED]
  backend/app/analysis/service.py → algo/hpc_algo/suggestions.py
- `header_merge_fill_for_sheet()` --calls--> `header_merge_fill_suggestions()`  [INFERRED]
  backend/app/analysis/service.py → algo/hpc_algo/suggestions.py
- `read_grid_fragment()` --calls--> `read_grid()`  [INFERRED]
  backend/app/analysis/service.py → algo/hpc_algo/workbook_editor.py
- `apply_grid_edits()` --calls--> `apply_cell_edits()`  [INFERRED]
  backend/app/analysis/service.py → algo/hpc_algo/workbook_editor.py

## Communities (119 total, 10 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (48): Первые ``rows`` строк листа после применения header_rows.      Используется в UI, sheet_preview(), _assign_role(), _clean_level(), column_levels(), describe_sheet_columns(), _extract_levels_from_multiindex(), flatten_columns() (+40 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (37): create_app(), _csrf_guard(), FastAPI-приложение hidden-patterns-combat backend., Простая проверка CSRF через middleware. Возвращает ответ-отказ     или ``None``, balls_zap_excel(), bout_resets_excel(), client(), dense_hmm_ready_excel() (+29 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (28): HMMView, MappingEditor, Tab, AuditTable(), DetectedColumns(), Props, RunsHistory(), STATE_TONE (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (24): api, ApiError, UserPublic, App(), RequireUser(), RequireVerified(), AuthContext, AuthContextValue (+16 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (29): Props, ROLE_OPTIONS, MUTATING, readCsrfTokenFromCookie(), request(), withCsrfHeader(), AnalysisResult, ApiError (+21 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (37): AthleteSuggestion, _cell_text(), forward_fill_athlete_suggestions(), ForwardFillReport, header_merge_fill_suggestions(), HeaderMergeReport, HeaderMergeSuggestion, _is_blank() (+29 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (22): CellEdit, ColumnMappingConfig, GridCellValue, SheetGridFragment, SheetMapping, SourcePrepPage(), ActiveCell, CellSuggestion (+14 more)

### Community 7 - "Community 7"
Cohesion: 0.15
Nodes (30): analyze(), count_empty_rows(), delete_mapping(), _ensure_draft(), get_athlete_forward_fill_suggestions(), get_header_merge_fill_suggestions(), get_header_rows_suggestion(), get_mapping() (+22 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (23): AnalysisRunFull, AnalysisRunSummary, CellEditPayload, GridEditsRequest, PreflightRequest, Опциональный фильтр листов для preflight'а., RemoveEmptyRowsRequest, Base (+15 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (23): expires_at(), generate_numeric_code(), hash_code(), Утилиты для коротких email-кодов (verification / password reset).  * Сгенерирова, Сгенерировать криптографически случайный числовой код заданной длины., verify_code(), hash_password(), authenticate() (+15 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (17): AnalysisView(), SourceSummary, Button(), BRAND, DOCS, LegalDocId, UPLOAD_GATE, DisclaimerBanner() (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.09
Nodes (22): Текущая задача, Что НЕ делает этот шаг, Группы задач, Порядок внедрения (рекомендуемый), Соответствие требованиям (обновлённая проекция), Сводка изменений относительно прежней версии спецификации, AUTH-LOGIN-1 — Вход по email и паролю (в т.ч. неподтверждённый email), AUTH-PWRESET-1 — Восстановление пароля по коду из email (+14 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (21): _basic_initial_distribution(), _basic_transition_matrix(), _build_bernoulli_interpretation(), _build_interpretation(), _detailed_initial_distribution(), _detailed_transition_matrix(), _emission_matrix(), _fit_bernoulli_variant() (+13 more)

### Community 13 - "Community 13"
Cohesion: 0.16
Nodes (21): _merged_header_xlsx_bytes(), Тесты мастера предобработки источника: lifecycle draft→ready и grid I/O., Очищаем ФИО в одной строке и ожидаем единственное предложение., Лист с двухстрочной шапкой и merge A1:B1 в верхней строке., Endpoint должен предложить материализацию slave-ячейки merged-шапки., Счётчик пустых строк должен совпадать с тем, что фактически удаляется., test_analyze_blocked_for_draft(), test_athlete_forward_fill_suggestions_blocked_after_finalize() (+13 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (19): AnalyzeConfig, preflight_mapping(), Вернуть предполагаемый :class:`ColumnMappingConfig`.      Функция нейтральная: э, Конфигурация анализа., test_analyze_with_bernoulli_on_dense(), Тесты HMM-ветки (TASK_SPEC_004).  Охватывают: * guard'ы блокируют HMM на тонких, test_different_seeds_may_differ(), test_fighter_style_is_never_used() (+11 more)

### Community 15 - "Community 15"
Cohesion: 0.19
Nodes (19): BernoulliHMMFitResult, BernoulliHMMParams, decode_sequence(), fit_bernoulli_hmm(), _forward_backward(), _log(), _log_bernoulli_likelihood(), _logsumexp() (+11 more)

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (18): ForgotPasswordRequest, LoginRequest, Публичный профиль текущего пользователя.      Поля, относящиеся к согласиям и он, RegisterRequest, ResetPasswordRequest, UserPublic, VerifyEmailRequest, BaseModel (+10 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (13): Конфигурация backend (без секретов в коде)., Настройки приложения.      Значения по умолчанию рассчитаны на локальный dev-реж, Settings, _build_auth_limiter(), get_auth_rate_limiter(), Глобальный rate-limiter и его DI-фабрика.  Выбор backend'а через ``HPC_RATE_LIMI, InMemoryRateLimiter, RateLimiter (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (18): _analyze_with_mapping(), _build_heuristic_warnings(), _build_hmm_charts(), _build_multirow_warning(), _build_report(), _decide_status_heuristic(), _decide_status_with_mapping(), _drop_totals_rows_from_frames() (+10 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (18): Enum, AnalysisResult, AnalysisStatus, AuditReport, ColumnDetectionReport, ColumnInfo, HiddenGroup, Pydantic-схемы для результата работы processing module.  Схема — единственный ко (+10 more)

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (17): delete_me(), _enforce_rate_limit(), _issue_session(), login(), me(), onboarding_complete(), HTTP-слой auth. Здесь только сериализация, cookie, коды ответа., Самостоятельное удаление аккаунта (PROFILE-1).      Каскадно удаляет источники п (+9 more)

### Community 21 - "Community 21"
Cohesion: 0.11
Nodes (18): Предметные инварианты (неперего­вариваемые), Структура репозитория, Быстрый старт, Проверка, Статус проекта, Известные проблемы и ограничения, Архитектурные границы (напоминание), Что делать дальше (+10 more)

### Community 22 - "Community 22"
Cohesion: 0.16
Nodes (15): analyze_source(), is_honest_baseline(), Проанализировать Excel-источник.      Если в ``config.column_mapping`` передан п, Ключевая проверка честности: HMM-поля в MVP отсутствуют., Domain invariant: наблюдения — ЗАП, а не действия спортсмена., test_analysis_summary_structure(), test_analyze_source_failed_for_missing_file(), test_analyze_source_flags_missing_mapping() (+7 more)

### Community 23 - "Community 23"
Cohesion: 0.11
Nodes (17): Область ответственности, Тесты и линт, Структура пакета, Известные ограничения, Установка (локально), Контракт `AnalysisResult`, Запуск из CLI, code:bash (python3.11 -m venv .venv) (+9 more)

### Community 24 - "Community 24"
Cohesion: 0.12
Nodes (12): athlete_forward_fill_suggestions(), create_pending_run(), header_merge_fill_for_sheet(), list_sheets(), Тонкая обёртка над :func:`hpc_algo.analyze_source`.  Любая исследовательская лог, Получить предложения forward-fill по колонке ФИО для одного листа.      Тонкая о, Получить предложения «материализовать merged-ячейки шапки».      Тонкая обёртка, Подсказать ``header_rows`` для листа + flatten-превью имён. (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.14
Nodes (16): build_bernoulli_sequences(), build_observation_sequences(), _channel_from_flat_name(), _episode_key(), EpisodeSequence, Свернуть одну строку-эпизод в один HMM-токен.      Один эпизод = одно ЗАП-наблюд, Совместимый wrapper над :func:`hpc_algo.mapping.episode_key`.      Раньше функци, Собрать последовательности бинарных векторов по эпизодам.      Возвращает ``(ent (+8 more)

### Community 26 - "Community 26"
Cohesion: 0.18
Nodes (14): evaluate_guards(), Вернуть список guard-предупреждений.      Возвращаются два класса предупреждений, episode_grouping_columns(), Колонки для уникальной идентификации эпизода в одном листе.      Эпизод — это те, _empty_audit(), Регрессии на ключи группировки эпизодов и серий.  Две разные оси:  * baseline-сч, Если на каждого борца одна строка-эпизод — guard должен сработать., Файл, где у двух борцов идут одинаковые номера эпизодов 1,2,3. (+6 more)

### Community 27 - "Community 27"
Cohesion: 0.19
Nodes (13): _detect_sheet(), _looks_like_scoring_scale(), Эвристическая детекция колонок-кандидатов для предметных групп.  Важно: это *тол, Похоже ли содержимое на судейскую балльную шкалу (малый алфавит ≤ 16)., Во сколько раз содержимое колонки похоже на ЗАП-наблюдение.      Поддерживаются, Насколько содержимое колонки похоже на время/длительность., Кандидаты в серой зоне — пригодны для ``warnings``, не для выводов., _Rule (+5 more)

### Community 28 - "Community 28"
Cohesion: 0.2
Nodes (13): analysis_summary(), analyze_cmd(), analyze_with_mapping_cmd(), _dump(), preflight_cmd(), CLI для независимого запуска processing module.  Примеры::      hpc-algo analyze, Короткий текстовый отчёт (ru)., Выполнить honest-анализ Excel-источника. (+5 more)

### Community 29 - "Community 29"
Cohesion: 0.15
Nodes (13): HMMRunConfig, Параметры запуска HMM-ветки., BaselineReport, HMMParameters, HMMResult, HMMTrajectory, Всё, что можно честно посчитать без полноценной HMM.      До применения column m, Параметры обученной HMM.      * ``state_labels`` — строго предметные (``маневрир (+5 more)

### Community 30 - "Community 30"
Cohesion: 0.16
Nodes (11): create_source(), delete_owned_source(), finalize_source(), get_owned_source(), Бизнес-операции источников (upload, list, get, delete)., Сохранить сериализованный ColumnMappingConfig.      Валидация (парсинг в pydanti, Перевести источник из ``draft`` в ``ready``.      Запретим финализацию, если у п, Пересчитать ``size_bytes`` и ``sha256`` после правки файла на диске. (+3 more)

### Community 31 - "Community 31"
Cohesion: 0.21
Nodes (12): register_verified(), test_delete_account_removes_user_and_session(), test_onboarding_complete(), test_password_reset_mismatched_passwords(), test_resend_for_already_verified_is_idempotent(), UPLOAD-GATE-1: без `confirm_upload=true` загрузка не начинается., test_upload_and_list(), test_upload_rejects_non_excel() (+4 more)

### Community 32 - "Community 32"
Cohesion: 0.19
Nodes (7): fmt(), HMMView(), Props, STATE_COLORS, ViterbiTimeline(), HMMResult, HMMTrajectory

### Community 33 - "Community 33"
Cohesion: 0.23
Nodes (12): _audit_sheet(), detect_totals_row_indices(), _is_totals_row(), Честный аудит Excel-источника.  Модуль отвечает только за структурные и статисти, Вернуть до ``limit`` уникальных непустых значений., Набор очень простых, но полезных флагов качества., True, если хотя бы одна строковая ячейка строки совпадает с маркером итогов., Найти позиционные индексы (0..len-1) строк-итогов в df.      Возвращает позиции (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.21
Nodes (11): ExcelLoadError, load_excel(), LoadedExcel, Чтение Excel-источника без какой-либо бизнес-логики.  Модуль сознательно узкий:, Результат чтения Excel-файла., Проблема при чтении Excel, которая должна быть отражена в ``errors``., Прочитать Excel-файл.      * читает все листы через ``pandas.read_excel(..., she, RuntimeError (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.15
Nodes (7): grid_with_empty_rows(), Тесты редактора Excel: чтение/запись блоков и удаление пустых строк., count_empty_rows должен возвращать ту же цифру, что и remove_empty_rows., Если заголовок занимает несколько строк, удаление data не сдвигает шапку., Лист с заголовком, тремя data-строками и двумя полностью пустыми., test_count_empty_rows_matches_remove(), test_remove_empty_rows_does_not_touch_multirow_header()

### Community 36 - "Community 36"
Cohesion: 0.24
Nodes (12): _bernoulli_params_from_result(), cross_validate(), CVFoldResult, CVReport, Cross-validation для HMM: k-fold по листам / весовым категориям.  Задача — честн, Вытянуть ``BernoulliHMMParams`` из уже собранного ``HMMResult``., Log-likelihood категориальной HMM на тестовой выборке.      Пишем скоринг сами,, K-fold CV по листам для текущей конфигурации HMM. (+4 more)

### Community 37 - "Community 37"
Cohesion: 0.27
Nodes (12): get_settings(), Функция-фабрика, удобна как Depends и для переопределения в тестах., create_access_token(), create_email_verification_token(), create_refresh_token(), decode_access_token(), decode_email_verification_token(), decode_refresh_token() (+4 more)

### Community 38 - "Community 38"
Cohesion: 0.22
Nodes (10): Тесты TASK_SPEC_009/010: refresh, email verification (codes), rate-limit backend, Ответ должен быть нейтральным независимо от существования аккаунта., Регистрация → выдача кода → POST /verify-email с правильным кодом., _register(), test_forgot_password_neutral_for_unknown_email(), test_refresh_issues_new_session(), test_register_creates_unverified_user(), test_resend_verification() (+2 more)

### Community 39 - "Community 39"
Cohesion: 0.27
Nodes (11): _aggregate_audit(), build_audit(), filter_audit_to_sheets(), Собрать :class:`AuditReport` для загруженного Excel., Сузить :class:`AuditReport` к подмножеству листов.      Используется в mapping-в, test_audit_counts_rows_and_columns(), test_audit_overall_null_ratio_bounded(), test_audit_preview_non_empty() (+3 more)

### Community 40 - "Community 40"
Cohesion: 0.2
Nodes (11): build_baseline(), Собрать :class:`BaselineReport` без подтверждённого mapping., detect_columns(), Собрать :class:`ColumnDetectionReport` по всем листам., Только уверенные ЗАП-кандидаты, пригодные для baseline-распределений., strong_zap_candidates(), ``detect_columns`` подсвечивает «Баллы» как ZAP-кандидата.      Проверяем сам фа, test_balls_supercol_detected_as_zap_candidate() (+3 more)

### Community 41 - "Community 41"
Cohesion: 0.23
Nodes (11): build_baseline_with_mapping(), build_charts(), _channel_from_flat_name(), _merge_counts(), Baseline-распределения и chart-ready данные.  Две ветки:  * ``build_baseline`` (, Value counts только по непустым значениям, как строки., Получить `channel_label` — атомарный (последний) уровень flatten-имени., Собрать baseline по подтверждённому mapping.      Возвращает ``(report, unknown_ (+3 more)

### Community 42 - "Community 42"
Cohesion: 0.27
Nodes (11): _analyze(), Тесты детализированной 7-state HMM (TASK_SPEC_005)., 40 эпизодов — detailed не должен пройти min_episodes_detailed=120., BIC-гейт: даже на плотной фикстуре 7-state не всегда выигрывает BIC.      Инвари, test_auto_falls_back_to_basic_on_thin_data(), test_auto_prefers_basic_unless_bic_wins(), test_detailed_blocked_on_thin_data(), test_detailed_runs_on_very_dense() (+3 more)

### Community 43 - "Community 43"
Cohesion: 0.21
Nodes (11): classify_zap_column(), Определить тип ЗАП-колонки и посчитать честные метрики событий.      Возвращает, Тесты TASK_SPEC_003_1: классификация ЗАП (categorical / binary / count / empty), test_binary_zap_populates_events_but_not_value_counts(), test_binary_zap_yields_chart_zap_events_by_channel(), test_categorical_zap_populates_events_by_channel(), test_classify_binary_only_zeros_and_ones(), test_classify_categorical() (+3 more)

### Community 44 - "Community 44"
Cohesion: 0.23
Nodes (11): _all_noop_per_athlete_excel(), _build_workbook(), _low_density_excel(), Регрессии Honesty layer HMM-ветки.  Покрывают изменения, которые делают вывод HM, Часть борцов имеет ВСЕ строки = noop, остальные — с ЗАП.      Используется для п, Собрать .xlsx с multi-row header и данными.      Структура — сильно упрощённая к, 200 эпизодов / 8 ZAP-событий по 2 каналам: плотность 4% (< 5%).      * min_episo, test_all_noop_sequences_are_excluded_from_training() (+3 more)

### Community 45 - "Community 45"
Cohesion: 0.32
Nodes (11): test_analyze_with_binary_zap_exposes_events_by_channel(), test_analyze_with_saved_mapping_returns_baseline_only(), test_delete_mapping(), test_mapping_endpoints_enforce_ownership(), test_preflight_returns_roles(), test_put_get_mapping_roundtrip(), test_put_mapping_rejects_invalid_payload(), test_sheet_columns_endpoint_ownership() (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.23
Nodes (11): _build_engine(), get_db(), get_engine(), get_sessionmaker(), init_schema(), SQLAlchemy session management., Создать таблицы. Для MVP используется вместо Alembic., FastAPI-зависимость для получения сессии БД. (+3 more)

### Community 47 - "Community 47"
Cohesion: 0.18
Nodes (9): Тесты инфраструктуры (TASK_SPEC_006): фоновый анализ, CSRF, rate-limit., Сброс пароля — без сессии; нельзя требовать X-CSRF-Token (cookie нет)., В prod-режиме (csrf_required=True) мутирующий запрос без     заголовка X-CSRF-To, _register(), test_background_analyze_reaches_done(), test_csrf_forgot_and_reset_work_without_token_when_csrf_on(), test_csrf_passes_with_matching_header(), test_csrf_required_blocks_mutations() (+1 more)

### Community 48 - "Community 48"
Cohesion: 0.17
Nodes (11): Области ответственности, Известные ограничения, Установка, Основные эндпоинты, Тесты, code:bash (source .venv/bin/activate), code:bash (uvicorn app.main:app --reload --app-dir backend), code:bash (cd backend) (+3 more)

### Community 49 - "Community 49"
Cohesion: 0.17
Nodes (11): Назначение алгоритма, Ожидаемый общий интерфейс, Скрытые состояния, Главный принцип, Наблюдения, Графики, Требования к честности результата, ALGORITHM_SPEC (+3 more)

### Community 50 - "Community 50"
Cohesion: 0.17
Nodes (11): Источники данных, Ожидаемые группы данных, Группы скрытых состояний, Маневрирование, КФВ, ВУП, Наблюдаемые ЗАП, Правило обработки неизвестной структуры (+3 more)

### Community 51 - "Community 51"
Cohesion: 0.18
Nodes (10): Текущая задача, Что должно появиться, Что НЕ делает этот шаг, Предметная структура, Инициализация `A`, `π`, `B`, Definition of Done, Guard'ы для 7-state, Режимы HMM (новый `hmm_mode`) (+2 more)

### Community 52 - "Community 52"
Cohesion: 0.18
Nodes (10): Текущая задача, Что НЕ делает этот шаг, Инварианты (неизменны), Формат миграции, Фоновая задача `analyze`, CSRF, Definition of Done, Формат docker-compose (+2 more)

### Community 53 - "Community 53"
Cohesion: 0.31
Nodes (8): ChartCard(), ChartsGrid(), extractMatrix(), getChartSheet(), HeatmapCard(), chart, toBarData(), ChartData

### Community 54 - "Community 54"
Cohesion: 0.24
Nodes (10): _attempt_bernoulli(), _attempt_categorical(), _detailed_data_ok(), fit_hmm(), fit_hmm_bernoulli(), Попробовать Bernoulli-вариант (detailed > auto по BIC > basic).      Возвращает, Однократная попытка обучить categorical HMM с расшифровкой причин.      Возвраща, Обучить HMM с учётом режима (basic / detailed / auto).      * ``mode == "basic"` (+2 more)

### Community 55 - "Community 55"
Cohesion: 0.2
Nodes (9): CellEdit, count_empty_rows(), _ensure_writable_cell(), GridFragment, Редактирование Excel-файла на стороне processing module.  Любая запись в xlsx до, Сосчитать data-строки, у которых все ячейки пустые.      Использует ту же логику, Удалить data-строки, у которых все ячейки пустые.      ``header_rows`` — 0-based, Для :class:`MergedCell` (все ячейки кроме master) — **снять** merge с     диапаз (+1 more)

### Community 56 - "Community 56"
Cohesion: 0.2
Nodes (8): _Bucket, client_ip(), enforce_csrf(), generate_csrf_token(), CSRF и rate-limit — лёгкие in-process реализации (TASK_SPEC_006).  Оба механизма, Выставить CSRF-cookie (НЕ HttpOnly — фронту нужно читать из JS)., FastAPI-зависимость: проверяет double-submit CSRF для мутирующих     запросов. В, set_csrf_cookie()

### Community 57 - "Community 57"
Cohesion: 0.24
Nodes (7): delete_source(), finalize_source(), get_source(), HTTP-слой /sources. Никакой логики алгоритма здесь нет., Подтвердить источник после прохождения мастера предобработки., _summary_for(), upload_source()

### Community 58 - "Community 58"
Cohesion: 0.2
Nodes (9): Текущая задача, Что НЕ делает этот шаг, Предметные инварианты, Схема наблюдений, Что должно появиться, Definition of Done, Guard'ы качества данных, Доменные ограничения на параметры HMM (+1 more)

### Community 59 - "Community 59"
Cohesion: 0.22
Nodes (8): drop_totals_rows(), Вернуть копию df без строк по указанным позициям (0..len-1).      Если ``positio, Регрессии для фильтра строк-итогов в audit (Phase 5).  Реальные файлы вида ``doc, «Итогов» как фамилия не должен срабатывать как маркер итогов., test_build_audit_records_totals_indices(), test_detect_totals_row_indices_ignores_substring_match(), test_detect_totals_row_indices_marks_known_marker(), test_drop_totals_rows_returns_clean_copy()

### Community 60 - "Community 60"
Cohesion: 0.22
Nodes (9): apply_grid_edits(), apply_cell_edits(), _coerce_for_excel(), Применить точечные правки ячеек к листу и сохранить файл.      Координаты ``row`, Привести входное значение из UI к тому, что openpyxl сохранит как ячейку., MergedCell: снять merge и записать в запрошенную (row, col), не в master., test_apply_cell_edits_merged_range_unmerges_and_writes_target_cell(), test_apply_cell_edits_unknown_sheet_raises() (+1 more)

### Community 61 - "Community 61"
Cohesion: 0.31
Nodes (8): Локальная dev-доставка email (TASK_SPEC_010).  Реальная SMTP-отправка не входит, Записать письмо в выбранный sink. Безопасно: исключения подавляются     и попада, _safe_filename(), send_email_verification_code(), send_mail(), send_password_reset_code(), forgot_password(), Запросить код восстановления.      Ответ всегда **нейтральный**, даже если email

### Community 62 - "Community 62"
Cohesion: 0.36
Nodes (6): _register_payload(), test_login_and_me(), test_login_wrong_password(), test_logout_clears_session(), test_register_duplicate_email(), test_register_sets_session_cookie()

### Community 63 - "Community 63"
Cohesion: 0.22
Nodes (8): AGENTS, Architecture Boundaries, Data and Modeling Rules, Delivery Expectations, Domain Invariants, Non-Negotiable Workflow, Purpose, Required Context

### Community 64 - "Community 64"
Cohesion: 0.22
Nodes (8): Цель продукта, Основной пользовательский сценарий, Интерфейсная идея, Главное продуктовое ограничение, Пользователь, MVP, Необязательно для первого MVP, PRODUCT_SPEC

### Community 65 - "Community 65"
Cohesion: 0.22
Nodes (8): Текущая задача, Предметные инварианты (без изменений), Что должно появиться, Правила и ограничения, code:python (class SheetMapping(BaseModel):), Формат `ColumnMappingConfig`, Definition of Done, TASK_SPEC_003_COLUMN_MAPPING

### Community 66 - "Community 66"
Cohesion: 0.25
Nodes (8): read_grid_fragment(), Подготовить значение Excel-ячейки к сериализации., Прочитать прямоугольный блок ячеек ``sheet_name``.      Координаты — 1-based, ка, read_grid(), _to_jsonable(), test_read_grid_pads_beyond_sheet(), test_read_grid_returns_full_block(), test_read_grid_unknown_sheet_raises()

### Community 67 - "Community 67"
Cohesion: 0.25
Nodes (7): Регрессии для проактивных warnings мэппинга (Phase 3).  * ``mapping.zap_candidat, Оператор разметил athlete/episode, но не положил «Баллы» в ZAP., Если «Баллы» в роли ZAP, warning не эмитится., Без athlete роли виртуальный bout не восстановить → mapping.bout_missing., test_bout_missing_fires_when_athlete_role_absent(), test_zap_candidate_unmapped_fires_when_balls_not_in_zap_role(), test_zap_candidate_unmapped_silenced_when_balls_explicitly_zap()

### Community 68 - "Community 68"
Cohesion: 0.43
Nodes (7): Тесты TASK_SPEC_007: история запусков и preview листа., _setup(), test_run_result_endpoint_rejects_non_done(), test_run_result_endpoint_returns_done_run(), test_runs_history_lists_all_runs(), test_sheet_preview_enforces_ownership(), test_sheet_preview_returns_rows_and_columns()

### Community 69 - "Community 69"
Cohesion: 0.25
Nodes (7): Быстрый старт, Что внутри, Ключевые места, Маршруты, code:bash (cd frontend), hpc-frontend — React + Vite SPA, Ограничения MVP

### Community 70 - "Community 70"
Cohesion: 0.25
Nodes (7): Предметная постановка, Наблюдения, Скрытые состояния, Причинная цепочка, Главная исследовательская задача, Важное методологическое ограничение, DOMAIN_SPEC

### Community 71 - "Community 71"
Cohesion: 0.33
Nodes (6): groupByCode(), severityLabels, severityStyles, WarningsList(), WarningItem, WarningSeverity

### Community 72 - "Community 72"
Cohesion: 0.33
Nodes (6): current_user(), current_verified_user(), _extract_token(), FastAPI-зависимости auth: чтение cookie, выдача текущего пользователя., Обязательная зависимость: возвращает пользователя или 401., Зависимость для рабочих маршрутов: требует подтверждённый email.      Использует

### Community 73 - "Community 73"
Cohesion: 0.33
Nodes (4): Локальный storage adapter. Абстрагирован, чтобы заменить на S3 позже., Обезопасить имя файла, сохраняя кириллицу., _safe_filename(), StoredFile

### Community 74 - "Community 74"
Cohesion: 0.48
Nodes (6): E2E для HMM-ветки (TASK_SPEC_004) через HTTP., _register_upload(), test_analyze_blocks_hmm_on_thin_data(), test_analyze_detailed_mode_on_dense_data(), test_analyze_rejects_unknown_mode(), test_analyze_returns_hmm_ready_on_dense_data()

### Community 75 - "Community 75"
Cohesion: 0.29
Nodes (6): Название проекта, Предметная идея, Будущий продукт, Идея интерфейса, Общая цель, PROJECT_CONTEXT

### Community 76 - "Community 76"
Cohesion: 0.29
Nodes (6): Главные правила, Запрещено, Требования к алгоритму, Требования к ответу после изменений, AGENT_RULES, Требования к работе с Excel

### Community 77 - "Community 77"
Cohesion: 0.29
Nodes (6): Текущая задача, Что должно появиться, Что НЕ делает этот шаг, Контекст, Definition of Done, TASK_SPEC_003_1_ZAP_ENCODING

### Community 78 - "Community 78"
Cohesion: 0.29
Nodes (6): Agent Context Guide, Core Meaning, How To Select The Active Task, Purpose, Read Order, Safe Default

### Community 79 - "Community 79"
Cohesion: 0.29
Nodes (6): Текущая задача, Мотивация, Что должно появиться, Что НЕ делает этот шаг, Definition of Done, TASK_SPEC_008_EMISSIONS_CV

### Community 80 - "Community 80"
Cohesion: 0.29
Nodes (6): Общие критерии, Для архитектурной итерации, Для backend MVP, DEFINITION_OF_DONE, Для frontend MVP, Для processing module

### Community 81 - "Community 81"
Cohesion: 0.29
Nodes (6): Текущая задача, Что должно появиться, Что НЕ делает этот шаг, Контекст, Definition of Done, TASK_SPEC_003_2_SHEET_COLUMNS

### Community 82 - "Community 82"
Cohesion: 0.4
Nodes (4): spec, translateWarning(), WARNING_SPECS, WarningSpec

### Community 83 - "Community 83"
Cohesion: 0.33
Nodes (5): Регрессии для эвристики детекции «Баллы» как ZAP-кандидата.  В реальных файлах в, preflight_mapping кладёт колонку «Баллы» в роль ЗАП.      После добавления марке, С «Баллы» в роли ZAP плотность сигнала достаточна для HMM.      Сценарий повторя, test_balls_in_mapping_lifts_zap_density(), test_preflight_assigns_balls_to_zap_role()

### Community 84 - "Community 84"
Cohesion: 0.33
Nodes (5): Текущая задача, Что НЕ делает этот шаг, Что должно появиться, Definition of Done, TASK_SPEC_007_UX

### Community 85 - "Community 85"
Cohesion: 0.33
Nodes (5): Текущий репозиторий, Что проверять в коде, Важные наблюдения из прошлых итераций, REPOSITORY_NOTES, Маппинг ZAP в реальных файлах

### Community 86 - "Community 86"
Cohesion: 0.33
Nodes (5): Текущая задача, Что должно появиться, Definition of Done, Processing module, TASK_SPEC_002_MVP_SCAFFOLD

### Community 87 - "Community 87"
Cohesion: 0.33
Nodes (5): Текущая задача, Что должно появиться, Что НЕ делает этот шаг, Definition of Done, TASK_SPEC_009_PROD_RELIABILITY

### Community 88 - "Community 88"
Cohesion: 0.33
Nodes (5): Задача, Что нужно получить от агента, Ограничения, Что нужно учесть, ARCHITECTURE_REQUEST

### Community 89 - "Community 89"
Cohesion: 0.4
Nodes (3): contains_any(), Небольшие утилиты работы с кириллическими заголовками.  Назначение ограничено: n, Вернуть ``True``, если в нормализованной строке встречается любой маркер.

### Community 90 - "Community 90"
Cohesion: 0.4
Nodes (5): _load_mapping_from_source(), Синхронный вариант запуска анализа — используется в CLI / тестах., Исполнитель фоновой задачи.      BackgroundTasks запускается после возврата HTTP, run_and_persist_sync(), _run_in_its_own_session()

### Community 91 - "Community 91"
Cohesion: 0.5
Nodes (4): _make_alembic_config(), Запуск Alembic-миграций из кода (без shell).  `create_all` не добавляет колонки, Применить миграции до ``head`` для ``settings.database_url``., run_alembic_upgrade_to_head()

### Community 92 - "Community 92"
Cohesion: 0.7
Nodes (4): _setup_source(), test_analyze_and_fetch_result(), test_analyze_isolated_between_users(), test_result_absent_until_analyze()

### Community 93 - "Community 93"
Cohesion: 0.4
Nodes (4): Текущая задача, На выходе нужно получить, Важно, TASK_SPEC_001_ARCHITECTURE

### Community 94 - "Community 94"
Cohesion: 0.5
Nodes (4): _bic(), _free_parameters(), Количество свободных параметров HMM (для BIC)., BIC = -2 * log_likelihood + k * log(N).

## Knowledge Gaps
- **501 isolated node(s):** `queryClient`, `AuthState`, `AuthContextValue`, `AuthContext`, `SortDir` (+496 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `analyze_source()` connect `Community 22` to `Community 34`, `Community 67`, `Community 39`, `Community 7`, `Community 40`, `Community 41`, `Community 42`, `Community 44`, `Community 43`, `Community 14`, `Community 18`, `Community 83`, `Community 90`, `Community 28`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `get_sessionmaker()` connect `Community 46` to `Community 90`, `Community 68`, `Community 38`, `Community 47`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `_run_in_its_own_session()` connect `Community 90` to `Community 7`, `Community 46`, `Community 14`, `Community 22`, `Community 24`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `analyze_source()` (e.g. with `load_excel()` and `str`) actually correct?**
  _`analyze_source()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 50 inferred relationships involving `str` (e.g. with `_detect_sheet()` and `_normalize_athlete()`) actually correct?**
  _`str` has 50 INFERRED edges - model-reasoned connections that need verification._
- **Are the 48 inferred relationships involving `AnalyzeConfig` (e.g. with `HMMRunConfig` and `ExcelLoadError`) actually correct?**
  _`AnalyzeConfig` has 48 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `preflight_mapping()` (e.g. with `preflight_cmd()` and `_analyze()`) actually correct?**
  _`preflight_mapping()` has 30 INFERRED edges - model-reasoned connections that need verification._