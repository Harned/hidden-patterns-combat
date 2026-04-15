# Data Dictionary

Документация теперь разделена на два уровня:

1. Нормализация исходных колонок (`raw -> cleaned`)
2. Продуктовый слой наблюдений/эпизодов (`cleaned -> canonical_episode_table`)

## 1) Raw/Cleaned mapping

Файл:
- `src/hidden_patterns_combat/preprocessing/resources/data_dictionary_v1.json`

API:
- `hidden_patterns_combat.preprocessing.data_dictionary.DataDictionary`

Логические группы:
- `metadata`
- `maneuvering`
- `kfv`
- `vup`
- `outcomes`
- `other`

## 2) Observation mapping (inverse mode)

Файл:
- `src/hidden_patterns_combat/preprocessing/resources/observation_mapping_v1.json`

API:
- `hidden_patterns_combat.preprocessing.observation_builder`

Канонические наблюдаемые классы:
- `zap_r`
- `zap_n`
- `zap_t`
- `hold`
- `arm_submission`
- `leg_submission`
- `no_score`
- `unknown`

Служебные поля observation layer:
- `observed_zap_source_columns`
- `observation_quality_flag`
- `mapping_version`

Правила:
- при `score == 0` и отсутствии завершающего действия -> `no_score`;
- при неоднозначности/недостаточности -> `unknown`;
- score-to-class применяется только как явное правило из mapping-конфига.

## 3) Canonical episode table (product layer)

Минимальные поля:
- `athlete_name`, `athlete_id`
- `sheet_name`, `weight_class`
- `episode_id`, `episode_time_sec`, `pause_time_sec`
- `score`
- `maneuver_right_code`, `maneuver_left_code`
- `kfv_capture_code`, `kfv_grip_code`, `kfv_wrap_code`, `kfv_hook_code`, `kfv_post_code`
- `vup_code`
- `observed_zap_class`, `observation_quality_flag`
- `is_total_row`, `is_train_eligible`
- `source_row_index`, `source_record_id` (traceability)

Train eligibility:
- total rows исключаются;
- `unknown` исключается из train;
- traceability сохраняется для всех строк.
