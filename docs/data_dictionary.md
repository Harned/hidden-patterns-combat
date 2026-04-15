# Data Dictionary

## Layers

1. `raw -> cleaned` (header normalization + logical blocks)
2. `cleaned -> observed layer` (observation builder)
3. `cleaned + observed -> canonical episode table`

## 1) Raw/Cleaned mapping

Primary dictionary:
- `src/hidden_patterns_combat/preprocessing/resources/data_dictionary_v1.json`

API:
- `hidden_patterns_combat.preprocessing.data_dictionary.DataDictionary`

## 2) Observation mapping (inverse mode)

Config:
- `src/hidden_patterns_combat/preprocessing/resources/observation_mapping_v1.json`

API:
- `hidden_patterns_combat.preprocessing.observation_builder`

Canonical observed classes:
- `zap_r`, `zap_n`, `zap_t`, `hold`, `arm_submission`, `leg_submission`, `no_score`, `unknown`

Quality fields:
- `observation_resolution_type`
  - `direct_finish_signal`
  - `inferred_from_score`
  - `no_score_rule`
  - `ambiguous`
  - `unknown`
- `observation_confidence_label`
  - `high`, `medium`, `low`
- `observation_quality_flag`
- `observed_zap_source_columns`
- `mapping_version`

Rule priority:
1. direct finish signals
2. controlled score fallback
3. explicit unknown/ambiguous marking when data is insufficient or conflicting

## 3) Canonical episode table

Built by:
- `hidden_patterns_combat.preprocessing.canonical_episode_table`

Core fields:
- episode identity and traceability:
  - `athlete_name`, `athlete_id`, `sheet_name`, `episode_id`
  - `source_row_index`, `source_record_id`
- sequence fields:
  - `sequence_id`
  - `sequence_quality_flag`
  - `sequence_resolution_type`
- tactical/temporal fields:
  - `maneuver_right_code`, `maneuver_left_code`
  - `kfv_capture_code`, `kfv_grip_code`, `kfv_wrap_code`, `kfv_hook_code`, `kfv_post_code`
  - `vup_code`
  - `episode_time_sec`, `pause_time_sec`, `score`
- observation fields:
  - `observed_zap_class`
  - `observation_resolution_type`
  - `observation_confidence_label`
  - `observation_quality_flag`
- filtering flags:
  - `is_total_row`
  - `is_train_eligible`
