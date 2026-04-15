# Project Context

## Active modes

1. `research` mode:
- HMM по инженерным признакам эпизода;
- цель: исследовательская структурная интерпретация.

2. `inverse-diagnostic` mode:
- HMM обратной диагностики по `observed_zap_class`;
- цель: продуктовая траектория скрытых состояний + quality-aware recommendation.

## Inverse-diagnostic: what is observed vs hidden

Hidden:
- `S1` (maneuvering)
- `S2` (KFV)
- `S3` (VUP)

Observed:
- `zap_r`, `zap_n`, `zap_t`, `hold`, `arm_submission`, `leg_submission`, `no_score`, `unknown`

Observation quality layer:
- `observation_resolution_type` (`direct_finish_signal`, `inferred_from_score`, `no_score_rule`, `ambiguous`, `unknown`)
- `observation_confidence_label` (`high`, `medium`, `low`)
- `observation_quality_flag`

## Sequence segmentation

`canonical_episode_table` формирует `sequence_id` так:
- сначала explicit `sequence_id/bout_id` из источника;
- иначе детерминированный surrogate key + блочная сегментация по эпизодам;
- качество сегментации помечается через `sequence_quality_flag`.

Низкое качество сегментации может исключать строки из train через `is_train_eligible`.

## Notebook role

Notebook не содержит отдельной реализации логики.
Он является frontend к `run_inverse_diagnostic_cycle(...)` и показывает артефакты inline.
