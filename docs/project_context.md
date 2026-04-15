# Project Context

## Статус

Проект перешел от единого исследовательского MVP к архитектуре с двумя режимами:

1. `research`:
- HMM по инженерным признакам эпизода;
- цель: исследование структуры скрытых состояний.

2. `inverse-diagnostic`:
- HMM обратной диагностики, где инференс идет по `observed_zap_class`;
- цель: продуктовая диагностика эпизодной траектории и рекомендация ЛПР.

## Формализация inverse режима

Скрытые состояния:
- `S1` (маневрирование)
- `S2` (КФВ)
- `S3` (ВУП)

Наблюдения:
- `zap_r`, `zap_n`, `zap_t`, `hold`, `arm_submission`, `leg_submission`, `no_score`, `unknown`

На обучении:
- можно использовать полные исторические данные;
- hidden-state feature layer используется для semantic initialization и post-hoc relabeling.

На инференсе:
- используется наблюдаемая последовательность `O_t`;
- результат: Viterbi-цепочка, posterior/confidence профиль, диагностическая интерпретация, рекомендация.

## Ключевые новые слои

- `preprocessing/observation_builder.py`: детерминированное построение `observed_zap_class`.
- `preprocessing/canonical_episode_table.py`: нормализованная таблица эпизодов + traceability.
- `features/hidden_state_features.py`: явное разделение hidden vs observed.
- `modeling/inverse_hmm.py`: discrete inverse HMM.
- `app/inverse_diagnostic_cycle.py`: orchestration продуктового контура.

## Ограничения

- Нет причинной интерпретации, только вероятностная диагностика.
- `unknown` используется явно при неоднозначном маппинге.
- Рекомендация помечается как недостаточно уверенная при низком confidence.
