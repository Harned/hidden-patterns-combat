# Roadmap

## Done in this iteration

1. Observation layer стал quality-aware (`resolution_type`, `confidence_label`, explicit score fallback transparency).
2. Sequence segmentation усилена: `sequence_id`, `sequence_quality_flag`, deterministic surrogate with explicit-first strategy.
3. Inverse pipeline diagnostics/report дополнились quality summaries и transition-aware recommendation profile.
4. Notebook frontend переведен на единый entrypoint `run_inverse_diagnostic_cycle(...)`.
5. Добавлены логические тесты для observation/sequence/inverse pipeline/CLI-notebook consistency.

## Next

1. Добавить расширенный калибратор confidence по историческим данным.
2. Добавить метрики качества inverse режима на train/validation split.
3. Расширить sequence segmentation для турниров с нестандартной нумерацией эпизодов.
4. Добавить режим semi-supervised доразметки ambiguous observations экспертом.
