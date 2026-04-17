# Analytics Artifacts (Versioned)

`analytics/runs/` stores published intermediate/final artifacts that are safe to version in Git for downstream analytics and reproducibility.

Policy:
- Do not commit raw sources (`data/raw/*`).
- Keep local heavy/ephemeral outputs in `artifacts/*`.
- Publish selected reproducible artifacts from a run into `analytics/runs/<run_id>/`.

Publish command:

```bash
PYTHONPATH=src .venv/bin/python scripts/publish_inverse_artifacts.py \
  --source artifacts/inverse_diagnostic \
  --target-root analytics/runs
```

Optional:

```bash
PYTHONPATH=src .venv/bin/python scripts/publish_inverse_artifacts.py \
  --source artifacts/inverse_diagnostic \
  --target-root analytics/runs \
  --include-plots
```
