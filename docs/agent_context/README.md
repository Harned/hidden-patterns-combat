# Agent Context Guide

## Purpose

This directory is the source of truth for repository context. Read it before planning, coding, refactoring, or reviewing.

## Read Order

Read files in this order:

1. `README.md`
2. `PROJECT_CONTEXT.md`
3. `DOMAIN_SPEC.md`
4. `AGENT_RULES.md`
5. `DATA_SPEC.md`
6. `PRODUCT_SPEC.md`
7. `ALGORITHM_SPEC.md`
8. `DEFINITION_OF_DONE.md`
9. `REPOSITORY_NOTES.md`

Then choose the active task from `TASK_SPEC_*.md`.

Read additionally when relevant:

- `ARCHITECTURE_REQUEST.md`
- `CLAUDE_START_PROMPT.md`

## Core Meaning

- Observations are `ЗАП`.
- Hidden-state logic follows `маневрирование -> КФВ -> ВУП -> ЗАП`.
- The algorithm must remain an independent module.
- Backend and frontend must not contain research logic.
- If data or mapping is insufficient, return honest `baseline` / `audit` / `warnings`.
- Do not fabricate HMM outputs or scientific conclusions.

## How To Select The Active Task

Use the user request and current repository stage to choose the relevant `TASK_SPEC`.

- Use `TASK_SPEC_001_ARCHITECTURE.md` for context analysis, architecture proposal, stack choice, module boundaries, MVP definition, and phased planning.
- Use `TASK_SPEC_002_MVP_SCAFFOLD.md` for implementing the first working product scaffold after architecture is agreed.
- Use `TASK_SPEC_003_COLUMN_MAPPING.md` for adding column mapping (multi-row headers, user-confirmed role assignment) once MVP scaffold is in place. Enables real baseline distributions for all four domain groups while still forbidding HMM.
- Use `TASK_SPEC_003_1_ZAP_ENCODING.md` for handling binary / count ZAP columns (not just categorical labels). Extends the baseline output with `zap_events_by_channel` and per-column classification. Still forbids HMM.
- Use `TASK_SPEC_003_2_SHEET_COLUMNS.md` for exposing the full flatten-column list of a selected sheet to the mapping editor, so the user can assign roles to columns that preflight did not pick up. Still forbids HMM.
- Use `TASK_SPEC_004_HMM.md` for the first HMM diagnostic pass (3-state: маневрирование / КФВ / ВУП, observations = ЗАП). HMM only runs when strict data-quality guards pass; otherwise baseline stays untouched and the algorithm returns an honest warning. Hidden state names, observation schema and `fighter_style` prohibition must be preserved.
- Use `TASK_SPEC_005_DETAILED_HMM.md` for the detailed 7-state HMM (маневры / захваты / хваты / обхваты / прихваты / упоры / ВУП) gated by stronger data-quality guards and a BIC comparison against the 3-state baseline. `hmm_mode` in AnalyzeConfig controls selection (`auto` / `detailed` / `basic` / `off`). Invariants from TASK_SPEC_004 remain unchanged.

If more than one task spec appears relevant, stop and clarify before implementation.

## Safe Default

If the task is ambiguous, treat it as analysis-first. Do not start coding until the current `TASK_SPEC` is explicit and aligned with the request.
