# AGENTS

## Purpose

This repository requires context-first work. Before planning, coding, refactoring, or reviewing, read the relevant files in `docs/agent_context`.

## Required Context

Always read `docs/agent_context/README.md` first.

Then always read:

- `docs/agent_context/PROJECT_CONTEXT.md`
- `docs/agent_context/DOMAIN_SPEC.md`
- `docs/agent_context/AGENT_RULES.md`
- `docs/agent_context/DATA_SPEC.md`
- `docs/agent_context/PRODUCT_SPEC.md`
- `docs/agent_context/ALGORITHM_SPEC.md`
- `docs/agent_context/DEFINITION_OF_DONE.md`
- `docs/agent_context/REPOSITORY_NOTES.md`

Then inspect `docs/agent_context/TASK_SPEC_*.md` and identify the active task before implementation. If the active `TASK_SPEC` is unclear, ask instead of guessing.

Also read when relevant:

- `docs/agent_context/ARCHITECTURE_REQUEST.md`
- `docs/agent_context/CLAUDE_START_PROMPT.md`

## Non-Negotiable Workflow

1. Read context first.
2. Identify the active `TASK_SPEC`.
3. Summarize the task in repository terms.
4. Decide whether the task is analysis-only, architecture-only, MVP scaffold, or implementation.
5. Do not start implementation until the current `TASK_SPEC` has been analyzed.
6. If the spec says analysis first or architecture first, stop at plan and do not code.

## Domain Invariants

- Observations are `ЗАП`.
- Hidden states follow the domain logic `маневрирование -> КФВ -> ВУП -> ЗАП`.
- Do not replace observations with athlete actions.
- Do not center the main result around `fighter_style`.
- Do not present technical model output as valid diagnosis without methodological support.

## Architecture Boundaries

- The algorithm must remain an independent processing module.
- Backend may orchestrate auth, uploads, persistence, and algorithm calls.
- Frontend may display charts, tables, warnings, and reports.
- Do not place research or modeling logic inside API endpoints.
- Do not place research or modeling logic inside UI components.
- Keep the processing module usable from backend, CLI, tests, and notebooks.

## Data and Modeling Rules

Always perform Excel audit before analysis:

- sheets
- shape
- columns
- dtypes
- missing values
- suspicious values
- candidate mappings for hidden-state groups, `ЗАП`, and time

If data is insufficient or mapping is unreliable:

- return an honest `baseline` / `audit` / `warnings` result
- expose detected columns and assumptions
- do not fabricate HMM outputs
- do not fabricate hidden trajectories or hidden states
- do not claim that diagnosis was completed

## Delivery Expectations

After completing a task, state:

- what was done
- why this approach was chosen
- how to run or verify it
- what limitations remain
- what the recommended next step is
