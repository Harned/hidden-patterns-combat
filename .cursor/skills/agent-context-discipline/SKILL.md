---
name: agent-context-discipline
description: Enforces repository-specific analysis discipline for hidden-patterns-combat. Use for any planning, implementation, refactor, architecture, backend, frontend, algorithm, Excel, HMM, or diagnostic task in this repository. Requires reading docs/agent_context, identifying the current TASK_SPEC before coding, preserving domain semantics where observations are ZAP and hidden states are maneuvering/KFV/VUP, keeping the algorithm as an independent module, and returning an honest baseline/audit/warnings when data is insufficient.
---

# Agent Context Discipline

## When to use

Apply this skill by default for any substantial task in `hidden-patterns-combat`.

## Required context read

Before proposing architecture, changing code, or reviewing implementation, read the relevant files in `docs/agent_context`.

Read `docs/agent_context/README.md` first.

Then read the minimum required set:

- `docs/agent_context/PROJECT_CONTEXT.md`
- `docs/agent_context/DOMAIN_SPEC.md`
- `docs/agent_context/AGENT_RULES.md`
- `docs/agent_context/DATA_SPEC.md`
- `docs/agent_context/PRODUCT_SPEC.md`
- `docs/agent_context/ALGORITHM_SPEC.md`
- `docs/agent_context/DEFINITION_OF_DONE.md`
- `docs/agent_context/REPOSITORY_NOTES.md`

Task selection step:

- Inspect available `docs/agent_context/TASK_SPEC_*.md` files.
- Identify the current task spec from the user request and repo context.
- If the active task spec is unclear, ask before implementation.

Also read when relevant:

- `docs/agent_context/ARCHITECTURE_REQUEST.md`
- `docs/agent_context/CLAUDE_START_PROMPT.md`

## Non-negotiable workflow

1. Read context first.
2. Identify the active `TASK_SPEC`.
3. Summarize the task in repository terms.
4. Check whether the task is analysis-only, architecture-only, MVP scaffold, or implementation.
5. Do not start implementation until the current `TASK_SPEC` has been analyzed.
6. If the task is architectural or the spec says not to code yet, stop at analysis and plan.

## Domain invariants

- Observations are `ЗАП`.
- Hidden states follow the domain chain `маневрирование -> КФВ -> ВУП -> ЗАП`.
- Do not replace observations with athlete actions.
- Do not center the main result around `fighter_style`.
- Do not claim scientific validity just because a model runs.

## Architecture constraints

- The algorithm must remain an independent processing module.
- Backend may call the algorithm, but must not contain research logic.
- Frontend may render results, but must not contain research logic.
- The processing module should be callable from API, CLI, tests, and notebooks.

## Data and modeling rules

Always perform Excel audit before analysis:

- sheets
- shape
- columns
- dtypes
- missing values
- suspicious values
- candidate mappings for hidden-state groups, `ЗАП`, and time

If the data is insufficient or mapping is unreliable:

- return an honest `baseline` / `audit` / `warnings` result
- surface detected columns and assumptions
- do not fabricate HMM outputs
- do not return fake trajectories or fake hidden states

## Expected agent behavior

In responses and implementations:

- explain what context was used
- state which `TASK_SPEC` is active
- separate algorithm / backend / frontend concerns
- prefer a truthful baseline over unsupported inference
- preserve reproducibility and clear run instructions
- name remaining risks and limitations explicitly

## Done check

Before finishing, verify:

- context from `docs/agent_context` was used
- the active `TASK_SPEC` was analyzed before coding
- domain semantics stayed correct
- algorithm independence was preserved
- backend/frontend did not absorb research logic
- insufficient data leads to baseline/audit/warnings instead of fake diagnostics
