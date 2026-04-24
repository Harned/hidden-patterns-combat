# hpc-frontend — React + Vite SPA

Русскоязычный интерфейс для hidden-patterns-combat. Отвечает только за
отображение `AnalysisResult` и за взаимодействие с backend.
**Исследовательской логики здесь нет.**

## Быстрый старт

```bash
cd frontend
npm install
npm run dev
```

Открыть http://127.0.0.1:5173. Vite проксирует `/api` на
`http://127.0.0.1:8000` (uvicorn).

## Что внутри

- React 18 + TypeScript + Vite.
- Роутинг: `react-router-dom` (`/login`, `/register`, `/app`).
- Запросы: `@tanstack/react-query` + `fetch` с `credentials: "include"` (HttpOnly-cookie).
- UI: Tailwind + минимальные собственные компоненты (`Button`, `Input`, `Card`, `Badge`, `Section`).
- Графики: `recharts` (bar / hbar).

## Ключевые места

- `src/api/types.ts` — типы зеркалят pydantic-схемы backend/algo.
- `src/api/client.ts` — тонкий fetch-клиент.
- `src/auth/AuthContext.tsx` — глобальная сессия через cookie + `/api/auth/me`.
- `src/pages/` — Login, Register, Dashboard.
- `src/features/sources/Sidebar.tsx` — левая панель источников + upload.
- `src/features/analysis/AnalysisView.tsx` — рендер AnalysisResult:
  StatusBadge, AuditTable, DetectedColumns, ChartsGrid, WarningsList.

## Ограничения MVP

- Нет реальных charts-heatmap (расширим при включении HMM).
- Нет UI-редактора column mapping — это задача `TASK_SPEC_003`.
- Анализ сейчас синхронный: для больших файлов UI блокируется на время POST.
- Нет i18n-переключателя: UI только на русском.
- ESLint-конфиг минимален, тесты UI пока не написаны.
