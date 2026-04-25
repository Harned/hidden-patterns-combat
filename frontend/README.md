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

## Маршруты

| Путь | Назначение |
|------|------------|
| `/login` | вход (ссылка на `/forgot-password`, `/register`) |
| `/register` | регистрация с двумя обязательными согласиями (LEGAL-REG-1) |
| `/verify-email` | ввод кода подтверждения, кнопка «Отправить код снова» |
| `/forgot-password` | запрос кода восстановления (нейтральное сообщение) |
| `/reset-password` | ввод кода + новый пароль |
| `/legal/:slug` | три документа-плейсхолдера (`terms` / `privacy` / `pdn-consent`) |
| `/profile` | статус email, документы, выход, удаление аккаунта (PROFILE-1) |
| `/app` | рабочая область (только подтверждённый email + onboarding) |

## Ограничения MVP

- Анализ синхронный/фоновый, но UI пока опрашивает `/runs/{id}` без WebSocket.
- Нет i18n-переключателя: UI только на русском.
- Юридические тексты в `src/copy/legal.ts` — плейсхолдеры; реальные документы предоставляет владелец.
- ESLint-конфиг минимальный; для smoke-проверки используется Vitest.
