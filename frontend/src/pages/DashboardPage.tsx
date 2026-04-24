import React, { useState } from "react";
import { Sidebar } from "@/features/sources/Sidebar";
import { AnalysisView } from "@/features/analysis/AnalysisView";

export const DashboardPage: React.FC = () => {
  const [selectedId, setSelectedId] = useState<number | null>(null);

  return (
    <div className="h-full flex">
      <Sidebar selectedId={selectedId} onSelect={setSelectedId} />
      <main className="flex-1 overflow-y-auto">
        {selectedId === null ? (
          <div className="h-full flex items-center justify-center text-brand-700/60 px-6 text-center">
            <div className="max-w-md">
              <div className="text-lg font-medium text-brand-900">
                Выберите источник слева
              </div>
              <p className="mt-2 text-sm">
                Загрузите Excel-файл с данными эпизодов соревновательной
                деятельности. Алгоритм выполнит честный аудит и, при
                достаточности данных, рассчитает базовые распределения.
              </p>
              <p className="mt-4 text-xs text-brand-700/60">
                Наблюдения = ЗАП. Скрытые состояния описывают соревновательную
                деятельность (маневрирование → КФВ → ВУП). Никакой диагностики
                без подтверждённого сопоставления колонок.
              </p>
            </div>
          </div>
        ) : (
          <AnalysisView sourceId={selectedId} />
        )}
      </main>
    </div>
  );
};
