import React, { useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type { SourceSummary } from "@/api/types";
import { Badge, Button } from "@/components/ui";
import { useAuth } from "@/auth/AuthContext";

interface Props {
  selectedId: number | null;
  onSelect: (id: number) => void;
}

const statusTone = (s: string | null): "neutral" | "info" | "warning" | "danger" => {
  if (!s) return "neutral";
  if (s === "baseline_only") return "info";
  if (s === "needs_column_mapping") return "warning";
  if (s === "failed") return "danger";
  return "neutral";
};

export const Sidebar: React.FC<Props> = ({ selectedId, onSelect }) => {
  const qc = useQueryClient();
  const { user, logout } = useAuth();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data, isLoading } = useQuery<SourceSummary[]>({
    queryKey: ["sources"],
    queryFn: api.listSources,
  });

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadSource(file),
    onSuccess: (created) => {
      void qc.invalidateQueries({ queryKey: ["sources"] });
      onSelect(created.id);
    },
  });

  const remove = useMutation({
    mutationFn: (id: number) => api.deleteSource(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const handleFile = (f: File | null) => {
    if (!f) return;
    upload.mutate(f);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <aside className="h-full w-80 shrink-0 border-r border-brand-200 bg-white flex flex-col">
      <div className="px-5 py-4 border-b border-brand-100">
        <div className="text-base font-semibold text-brand-900">Источники</div>
        <div className="text-xs text-brand-700/70 mt-0.5 truncate" title={user?.email}>
          {user?.email}
        </div>
      </div>

      <div className="px-5 py-3 border-b border-brand-100">
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        <Button
          onClick={() => fileInputRef.current?.click()}
          className="w-full"
          disabled={upload.isPending}
        >
          {upload.isPending ? "Загружаем..." : "Загрузить Excel"}
        </Button>
        {upload.isError && (
          <div className="mt-2 text-xs text-red-700">
            {upload.error instanceof ApiError
              ? upload.error.message
              : "Ошибка загрузки"}
          </div>
        )}
        <p className="mt-2 text-[11px] text-brand-700/60 leading-snug">
          Принимаются .xlsx / .xls. Макросы (.xlsm) запрещены.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {isLoading && (
          <div className="px-3 py-2 text-sm text-brand-700/60">Загрузка...</div>
        )}
        {!isLoading && data?.length === 0 && (
          <div className="px-3 py-2 text-sm text-brand-700/60">
            Пока нет загруженных источников.
          </div>
        )}
        {data?.map((s) => {
          const active = s.id === selectedId;
          return (
            <button
              key={s.id}
              onClick={() => onSelect(s.id)}
              className={`w-full text-left rounded-md px-3 py-2 group ${
                active
                  ? "bg-brand-100 text-brand-900"
                  : "hover:bg-brand-50 text-brand-800"
              }`}
            >
              <div className="text-sm font-medium truncate" title={s.original_filename}>
                {s.original_filename}
              </div>
              <div className="mt-1 flex items-center gap-2">
                <Badge tone={statusTone(s.last_analysis_status)}>
                  {s.last_analysis_status ?? "не анализирован"}
                </Badge>
                <span className="text-[11px] text-brand-700/60">
                  {(s.size_bytes / 1024).toFixed(1)} KB
                </span>
              </div>
            </button>
          );
        })}
      </div>

      <div className="border-t border-brand-100 px-5 py-3 flex items-center justify-between gap-2">
        {selectedId !== null && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              if (confirm("Удалить выбранный источник?")) remove.mutate(selectedId);
            }}
          >
            Удалить
          </Button>
        )}
        <Button variant="ghost" size="sm" onClick={() => void logout()}>
          Выйти
        </Button>
      </div>
    </aside>
  );
};
