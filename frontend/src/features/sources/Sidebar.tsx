import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type { SourceSummary } from "@/api/types";
import { Badge, Button } from "@/components/ui";
import { useAuth } from "@/auth/AuthContext";
import { UploadGateModal } from "./UploadGateModal";

interface Props {
  selectedId: number | null;
  onSelect: (id: number | null) => void;
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
  const { user } = useAuth();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [openMenuId, setOpenMenuId] = useState<number | null>(null);

  const { data, isLoading } = useQuery<SourceSummary[]>({
    queryKey: ["sources"],
    queryFn: api.listSources,
  });

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadSource(file, true),
    onSuccess: (created) => {
      void qc.invalidateQueries({ queryKey: ["sources"] });
      onSelect(created.id);
      setPendingFile(null);
    },
  });

  const remove = useMutation({
    mutationFn: (id: number) => api.deleteSource(id),
    onSuccess: (_data, id) => {
      void qc.invalidateQueries({ queryKey: ["sources"] });
      if (selectedId === id) onSelect(null);
      setOpenMenuId(null);
    },
  });

  // Закрываем меню «⋯» по клику вне.
  useEffect(() => {
    if (openMenuId === null) return;
    const onClick = () => setOpenMenuId(null);
    window.addEventListener("click", onClick);
    return () => window.removeEventListener("click", onClick);
  }, [openMenuId]);

  const handleFile = (f: File | null) => {
    if (!f) return;
    setPendingFile(f);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <aside className="h-full w-80 shrink-0 border-r border-brand-200 bg-white flex flex-col">
      <div className="px-5 py-4 border-b border-brand-100">
        <div className="text-base font-semibold text-brand-900">Источники</div>
        <div className="mt-0.5 flex items-center gap-2 text-xs text-brand-700/70">
          <span className="truncate" title={user?.email}>
            {user?.email}
          </span>
          {user?.email_verified_at ? (
            <Badge tone="success">подтверждён</Badge>
          ) : (
            <Badge tone="warning">не подтверждён</Badge>
          )}
        </div>
        <div className="mt-2">
          <Link
            to="/profile"
            className="text-xs text-brand-600 hover:text-brand-700"
          >
            Перейти в профиль →
          </Link>
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
          Принимаются .xlsx / .xls. Макросы (.xlsm) запрещены. Перед
          загрузкой подтвердите, что данные обезличены.
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
          const menuOpen = openMenuId === s.id;
          return (
            <div
              key={s.id}
              className={`relative rounded-md ${
                active ? "bg-brand-100" : "hover:bg-brand-50"
              }`}
            >
              <button
                onClick={() => onSelect(s.id)}
                className="w-full text-left px-3 py-2 pr-10"
              >
                <div
                  className={`text-sm font-medium truncate ${
                    active ? "text-brand-900" : "text-brand-800"
                  }`}
                  title={s.original_filename}
                >
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
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setOpenMenuId(menuOpen ? null : s.id);
                }}
                className="absolute top-2 right-2 rounded px-1.5 text-brand-700/70 hover:text-brand-900 hover:bg-brand-100"
                aria-label="Меню источника"
                title="Меню источника"
              >
                ⋯
              </button>
              {menuOpen && (
                <div
                  className="absolute right-2 top-9 z-10 rounded-md border border-brand-200 bg-white shadow-lg w-44 py-1 text-sm"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    className="w-full text-left px-3 py-2 hover:bg-brand-50"
                    onClick={() => {
                      onSelect(s.id);
                      setOpenMenuId(null);
                    }}
                  >
                    Открыть
                  </button>
                  <button
                    className="w-full text-left px-3 py-2 text-red-700 hover:bg-red-50"
                    onClick={() => {
                      if (
                        confirm(
                          `Удалить источник «${s.original_filename}» и все связанные результаты?`
                        )
                      ) {
                        remove.mutate(s.id);
                      }
                    }}
                    disabled={remove.isPending}
                  >
                    Удалить файл и результаты
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="border-t border-brand-100 px-5 py-3 text-xs text-brand-700/70">
        Управление аккаунтом —{" "}
        <button
          onClick={() => navigate("/profile")}
          className="text-brand-600 hover:text-brand-700 underline"
        >
          в профиле
        </button>
        .
      </div>

      {pendingFile && (
        <UploadGateModal
          fileName={pendingFile.name}
          onCancel={() => setPendingFile(null)}
          onConfirm={() => upload.mutate(pendingFile)}
        />
      )}
    </aside>
  );
};
