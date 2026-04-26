import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/api/client";
import type { SourceSummary } from "@/api/types";
import { Badge, Button } from "@/components/ui";
import { useAuth } from "@/auth/AuthContext";
import { UploadGateModal } from "./UploadGateModal";
import { DisclaimerBanner } from "@/features/onboarding/DisclaimerBanner";

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

const initialFor = (email: string | undefined): string => {
  if (!email) return "·";
  const ch = email.trim().charAt(0).toUpperCase();
  return ch || "·";
};

export const Sidebar: React.FC<Props> = ({ selectedId, onSelect }) => {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { user } = useAuth();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadGateOpen, setUploadGateOpen] = useState(false);
  const [openMenuId, setOpenMenuId] = useState<number | null>(null);

  const { data, isLoading } = useQuery<SourceSummary[]>({
    queryKey: ["sources"],
    queryFn: api.listSources,
  });

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadSource(file, true),
    onSuccess: (created) => {
      void qc.invalidateQueries({ queryKey: ["sources"] });
      // Только что загруженный источник всегда draft — отправляем в мастер
      // предобработки. После finalize пользователь вернётся в /app.
      navigate(`/sources/${created.id}/prepare`);
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

  // Шаг 1: пользователь подтвердил условия в модалке -> открываем системный
  // диалог выбора файла. Шаг 2: onChange у input — собственно загрузка.
  const onGateConfirmed = () => {
    setUploadGateOpen(false);
    fileInputRef.current?.click();
  };

  const onFileChosen = (f: File | null) => {
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (!f) return;
    upload.mutate(f);
  };

  return (
    <aside className="h-full w-80 shrink-0 border-r border-brand-200 bg-white flex flex-col">
      <div className="px-5 py-4 border-b border-brand-100">
        <Link
          to="/profile"
          className="flex items-center gap-3 rounded-md -mx-1 px-1 py-1 hover:bg-brand-50 focus:outline-none focus:ring-2 focus:ring-brand-300"
          aria-label="Профиль"
        >
          <span
            aria-hidden
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand-100 text-brand-700 text-sm font-semibold"
          >
            {initialFor(user?.email)}
          </span>
          <span className="min-w-0 flex-1">
            <span
              className="block truncate text-sm font-medium text-brand-900"
              title={user?.email}
            >
              {user?.email}
            </span>
            <span className="mt-0.5 block">
              {user?.email_verified_at ? (
                <Badge tone="success">подтверждён</Badge>
              ) : (
                <Badge tone="warning">не подтверждён</Badge>
              )}
            </span>
          </span>
        </Link>
      </div>

      <div className="px-5 pt-4 pb-2">
        <div className="text-base font-semibold text-brand-900">Источники</div>
      </div>

      <div className="px-5 pb-3">
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls"
          className="hidden"
          onChange={(e) => onFileChosen(e.target.files?.[0] ?? null)}
        />
        <Button
          onClick={() => setUploadGateOpen(true)}
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
          const isDraft = s.preparation_state === "draft";
          return (
            <div
              key={s.id}
              className={`relative rounded-md flex items-center ${
                active ? "bg-brand-100" : "hover:bg-brand-50"
              }`}
            >
              <button
                onClick={() => {
                  if (isDraft) {
                    navigate(`/sources/${s.id}/prepare`);
                    return;
                  }
                  onSelect(s.id);
                }}
                className="flex-1 min-w-0 text-left px-3 py-2"
              >
                <div
                  className={`text-sm font-medium truncate ${
                    active ? "text-brand-900" : "text-brand-800"
                  }`}
                  title={s.original_filename}
                >
                  {s.original_filename}
                </div>
                <div className="mt-1 flex items-center gap-2 flex-wrap">
                  {isDraft ? (
                    <Badge tone="warning">черновик</Badge>
                  ) : (
                    <Badge tone={statusTone(s.last_analysis_status)}>
                      {s.last_analysis_status ?? "не анализирован"}
                    </Badge>
                  )}
                  <span className="text-[11px] text-brand-700/60">
                    {(s.size_bytes / 1024).toFixed(1)} KB
                  </span>
                </div>
              </button>
              <div className="relative shrink-0 pr-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenMenuId(menuOpen ? null : s.id);
                  }}
                  className="rounded px-1.5 py-0.5 text-brand-700/70 hover:text-brand-900 hover:bg-brand-100"
                  aria-label="Меню источника"
                  title="Меню источника"
                >
                  ⋯
                </button>
                {menuOpen && (
                  <div
                    className="absolute right-0 top-full mt-1 z-10 rounded-md border border-brand-200 bg-white shadow-lg w-52 py-1 text-sm"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {isDraft ? (
                      <button
                        className="w-full text-left px-3 py-2 hover:bg-brand-50"
                        onClick={() => {
                          navigate(`/sources/${s.id}/prepare`);
                          setOpenMenuId(null);
                        }}
                      >
                        Продолжить подготовку
                      </button>
                    ) : (
                      <button
                        className="w-full text-left px-3 py-2 hover:bg-brand-50"
                        onClick={() => {
                          onSelect(s.id);
                          setOpenMenuId(null);
                        }}
                      >
                        Открыть
                      </button>
                    )}
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
            </div>
          );
        })}
      </div>

      <DisclaimerBanner variant="footer" />

      {uploadGateOpen && (
        <UploadGateModal
          onCancel={() => setUploadGateOpen(false)}
          onConfirm={onGateConfirmed}
        />
      )}
    </aside>
  );
};
