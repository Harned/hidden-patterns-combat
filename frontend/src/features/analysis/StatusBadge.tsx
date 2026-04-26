import React from "react";
import type { AnalysisStatus } from "@/api/types";
import { Badge } from "@/components/ui";

const labels: Record<AnalysisStatus, string> = {
  audit_only: "Только аудит",
  baseline_only: "Baseline",
  needs_column_mapping: "Нужен column mapping",
  hmm_ready: "HMM готов",
  hmm_low_signal: "HMM (низкий сигнал ZAP)",
  failed: "Ошибка",
};

const tones: Record<
  AnalysisStatus,
  "neutral" | "success" | "warning" | "danger" | "info"
> = {
  audit_only: "neutral",
  baseline_only: "info",
  needs_column_mapping: "warning",
  hmm_ready: "success",
  hmm_low_signal: "warning",
  failed: "danger",
};

export const StatusBadge: React.FC<{ status: AnalysisStatus }> = ({ status }) => (
  <Badge tone={tones[status]}>{labels[status]}</Badge>
);
