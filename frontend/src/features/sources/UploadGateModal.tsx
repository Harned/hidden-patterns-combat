import React, { useState } from "react";
import { Button } from "@/components/ui";
import { UPLOAD_GATE } from "@/copy/legal";

interface Props {
  onCancel: () => void;
  /**
   * Вызывается после подтверждения чекбокса. Дальше сайдбар открывает
   * системный диалог выбора файла и сам запускает загрузку.
   */
  onConfirm: () => void;
}

export const UploadGateModal: React.FC<Props> = ({ onCancel, onConfirm }) => {
  const [accepted, setAccepted] = useState(false);

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-brand-900/40 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <p className="text-xs uppercase tracking-wide text-brand-700/60">
          Загрузка источника
        </p>
        <h2 className="mt-1 text-lg font-semibold text-brand-900">
          {UPLOAD_GATE.title}
        </h2>
        <p className="mt-2 text-sm text-brand-700/80">
          Сначала подтвердите условия — затем выберете файл.
        </p>

        <ul className="mt-4 space-y-2 text-sm text-brand-900/90">
          {UPLOAD_GATE.notes.map((note, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-brand-500">•</span>
              <span>{note}</span>
            </li>
          ))}
        </ul>

        <label className="mt-5 flex items-start gap-2 text-sm text-brand-900/90">
          <input
            type="checkbox"
            className="mt-1"
            checked={accepted}
            onChange={(e) => setAccepted(e.target.checked)}
          />
          <span>{UPLOAD_GATE.checkboxLabel}</span>
        </label>

        <div className="mt-6 flex justify-end gap-2">
          <Button variant="ghost" onClick={onCancel}>
            Отмена
          </Button>
          <Button onClick={onConfirm} disabled={!accepted}>
            Выбрать файл
          </Button>
        </div>
      </div>
    </div>
  );
};
