import React, { useState } from "react";
import { api, ApiError } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { Button } from "@/components/ui";
import { ONBOARDING } from "@/copy/legal";

export const OnboardingModal: React.FC = () => {
  const { user, setUser } = useAuth();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Показываем только подтверждённым пользователям, ещё не прошедшим онбординг.
  if (!user) return null;
  if (!user.email_verified_at) return null;
  if (user.onboarding_completed_at) return null;

  const accept = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const updated = await api.onboardingComplete();
      setUser(updated);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Не удалось сохранить"
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-brand-900/40 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full p-6">
        <p className="text-xs uppercase tracking-wide text-brand-700/60">
          Перед началом работы
        </p>
        <h2 className="mt-1 text-lg font-semibold text-brand-900">
          {ONBOARDING.title}
        </h2>
        <ul className="mt-4 space-y-2 text-sm text-brand-900/90">
          {ONBOARDING.bullets.map((b, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-brand-500">•</span>
              <span>{b}</span>
            </li>
          ))}
        </ul>

        {error && (
          <div className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="mt-6 flex justify-end">
          <Button onClick={accept} disabled={submitting}>
            {submitting ? "Сохраняем..." : ONBOARDING.buttonLabel}
          </Button>
        </div>
      </div>
    </div>
  );
};
