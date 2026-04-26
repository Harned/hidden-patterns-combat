import React, { useEffect, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { api, ApiError } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { Button, Card, Input, Label } from "@/components/ui";

const VERIFY_AUTOSEND_KEY = "hpc_verify_autosent";

function shouldDedupeAutosend(userId: number): boolean {
  try {
    const key = `${VERIFY_AUTOSEND_KEY}_${userId}`;
    const last = sessionStorage.getItem(key);
    const now = Date.now();
    if (last && now - Number(last) < 2500) return true;
    sessionStorage.setItem(key, String(now));
  } catch {
    /* ignore */
  }
  return false;
}

export const VerifyEmailPage: React.FC = () => {
  const { user, setUser, logout } = useAuth();
  const navigate = useNavigate();
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resending, setResending] = useState(false);

  useEffect(() => {
    if (user == null || user.email_verified_at) return;
    if (shouldDedupeAutosend(user.id)) return;
    setError(null);
    setResending(true);
    void (async () => {
      try {
        await api.resendVerification();
        setInfo("Код отправлен. Проверьте письмо (или dev-логи).");
      } catch (err) {
        setError(
          err instanceof ApiError
            ? err.message
            : "Не удалось автоматически отправить код"
        );
      } finally {
        setResending(false);
      }
    })();
  }, [user?.id, user?.email_verified_at]);

  if (!user) return <Navigate to="/login" replace />;
  if (user.email_verified_at) return <Navigate to="/app" replace />;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setInfo(null);
    setSubmitting(true);
    try {
      const updated = await api.verifyEmail(code.trim());
      setUser(updated);
      navigate("/app");
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Не удалось подтвердить email"
      );
    } finally {
      setSubmitting(false);
    }
  };

  const resend = async () => {
    setError(null);
    setInfo(null);
    setResending(true);
    try {
      await api.resendVerification();
      setInfo("Новый код отправлен. Проверьте письмо (или dev-логи).");
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Не удалось отправить код"
      );
    } finally {
      setResending(false);
    }
  };

  return (
    <div className="min-h-full flex items-center justify-center p-6">
      <Card className="w-full max-w-md p-8">
        <p className="text-xs uppercase tracking-wide text-brand-700/60">
          Подтверждение email
        </p>
        <h1 className="mt-1 text-xl font-semibold text-brand-900">
          Введите код подтверждения
        </h1>
        <p className="mt-2 text-sm text-brand-700/80">
          Код подтверждения на адрес{" "}
          <span className="font-medium text-brand-900">{user.email}</span>{" "}
          запрашивается при открытии этой страницы. При необходимости можно
          выслать ещё раз — кнопка ниже.
        </p>

        <form onSubmit={submit} className="mt-6 space-y-4">
          <div>
            <Label htmlFor="code">Код из письма</Label>
            <Input
              id="code"
              type="text"
              inputMode="numeric"
              autoComplete="one-time-code"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              required
              minLength={4}
              maxLength={10}
            />
          </div>

          {error && (
            <div className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          {info && (
            <div className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
              {info}
            </div>
          )}

          <Button
            type="submit"
            disabled={submitting || code.length < 4}
            className="w-full"
          >
            {submitting ? "Подтверждаем..." : "Подтвердить"}
          </Button>
        </form>

        <div className="mt-4 flex items-center justify-between text-sm">
          <button
            onClick={resend}
            disabled={resending}
            className="text-brand-600 hover:text-brand-700"
          >
            {resending ? "Отправляем..." : "Отправить код снова"}
          </button>
          <button
            onClick={async () => {
              await logout();
              navigate("/login");
            }}
            className="text-brand-700/70 hover:text-brand-900"
          >
            Выйти
          </button>
        </div>
      </Card>
    </div>
  );
};
