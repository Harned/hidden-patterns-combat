import React, { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, ApiError } from "@/api/client";
import { Button, Card, Input, Label } from "@/components/ui";

const RESET_AUTOSEND_KEY = "hpc_pwdreset_autosent";

function shouldDedupePwdResetAutosend(email: string): boolean {
  const key = `${RESET_AUTOSEND_KEY}_${email.toLowerCase()}`;
  try {
    const last = sessionStorage.getItem(key);
    const now = Date.now();
    if (last && now - Number(last) < 2500) return true;
    sessionStorage.setItem(key, String(now));
  } catch {
    /* ignore */
  }
  return false;
}

export const ResetPasswordPage: React.FC = () => {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [email, setEmail] = useState(params.get("email") ?? "");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [repeat, setRepeat] = useState("");
  const [resending, setResending] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const emailFromUrl = (params.get("email") ?? "").trim();

  useEffect(() => {
    if (!emailFromUrl) return;
    if (shouldDedupePwdResetAutosend(emailFromUrl)) return;
    setResending(true);
    setError(null);
    void (async () => {
      try {
        const resp = await api.forgotPassword(emailFromUrl);
        setInfo(resp.message);
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
  }, [emailFromUrl]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setInfo(null);
    if (password !== repeat) {
      setError("Пароли не совпадают.");
      return;
    }
    setSubmitting(true);
    try {
      await api.resetPassword(email, code.trim(), password, repeat);
      setInfo("Пароль изменён. Сейчас перенаправим на вход.");
      setTimeout(() => navigate("/login"), 1200);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Не удалось обновить пароль"
      );
    } finally {
      setSubmitting(false);
    }
  };

  const resend = async () => {
    if (!email) {
      setError("Сначала укажите email.");
      return;
    }
    setError(null);
    setInfo(null);
    setResending(true);
    try {
      const resp = await api.forgotPassword(email);
      setInfo(resp.message);
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
          Восстановление пароля
        </p>
        <h1 className="mt-1 text-xl font-semibold text-brand-900">
          Введите код восстановления
        </h1>
        <p className="mt-2 text-sm text-brand-700/80">
          Это код <b>восстановления</b> пароля; не путайте с кодом
          подтверждения регистрации. Если в ссылке есть email, запрос кода
          делаем при открытии страницы; кнопка ниже — повторная отправка.
        </p>

        <form onSubmit={submit} className="mt-6 space-y-4">
          <div>
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div>
            <Label htmlFor="code">Код восстановления</Label>
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
          <div>
            <Label htmlFor="new-password">Новый пароль</Label>
            <Input
              id="new-password"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
            />
          </div>
          <div>
            <Label htmlFor="new-password-repeat">Повторите пароль</Label>
            <Input
              id="new-password-repeat"
              type="password"
              autoComplete="new-password"
              value={repeat}
              onChange={(e) => setRepeat(e.target.value)}
              required
              minLength={8}
            />
          </div>

          {error && (
            <div className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          {info && (
            <div className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
              {info}
            </div>
          )}

          <Button type="submit" disabled={submitting} className="w-full">
            {submitting ? "Сохраняем..." : "Сохранить новый пароль"}
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
          <Link to="/login" className="text-brand-700/70 hover:text-brand-900">
            Войти
          </Link>
        </div>
      </Card>
    </div>
  );
};
