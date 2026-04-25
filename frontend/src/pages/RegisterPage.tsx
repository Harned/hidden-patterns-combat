import React, { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import { Button, Card, Input, Label } from "@/components/ui";
import { BRAND, REGISTRATION_CONSENTS } from "@/copy/legal";

export const RegisterPage: React.FC = () => {
  const { user, register, loading } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [acceptTerms, setAcceptTerms] = useState(false);
  const [acceptPdn, setAcceptPdn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (user && user.email_verified_at) return <Navigate to="/app" replace />;
  if (user && !user.email_verified_at)
    return <Navigate to="/verify-email" replace />;

  const ready = acceptTerms && acceptPdn && password.length >= 8;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ready) return;
    setSubmitting(true);
    setError(null);
    try {
      await register(email, password, acceptTerms, acceptPdn);
      navigate("/verify-email");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка регистрации");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-full flex items-center justify-center p-6">
      <Card className="w-full max-w-md p-8">
        <p className="text-xs uppercase tracking-wide text-brand-700/60">
          {BRAND.tagline}
        </p>
        <h1 className="mt-1 text-xl font-semibold text-brand-900">
          Регистрация
        </h1>
        <p className="mt-2 text-sm text-brand-700/80">
          Создайте аккаунт, чтобы загрузить Excel и запустить honest-анализ.
          После регистрации мы отправим код подтверждения на email.
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
            <Label htmlFor="password">Пароль</Label>
            <Input
              id="password"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
            />
            <p className="mt-1 text-xs text-brand-700/70">Минимум 8 символов.</p>
          </div>

          <fieldset className="space-y-3 border border-brand-100 rounded-md p-3 bg-brand-50/40">
            <legend className="px-1 text-xs uppercase tracking-wide text-brand-700/70">
              Обязательные согласия
            </legend>
            <label className="flex items-start gap-2 text-sm text-brand-900/90">
              <input
                type="checkbox"
                className="mt-1"
                checked={acceptTerms}
                onChange={(e) => setAcceptTerms(e.target.checked)}
              />
              <span>
                {REGISTRATION_CONSENTS.terms}{" "}
                <Link
                  to="/legal/terms"
                  target="_blank"
                  className="text-brand-600 hover:text-brand-700"
                >
                  (читать)
                </Link>
              </span>
            </label>
            <label className="flex items-start gap-2 text-sm text-brand-900/90">
              <input
                type="checkbox"
                className="mt-1"
                checked={acceptPdn}
                onChange={(e) => setAcceptPdn(e.target.checked)}
              />
              <span>
                {REGISTRATION_CONSENTS.pdn}{" "}
                <Link
                  to="/legal/pdn-consent"
                  target="_blank"
                  className="text-brand-600 hover:text-brand-700"
                >
                  (Согласие)
                </Link>{" "}
                <Link
                  to="/legal/privacy"
                  target="_blank"
                  className="text-brand-600 hover:text-brand-700"
                >
                  (Политика)
                </Link>
              </span>
            </label>
          </fieldset>

          {error && (
            <div className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}

          <Button
            type="submit"
            disabled={!ready || submitting || loading}
            className="w-full"
          >
            {submitting ? "Создаём..." : "Создать аккаунт"}
          </Button>
        </form>

        <p className="mt-6 text-sm text-brand-700/80">
          Уже есть аккаунт?{" "}
          <Link to="/login" className="text-brand-600 hover:text-brand-700 font-medium">
            Войти
          </Link>
        </p>
      </Card>
    </div>
  );
};
