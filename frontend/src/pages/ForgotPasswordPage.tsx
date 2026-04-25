import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError } from "@/api/client";
import { Button, Card, Input, Label } from "@/components/ui";

export const ForgotPasswordPage: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setMessage(null);
    setError(null);
    try {
      const resp = await api.forgotPassword(email);
      // Сообщение нейтральное; backend возвращает фразу для обоих случаев.
      setMessage(resp.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ошибка запроса");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-full flex items-center justify-center p-6">
      <Card className="w-full max-w-md p-8">
        <p className="text-xs uppercase tracking-wide text-brand-700/60">
          Восстановление пароля
        </p>
        <h1 className="mt-1 text-xl font-semibold text-brand-900">
          Запрос кода восстановления
        </h1>
        <p className="mt-2 text-sm text-brand-700/80">
          Введите email, указанный при регистрации. Мы отправим код для
          восстановления доступа.
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
          {error && (
            <div className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          {message && (
            <div className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
              {message}
            </div>
          )}
          <Button type="submit" disabled={submitting} className="w-full">
            {submitting ? "Отправляем..." : "Отправить код"}
          </Button>
          {message && (
            <Button
              type="button"
              variant="secondary"
              className="w-full"
              onClick={() =>
                navigate(`/reset-password?email=${encodeURIComponent(email)}`)
              }
            >
              Перейти к вводу кода восстановления
            </Button>
          )}
        </form>

        <p className="mt-6 text-sm text-brand-700/80">
          Вспомнили пароль?{" "}
          <Link to="/login" className="text-brand-600 hover:text-brand-700">
            Войти
          </Link>
        </p>
      </Card>
    </div>
  );
};
