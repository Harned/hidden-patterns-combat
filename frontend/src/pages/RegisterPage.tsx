import React, { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import { Button, Card, Input, Label } from "@/components/ui";

export const RegisterPage: React.FC = () => {
  const { user, register, loading } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) return <Navigate to="/app" replace />;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password.length < 8) {
      setError("Пароль должен содержать минимум 8 символов.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await register(email, password);
      navigate("/app");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка регистрации");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-full flex items-center justify-center p-6">
      <Card className="w-full max-w-md p-8">
        <h1 className="text-xl font-semibold text-brand-900">Регистрация</h1>
        <p className="mt-1 text-sm text-brand-700/80">
          Создайте аккаунт, чтобы загрузить Excel и запустить honest-анализ.
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
          {error && (
            <div className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          <Button type="submit" disabled={submitting || loading} className="w-full">
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
