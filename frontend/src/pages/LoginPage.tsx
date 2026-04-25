import React, { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import { Button, Card, Input, Label } from "@/components/ui";
import { BRAND } from "@/copy/legal";

export const LoginPage: React.FC = () => {
  const { user, login, loading } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) return <Navigate to="/app" replace />;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const u = await login(email, password);
      if (!u.email_verified_at) {
        navigate("/verify-email");
      } else {
        navigate("/app");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка входа");
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
        <h1 className="mt-1 text-xl font-semibold text-brand-900">Вход</h1>
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
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
            />
          </div>
          {error && (
            <div className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          <Button type="submit" disabled={submitting || loading} className="w-full">
            {submitting ? "Входим..." : "Войти"}
          </Button>
        </form>

        <div className="mt-4 flex items-center justify-between text-sm text-brand-700/80">
          <Link
            to="/forgot-password"
            className="text-brand-600 hover:text-brand-700"
          >
            Забыли пароль?
          </Link>
          <Link
            to="/register"
            className="text-brand-600 hover:text-brand-700 font-medium"
          >
            Регистрация
          </Link>
        </div>

        <p className="mt-6 text-xs text-brand-700/70 leading-relaxed">
          {BRAND.banner}
        </p>
      </Card>
    </div>
  );
};
