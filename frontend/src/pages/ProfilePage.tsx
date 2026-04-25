import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { Badge, Button, Card, Section } from "@/components/ui";

const fmt = (value: string | null | undefined): string =>
  value ? new Date(value).toLocaleString("ru-RU") : "—";

export const ProfilePage: React.FC = () => {
  const { user, refresh, logout, setUser } = useAuth();
  const navigate = useNavigate();
  const [resending, setResending] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!user) {
    return (
      <div className="min-h-full flex items-center justify-center p-6">
        <Card className="px-6 py-8 max-w-md w-full text-center">
          <p className="text-sm text-brand-700/80">
            Сначала войдите.{" "}
            <Link
              to="/login"
              className="text-brand-600 hover:text-brand-700 underline"
            >
              Перейти ко входу
            </Link>
            .
          </p>
        </Card>
      </div>
    );
  }

  const resend = async () => {
    setError(null);
    setInfo(null);
    setResending(true);
    try {
      await api.resendVerification();
      setInfo("Новый код подтверждения отправлен.");
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Не удалось отправить код"
      );
    } finally {
      setResending(false);
    }
  };

  const onDelete = async () => {
    if (
      !confirm(
        "Удалить аккаунт? Это удалит все ваши источники и результаты анализа. Действие необратимо."
      )
    )
      return;
    setError(null);
    setDeleting(true);
    try {
      await api.deleteAccount();
      setUser(null);
      navigate("/login");
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Не удалось удалить аккаунт"
      );
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="min-h-full">
      <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-brand-900">Профиль</h1>
          <Link
            to="/app"
            className="text-sm text-brand-600 hover:text-brand-700"
          >
            ← В рабочую область
          </Link>
        </div>

        <Section title="Аккаунт">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-xs text-brand-700/70">Email</div>
              <div className="font-medium text-brand-900">{user.email}</div>
            </div>
            <div>
              <div className="text-xs text-brand-700/70">Создан</div>
              <div className="text-brand-900">{fmt(user.created_at)}</div>
            </div>
            <div>
              <div className="text-xs text-brand-700/70">Email подтверждён</div>
              <div className="flex items-center gap-2">
                {user.email_verified_at ? (
                  <>
                    <Badge tone="success">подтверждён</Badge>
                    <span className="text-brand-700/70">
                      {fmt(user.email_verified_at)}
                    </span>
                  </>
                ) : (
                  <>
                    <Badge tone="warning">не подтверждён</Badge>
                    <button
                      onClick={resend}
                      disabled={resending}
                      className="text-brand-600 hover:text-brand-700 underline"
                    >
                      {resending ? "Отправляем..." : "Отправить код снова"}
                    </button>
                    <Link
                      to="/verify-email"
                      className="text-brand-600 hover:text-brand-700 underline"
                    >
                      Ввести код
                    </Link>
                  </>
                )}
              </div>
            </div>
            <div>
              <div className="text-xs text-brand-700/70">Дисклеймер принят</div>
              <div className="text-brand-900">
                {fmt(user.onboarding_completed_at)}
              </div>
            </div>
            <div>
              <div className="text-xs text-brand-700/70">Условия приняты</div>
              <div className="text-brand-900">{fmt(user.terms_accepted_at)}</div>
            </div>
            <div>
              <div className="text-xs text-brand-700/70">
                Согласие на обработку ПД
              </div>
              <div className="text-brand-900">{fmt(user.pdn_accepted_at)}</div>
            </div>
          </div>
          {(error || info) && (
            <div className="mt-4">
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
            </div>
          )}
        </Section>

        <Section title="Документы">
          <ul className="text-sm text-brand-900/90 space-y-1">
            <li>
              <Link
                to="/legal/terms"
                target="_blank"
                className="text-brand-600 hover:text-brand-700"
              >
                Условия использования исследовательского стенда
              </Link>
            </li>
            <li>
              <Link
                to="/legal/privacy"
                target="_blank"
                className="text-brand-600 hover:text-brand-700"
              >
                Политика обработки персональных данных
              </Link>
            </li>
            <li>
              <Link
                to="/legal/pdn-consent"
                target="_blank"
                className="text-brand-600 hover:text-brand-700"
              >
                Согласие на обработку персональных данных
              </Link>
            </li>
          </ul>
        </Section>

        <Section
          title="Управление аккаунтом"
          description="Кнопки выхода и удаления аккаунта расположены здесь, чтобы случайно их не задеть в рабочей области."
        >
          <div className="flex flex-wrap gap-3">
            <Button
              variant="secondary"
              onClick={async () => {
                await logout();
                await refresh();
                navigate("/login");
              }}
            >
              Выйти из аккаунта
            </Button>
            <Button variant="danger" onClick={onDelete} disabled={deleting}>
              {deleting ? "Удаляем..." : "Удалить аккаунт"}
            </Button>
          </div>
        </Section>
      </div>
    </div>
  );
};
