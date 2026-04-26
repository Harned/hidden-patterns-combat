import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "@/auth/AuthContext";
import { LoginPage } from "@/pages/LoginPage";
import { RegisterPage } from "@/pages/RegisterPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { VerifyEmailPage } from "@/pages/VerifyEmailPage";
import { ForgotPasswordPage } from "@/pages/ForgotPasswordPage";
import { ResetPasswordPage } from "@/pages/ResetPasswordPage";
import { ProfilePage } from "@/pages/ProfilePage";
import { LegalPage } from "@/pages/LegalPage";
import { SourcePrepPage } from "@/pages/SourcePrepPage";
import { OnboardingModal } from "@/features/onboarding/OnboardingModal";

const Loader: React.FC = () => (
  <div className="h-full flex items-center justify-center text-brand-700/70">
    Проверяем сессию...
  </div>
);

const RequireUser: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();
  if (loading) return <Loader />;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
};

/** Гард для рабочей области: пользователь должен быть подтверждён. */
const RequireVerified: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();
  if (loading) return <Loader />;
  if (!user) return <Navigate to="/login" replace />;
  if (!user.email_verified_at) return <Navigate to="/verify-email" replace />;
  return <>{children}</>;
};

/** Layout для рабочих экранов — onboarding. Баннер тестового режима
 * рендерится внутри сайдбара (см. `Sidebar`). */
const WorkLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="h-full flex flex-col">
    <div className="flex-1 min-h-0">{children}</div>
    <OnboardingModal />
  </div>
);

export const App: React.FC = () => (
  <AuthProvider>
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route path="/legal/:slug" element={<LegalPage />} />

      <Route
        path="/verify-email"
        element={
          <RequireUser>
            <VerifyEmailPage />
          </RequireUser>
        }
      />

      <Route
        path="/profile"
        element={
          <RequireUser>
            <WorkLayout>
              <ProfilePage />
            </WorkLayout>
          </RequireUser>
        }
      />

      <Route
        path="/app"
        element={
          <RequireVerified>
            <WorkLayout>
              <DashboardPage />
            </WorkLayout>
          </RequireVerified>
        }
      />
      <Route
        path="/sources/:sourceId/prepare"
        element={
          <RequireVerified>
            <SourcePrepPage />
          </RequireVerified>
        }
      />
      <Route path="*" element={<Navigate to="/app" replace />} />
    </Routes>
  </AuthProvider>
);
