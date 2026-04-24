import React, { createContext, useContext, useEffect, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { UserPublic } from "@/api/types";

interface AuthState {
  user: UserPublic | null;
  loading: boolean;
  error: string | null;
}

interface AuthContextValue extends AuthState {
  register: (email: string, password: string) => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AuthState>({ user: null, loading: true, error: null });

  const refresh = async () => {
    try {
      const user = await api.me();
      setState({ user, loading: false, error: null });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setState({ user: null, loading: false, error: null });
      } else {
        setState({
          user: null,
          loading: false,
          error: err instanceof Error ? err.message : "Ошибка проверки сессии",
        });
      }
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const register = async (email: string, password: string) => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const user = await api.register(email, password);
      setState({ user, loading: false, error: null });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Ошибка регистрации";
      setState({ user: null, loading: false, error: msg });
      throw err;
    }
  };

  const login = async (email: string, password: string) => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const user = await api.login(email, password);
      setState({ user, loading: false, error: null });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Ошибка входа";
      setState({ user: null, loading: false, error: msg });
      throw err;
    }
  };

  const logout = async () => {
    try {
      await api.logout();
    } finally {
      setState({ user: null, loading: false, error: null });
    }
  };

  return (
    <AuthContext.Provider value={{ ...state, register, login, logout, refresh }}>
      {children}
    </AuthContext.Provider>
  );
};

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
