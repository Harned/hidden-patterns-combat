import React from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md";
};

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  size = "md",
  className = "",
  ...rest
}) => {
  const base =
    "inline-flex items-center justify-center rounded-md font-medium transition-colors " +
    "disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-1";
  const sizes = { sm: "h-8 px-3 text-sm", md: "h-10 px-4 text-sm" };
  const variants = {
    primary: "bg-brand-500 text-white hover:bg-brand-600",
    secondary: "bg-brand-100 text-brand-900 hover:bg-brand-200",
    danger: "bg-red-500 text-white hover:bg-red-600",
    ghost: "bg-transparent text-brand-700 hover:bg-brand-100",
  };
  return (
    <button
      className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}
      {...rest}
    />
  );
};

type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Input: React.FC<InputProps> = ({ className = "", ...rest }) => (
  <input
    className={`h-10 w-full rounded-md border border-brand-200 bg-white px-3 text-sm placeholder:text-brand-300 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 ${className}`}
    {...rest}
  />
);

export const Card: React.FC<{ className?: string; children: React.ReactNode }> = ({
  className = "",
  children,
}) => (
  <div className={`rounded-lg border border-brand-200 bg-white shadow-sm ${className}`}>
    {children}
  </div>
);

export const Label: React.FC<{ htmlFor?: string; children: React.ReactNode }> = ({
  htmlFor,
  children,
}) => (
  <label htmlFor={htmlFor} className="block text-sm font-medium text-brand-900 mb-1">
    {children}
  </label>
);

export const Badge: React.FC<{
  tone?: "neutral" | "success" | "warning" | "danger" | "info";
  children: React.ReactNode;
}> = ({ tone = "neutral", children }) => {
  const tones = {
    neutral: "bg-brand-100 text-brand-900",
    success: "bg-emerald-100 text-emerald-900",
    warning: "bg-amber-100 text-amber-900",
    danger: "bg-red-100 text-red-900",
    info: "bg-sky-100 text-sky-900",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${tones[tone]}`}
    >
      {children}
    </span>
  );
};

export const Section: React.FC<{
  title: string;
  description?: string;
  children: React.ReactNode;
}> = ({ title, description, children }) => (
  <Card className="p-5">
    <h2 className="text-base font-semibold text-brand-900">{title}</h2>
    {description && (
      <p className="mt-1 text-sm text-brand-700/80">{description}</p>
    )}
    <div className="mt-4">{children}</div>
  </Card>
);
