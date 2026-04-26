import React from "react";
import { Link } from "react-router-dom";
import { BRAND } from "@/copy/legal";

interface Props {
  /**
   * `bar` — широкая горизонтальная плашка во всю ширину экрана.
   * `footer` — компактный блок для футера сайдбара: вертикальный лейаут,
   *   мелкий текст, ссылки в одну строку.
   */
  variant?: "bar" | "footer";
}

export const DisclaimerBanner: React.FC<Props> = ({ variant = "bar" }) => {
  if (variant === "footer") {
    return (
      <div className="border-t border-amber-200 bg-amber-50 text-amber-900 px-4 py-2 text-[11px] leading-snug">
        <div className="flex items-center gap-2">
          <span className="uppercase tracking-wide font-semibold">
            тестовый режим
          </span>
        </div>
        <p className="mt-1 text-amber-900/90">{BRAND.banner}</p>
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
          <Link to="/legal/terms" className="underline hover:text-amber-700">
            Условия
          </Link>
          <Link to="/legal/privacy" className="underline hover:text-amber-700">
            Политика
          </Link>
          <Link
            to="/legal/pdn-consent"
            className="underline hover:text-amber-700"
          >
            Согласие
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="border-b border-amber-200 bg-amber-50 text-amber-900 px-4 py-1.5 text-xs flex flex-wrap items-center gap-2">
      <span className="uppercase tracking-wide font-semibold">тестовый режим</span>
      <span>{BRAND.banner}</span>
      <span className="ml-auto flex gap-3">
        <Link to="/legal/terms" className="underline hover:text-amber-700">
          Условия
        </Link>
        <Link to="/legal/privacy" className="underline hover:text-amber-700">
          Политика
        </Link>
        <Link to="/legal/pdn-consent" className="underline hover:text-amber-700">
          Согласие
        </Link>
      </span>
    </div>
  );
};
