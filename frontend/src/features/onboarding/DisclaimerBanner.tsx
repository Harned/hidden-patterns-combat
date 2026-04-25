import React from "react";
import { Link } from "react-router-dom";
import { BRAND } from "@/copy/legal";

export const DisclaimerBanner: React.FC = () => (
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
