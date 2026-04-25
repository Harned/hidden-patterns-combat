import React from "react";
import { Link, useParams } from "react-router-dom";
import { Card } from "@/components/ui";
import { DOCS, type LegalDocId } from "@/copy/legal";

const ALIASES: Record<string, LegalDocId> = {
  terms: "terms",
  privacy: "privacy",
  "pdn-consent": "pdnConsent",
};

export const LegalPage: React.FC = () => {
  const { slug } = useParams<{ slug: string }>();
  const docKey = slug ? ALIASES[slug] : undefined;
  const doc = docKey ? DOCS[docKey] : undefined;

  if (!doc) {
    return (
      <div className="min-h-full flex items-center justify-center p-6">
        <Card className="max-w-2xl w-full p-8 text-center">
          <h1 className="text-xl font-semibold text-brand-900">
            Документ не найден
          </h1>
          <p className="mt-2 text-sm text-brand-700/80">
            Доступные документы:
          </p>
          <ul className="mt-3 text-sm text-brand-700">
            <li>
              <Link className="text-brand-600 hover:text-brand-700" to="/legal/terms">
                Условия использования
              </Link>
            </li>
            <li>
              <Link className="text-brand-600 hover:text-brand-700" to="/legal/privacy">
                Политика обработки ПД
              </Link>
            </li>
            <li>
              <Link className="text-brand-600 hover:text-brand-700" to="/legal/pdn-consent">
                Согласие на обработку ПД
              </Link>
            </li>
          </ul>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-full p-6 max-w-3xl mx-auto">
      <Card className="p-8">
        <p className="text-xs uppercase tracking-wide text-brand-700/60">
          Документ исследовательского стенда (плейсхолдер)
        </p>
        <h1 className="mt-1 text-2xl font-semibold text-brand-900">
          {doc.title}
        </h1>
        <div className="mt-6 space-y-5">
          {doc.sections.map((s) => (
            <section key={s.heading}>
              <h2 className="text-base font-semibold text-brand-900">
                {s.heading}
              </h2>
              <p className="mt-1 text-sm text-brand-900/85 leading-relaxed">
                {s.body}
              </p>
            </section>
          ))}
        </div>
        <p className="mt-8 text-xs text-brand-700/60">
          Этот текст — плейсхолдер. Полный документ предоставляет владелец
          продукта; UI и навигация уже подготовлены.
        </p>
        <div className="mt-6">
          <Link to="/" className="text-sm text-brand-600 hover:text-brand-700">
            ← Вернуться
          </Link>
        </div>
      </Card>
    </div>
  );
};
