"""Локальная dev-доставка email (TASK_SPEC_010).

Реальная SMTP-отправка не входит в эпик. Здесь — компактный sink,
который:

* всегда логирует факт отправки в structured logger;
* при ``mail_sink == "file"`` дублирует письмо в
  ``storage/devmail/<email>__<timestamp>.txt`` — это удобный способ
  вытащить код в локальном dev/QA без подключения почтового сервера.

Сообщения короткие и предметные. Никаких HTML/шаблонов: задача —
доставить пользователю код для проверки потока.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings


logger = logging.getLogger(__name__)


def _safe_filename(email: str) -> str:
    return email.replace("/", "_").replace("\\", "_").replace("@", "_at_")


def send_mail(
    settings: Settings,
    to_email: str,
    subject: str,
    body: str,
    *,
    kind: str = "info",
) -> None:
    """Записать письмо в выбранный sink. Безопасно: исключения подавляются
    и попадают в лог — потеря dev-почты не должна ронять регистрацию."""

    timestamp = datetime.now(UTC).isoformat()
    logger.info(
        "[mail/%s] to=%s subject=%r\n%s",
        kind,
        to_email,
        subject,
        body,
    )
    if settings.mail_sink == "file":
        try:
            settings.mail_dir.mkdir(parents=True, exist_ok=True)
            path: Path = settings.mail_dir / (
                f"{_safe_filename(to_email)}__{timestamp.replace(':', '-')}.txt"
            )
            path.write_text(
                f"To: {to_email}\nSubject: {subject}\nKind: {kind}\nDate: {timestamp}\n\n{body}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to persist dev-mail to %s: %s", settings.mail_dir, exc)


def send_email_verification_code(
    settings: Settings, to_email: str, code: str
) -> None:
    send_mail(
        settings,
        to_email=to_email,
        subject="Подтверждение email — hidden-patterns-combat",
        body=(
            "Код подтверждения email: " + code + "\n\n"
            "Введите его на экране подтверждения, чтобы завершить регистрацию.\n"
            "Если вы не регистрировались — просто проигнорируйте письмо."
        ),
        kind="email_verification",
    )


def send_password_reset_code(
    settings: Settings, to_email: str, code: str
) -> None:
    send_mail(
        settings,
        to_email=to_email,
        subject="Восстановление пароля — hidden-patterns-combat",
        body=(
            "Код для восстановления пароля: " + code + "\n\n"
            "Введите его на экране восстановления и задайте новый пароль.\n"
            "Если вы не запрашивали восстановление — проигнорируйте письмо."
        ),
        kind="password_reset",
    )
