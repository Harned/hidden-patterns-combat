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
    one_time_code_for_dev_log: str | None = None,
) -> None:
    """Записать письмо в выбранный sink. Безопасно: исключения подавляются
    и попадают в лог — потеря dev-почты не должна ронять регистрацию."""

    timestamp = datetime.now(UTC).isoformat()
    # WARNING: подробный блок (иногда в docker не виден у дочерних логгеров).
    logger.warning(
        "\n%s\n[DEV MAIL | %s] → %s\n%s\n%s\n%s",
        "=" * 60,
        kind,
        to_email,
        subject,
        body,
        "=" * 60,
    )
    # Одна строка с самим кодом: дочерний `app.mail` + многострочные WARNING
    # часто не видны в консоли uvicorn / из‑за `HPC_DEBUG=false` старый
    # однострочник не вызывался. Пишем в root + uvicorn.error, без gate на
    # debug — при `mail_sink=log|file` реальной почты всё равно нет.
    if one_time_code_for_dev_log is not None and settings.mail_sink in (
        "log",
        "file",
    ):
        line = (
            f"hpc dev-mail: kind={kind} to={to_email} "
            f"код={one_time_code_for_dev_log}"
        )
        logging.getLogger().warning("%s", line)
        logging.getLogger("uvicorn.error").warning("%s", line)
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
        one_time_code_for_dev_log=code,
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
        one_time_code_for_dev_log=code,
    )
