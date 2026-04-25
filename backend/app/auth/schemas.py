"""Pydantic-схемы auth."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    # LEGAL-REG-1: оба согласия обязательны.
    accept_terms: bool
    accept_pdn: bool

    @field_validator("accept_terms", "accept_pdn")
    @classmethod
    def _must_be_true(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("Требуется согласие.")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class VerifyEmailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=4, max_length=10)


class ForgotPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    code: str = Field(..., min_length=4, max_length=10)
    new_password: str = Field(..., min_length=8, max_length=128)
    new_password_repeat: str = Field(..., min_length=8, max_length=128)


class UserPublic(BaseModel):
    """Публичный профиль текущего пользователя.

    Поля, относящиеся к согласиям и онбордингу, нужны фронту, чтобы:
    * показывать статус подтверждения email на странице профиля;
    * правильно роутить на экран ввода кода / онбординга.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    created_at: datetime
    email_verified_at: datetime | None = None
    terms_accepted_at: datetime | None = None
    pdn_accepted_at: datetime | None = None
    onboarding_completed_at: datetime | None = None
    csrf_token: str | None = None
