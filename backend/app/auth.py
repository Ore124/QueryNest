from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, Protocol
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash

from .settings import Settings, get_settings


UserRole = Literal["personal", "admin"]
_password_hash = PasswordHash.recommended()
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Actor:
    user_id: UUID
    role: UserRole


class ActiveUserStore(Protocol):
    def get_active_actor(self, user_id: str) -> dict[str, str] | None: ...


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must not be empty")
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    return _password_hash.verify(password, password_hash)


def create_access_token(
    actor: Actor, settings: Settings, *, expires_delta: timedelta | None = None
) -> str:
    expires_at = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.auth_jwt_expiry_minutes)
    )
    return jwt.encode(
        {
            "sub": str(actor.user_id),
            "role": actor.role,
            "iss": settings.auth_jwt_issuer,
            "iat": datetime.now(timezone.utc),
            "exp": expires_at,
        },
        settings.auth_jwt_secret,
        algorithm="HS256",
    )


def decode_access_token(token: str, settings: Settings) -> Actor:
    try:
        claims = jwt.decode(
            token,
            settings.auth_jwt_secret,
            algorithms=["HS256"],
            issuer=settings.auth_jwt_issuer,
            options={"require": ["sub", "role", "iss", "exp"]},
        )
        role = claims["role"]
        if role not in {"personal", "admin"}:
            raise ValueError("invalid role")
        return Actor(user_id=UUID(claims["sub"]), role=role)
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
        raise _credentials_exception() from exc


def get_current_actor(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
    user_store: ActiveUserStore | None = None,
) -> Actor:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _credentials_exception()
    token_actor = decode_access_token(credentials.credentials, settings)
    if user_store is None:
        return token_actor
    current = user_store.get_active_actor(str(token_actor.user_id))
    if current is None or current["role"] not in {"personal", "admin"}:
        raise _credentials_exception()
    return Actor(user_id=UUID(current["user_id"]), role=current["role"])


def require_admin(
    actor: Annotated[Actor, Depends(get_current_actor)],
) -> Actor:
    if actor.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access is required.")
    return actor


def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
