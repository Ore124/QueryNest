from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from redis import Redis


class JsonCache(Protocol):
    backend: str

    def get_json(self, namespace: str, payload: dict[str, Any]) -> Any | None: ...

    def set_json(
        self,
        namespace: str,
        payload: dict[str, Any],
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None: ...

    def ping(self) -> bool: ...


class NullJsonCache:
    backend = "disabled"

    def get_json(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        return None

    def set_json(
        self,
        namespace: str,
        payload: dict[str, Any],
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        return None

    def ping(self) -> bool:
        return False


class RedisJsonCache:
    backend = "redis"

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str,
        default_ttl_seconds: int,
        client: Redis | None = None,
    ) -> None:
        self.client = client or Redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = key_prefix.rstrip(":")
        self.default_ttl_seconds = default_ttl_seconds

    def get_json(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        try:
            value = self.client.get(self._key(namespace, payload))
            return json.loads(value) if value else None
        except Exception:
            return None

    def set_json(
        self,
        namespace: str,
        payload: dict[str, Any],
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        try:
            self.client.set(
                self._key(namespace, payload),
                json.dumps(value, ensure_ascii=False),
                ex=ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds,
            )
        except Exception:
            return None

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def _key(self, namespace: str, payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:{namespace}:{digest}"
