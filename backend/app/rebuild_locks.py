from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from .kb_metadata import PostgresMetadataStore


class KbRebuildLockBusy(RuntimeError):
    pass


class KbRebuildLock(Protocol):
    backend: str

    def acquire(self, *, kb_id: str, worker_id: str, timeout_seconds: int) -> AbstractContextManager[None]: ...


@dataclass(frozen=True)
class RedisKbRebuildLock:
    client: object
    key_prefix: str = "rag:rebuild-lock"
    backend: str = "redis"

    def acquire(self, *, kb_id: str, worker_id: str, timeout_seconds: int) -> AbstractContextManager[None]:
        key = f"{self.key_prefix}:{kb_id}"
        lock = self.client.lock(
            key,
            timeout=timeout_seconds,
            blocking=False,
            thread_local=False,
        )
        return _RedisLockContext(lock=lock, kb_id=kb_id, worker_id=worker_id)


@dataclass(frozen=True)
class PostgresAdvisoryKbRebuildLock:
    store: PostgresMetadataStore
    backend: str = "postgresql-advisory"

    def acquire(self, *, kb_id: str, worker_id: str, timeout_seconds: int) -> AbstractContextManager[None]:
        del timeout_seconds
        key = _advisory_lock_key(kb_id)
        return _PostgresAdvisoryLockContext(store=self.store, lock_key=key, kb_id=kb_id, worker_id=worker_id)


@dataclass
class _RedisLockContext(AbstractContextManager[None]):
    lock: object
    kb_id: str
    worker_id: str

    def __enter__(self) -> None:
        acquired = self.lock.acquire()
        if not acquired:
            raise KbRebuildLockBusy(f"KB rebuild already running for kb_id={self.kb_id}")
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.lock.release()
        except Exception:
            pass
        return False


@dataclass
class _PostgresAdvisoryLockContext(AbstractContextManager[None]):
    store: PostgresMetadataStore
    lock_key: int
    kb_id: str
    worker_id: str

    def __enter__(self) -> None:
        acquired = self.store.try_advisory_lock(self.lock_key)
        if not acquired:
            raise KbRebuildLockBusy(f"KB rebuild already running for kb_id={self.kb_id}")
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.store.release_advisory_lock(self.lock_key)
        return False


def _advisory_lock_key(kb_id: str) -> int:
    digest = hashlib.sha256(kb_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)
