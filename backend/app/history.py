from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

from redis import Redis


class HistoryStore(Protocol):
    backend: str

    def load(self, session_id: str, limit: int = 12) -> list[dict[str, str]]: ...

    def append(self, session_id: str, role: str, content: str) -> None: ...

    def ping(self) -> bool: ...


class ChatHistoryStore:
    backend = "sqlite"

    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def load(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def append(self, session_id: str, role: str, content: str) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            conn.commit()

    def ping(self) -> bool:
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def _init_db(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
            conn.commit()


class RedisChatHistoryStore:
    backend = "redis"

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str,
        ttl_seconds: int,
        max_messages: int,
        client: Redis | None = None,
    ) -> None:
        self.client = client or Redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = key_prefix.rstrip(":")
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    def load(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        values = self.client.lrange(self._key(session_id), -limit, -1)
        return [json.loads(value) for value in values]

    def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        value = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        with self.client.pipeline() as pipeline:
            pipeline.rpush(key, value)
            pipeline.ltrim(key, -self.max_messages, -1)
            pipeline.expire(key, self.ttl_seconds)
            pipeline.execute()

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"
