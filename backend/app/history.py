from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from redis import Redis

if TYPE_CHECKING:
    from .auth import Actor


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class HistoryStoreUnavailable(RuntimeError):
    pass


class ConversationAccessDenied(PermissionError):
    pass


class UsernameAlreadyExists(ValueError):
    pass


class HistoryStore(Protocol):
    backend: str

    def load(self, session_id: str, limit: int = 12) -> list[dict[str, str]]: ...

    def append(self, session_id: str, role: str, content: str) -> None: ...

    def ping(self) -> bool: ...


class HistoryCache(Protocol):
    def load(self, session_id: str, limit: int = 12) -> list[dict[str, str]]: ...

    def append(self, session_id: str, role: str, content: str, revision: int) -> None: ...

    def replace(self, session_id: str, messages: list[dict[str, str]], revision: int) -> None: ...

    def get_revision(self, session_id: str) -> int | None: ...

    def clear(self, session_id: str) -> None: ...


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


class PostgresChatHistoryStore:
    backend = "postgresql"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def run_migrations(self) -> None:
        with self._connect() as conn:
            for name in ("004_chat_history.sql", "006_auth_ownership.sql", "007_auth_credentials.sql"):
                conn.execute((MIGRATIONS_DIR / name).read_text(encoding="utf-8"))

    def load(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY message_id DESC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def append(self, session_id: str, role: str, content: str) -> int:
        with self._connect() as conn:
            revision = conn.execute(
                """
                INSERT INTO conversations(session_id, revision)
                VALUES (%s, 1)
                ON CONFLICT (session_id) DO UPDATE
                SET revision = conversations.revision + 1, updated_at = now()
                RETURNING revision
                """,
                (session_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO chat_messages(session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        return revision

    def create_user(
        self,
        user_id: str,
        *,
        role: str = "personal",
        username: str | None = None,
        password_hash: str | None = None,
        is_active: bool = True,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users(user_id, role, username, password_hash, is_active)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    role = EXCLUDED.role,
                    username = COALESCE(EXCLUDED.username, users.username),
                    password_hash = COALESCE(EXCLUDED.password_hash, users.password_hash),
                    is_active = EXCLUDED.is_active
                """,
                (user_id, role, username, password_hash, is_active),
            )

    def bootstrap_admin(self, username: str, password_hash: str) -> str | None:
        """Create the configured administrator only for an empty user database."""
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None:
                return None
            row = conn.execute(
                """
                INSERT INTO users(user_id, role, username, password_hash, is_active)
                VALUES (%s, 'admin', %s, %s, TRUE)
                RETURNING user_id
                """,
                (str(uuid.uuid4()), username, password_hash),
            ).fetchone()
        return str(row[0])

    def create_personal_user(
        self, actor_user_id: str, username: str, password_hash: str
    ) -> dict[str, str]:
        user_id = str(uuid.uuid4())
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO users(user_id, role, username, password_hash, is_active)
                VALUES (%s, 'personal', %s, %s, TRUE)
                ON CONFLICT DO NOTHING
                RETURNING user_id, username, role
                """,
                (user_id, username, password_hash),
            ).fetchone()
            if row is None:
                raise UsernameAlreadyExists(username)
            conn.execute(
                """
                INSERT INTO audit_events(actor_user_id, action, owner_user_id)
                VALUES (%s, 'user.create.personal', %s)
                """,
                (actor_user_id, user_id),
            )
        return {"user_id": str(row[0]), "username": row[1], "role": row[2]}

    def find_user_by_username(self, username: str) -> dict[str, str | bool] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, role, password_hash, is_active
                FROM users
                WHERE username = %s
                """,
                (username,),
            ).fetchone()
        if row is None:
            return None
        return {"user_id": str(row[0]), "role": row[1], "password_hash": row[2] or "", "is_active": row[3]}

    def get_active_actor(self, user_id: str) -> dict[str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, role FROM users WHERE user_id = %s AND is_active = TRUE",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {"user_id": str(row[0]), "role": row[1]}

    def list_users_for_admin(self) -> list[dict[str, str | bool]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, username, role, is_active
                FROM users
                WHERE role = 'personal'
                ORDER BY username NULLS LAST, user_id
                """
            ).fetchall()
        return [
            {
                "user_id": str(user_id),
                "username": username or "",
                "role": role,
                "is_active": is_active,
            }
            for user_id, username, role, is_active in rows
        ]

    def list_conversations_for_admin(
        self, actor_user_id: str, owner_user_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            actor = conn.execute(
                "SELECT role FROM users WHERE user_id = %s AND is_active = TRUE", (actor_user_id,)
            ).fetchone()
            owner = conn.execute("SELECT 1 FROM users WHERE user_id = %s", (owner_user_id,)).fetchone()
            if actor is None or actor[0] != "admin" or owner is None:
                raise ConversationAccessDenied("conversation access is denied")
            if owner_user_id != actor_user_id:
                conn.execute(
                    """
                    INSERT INTO audit_events(actor_user_id, action, owner_user_id)
                    VALUES (%s, 'conversation.list.cross_owner', %s)
                    """,
                    (actor_user_id, owner_user_id),
                )
            rows = conn.execute(
                """
                SELECT c.session_id, c.owner_user_id, c.created_at, c.updated_at, COUNT(m.message_id), title.content
                FROM conversations AS c
                LEFT JOIN chat_messages AS m ON m.session_id = c.session_id
                LEFT JOIN LATERAL (
                    SELECT content
                    FROM chat_messages
                    WHERE session_id = c.session_id AND role = 'user'
                    ORDER BY message_id
                    LIMIT 1
                ) AS title ON TRUE
                WHERE c.owner_user_id = %s
                GROUP BY c.session_id, c.owner_user_id, c.created_at, c.updated_at, title.content
                ORDER BY c.updated_at DESC
                LIMIT %s
                """,
                (owner_user_id, limit),
            ).fetchall()
        return [
            {
                "session_id": str(session_id),
                "owner_user_id": str(conversation_owner_user_id),
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": message_count,
                "title": title,
            }
            for session_id, conversation_owner_user_id, created_at, updated_at, message_count, title in rows
        ]

    def list_conversations_for_user(self, actor_user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            actor = conn.execute(
                "SELECT 1 FROM users WHERE user_id = %s AND is_active = TRUE", (actor_user_id,)
            ).fetchone()
            if actor is None:
                raise ConversationAccessDenied("conversation access is denied")
            rows = conn.execute(
                """
                SELECT c.session_id, c.created_at, c.updated_at, COUNT(m.message_id), title.content
                FROM conversations AS c
                LEFT JOIN chat_messages AS m ON m.session_id = c.session_id
                LEFT JOIN LATERAL (
                    SELECT content
                    FROM chat_messages
                    WHERE session_id = c.session_id AND role = 'user'
                    ORDER BY message_id
                    LIMIT 1
                ) AS title ON TRUE
                WHERE c.owner_user_id = %s
                GROUP BY c.session_id, c.created_at, c.updated_at, title.content
                ORDER BY c.updated_at DESC
                LIMIT %s
                """,
                (actor_user_id, limit),
            ).fetchall()
        return [
            {
                "session_id": str(session_id),
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": message_count,
                "title": title,
            }
            for session_id, created_at, updated_at, message_count, title in rows
        ]

    def load_messages_for_user(
        self, actor_user_id: str, session_id: str, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            actor = conn.execute(
                "SELECT role FROM users WHERE user_id = %s AND is_active = TRUE", (actor_user_id,)
            ).fetchone()
            conversation = conn.execute(
                "SELECT owner_user_id FROM conversations WHERE session_id = %s", (session_id,)
            ).fetchone()
            if actor is None or actor[0] != "admin" or conversation is None or conversation[0] is None:
                raise ConversationAccessDenied("conversation access is denied")
            owner_user_id = str(conversation[0])
            if owner_user_id != actor_user_id:
                conn.execute(
                    """
                    INSERT INTO audit_events(actor_user_id, action, session_id, owner_user_id)
                    VALUES (%s, 'conversation.read.cross_owner', %s, %s)
                    """,
                    (actor_user_id, session_id, owner_user_id),
                )
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY message_id DESC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {"role": role, "content": content, "created_at": created_at}
            for role, content, created_at in reversed(rows)
        ]

    def load_messages_for_owner(
        self, actor_user_id: str, session_id: str, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            actor = conn.execute(
                "SELECT 1 FROM users WHERE user_id = %s AND is_active = TRUE", (actor_user_id,)
            ).fetchone()
            conversation = conn.execute(
                "SELECT owner_user_id FROM conversations WHERE session_id = %s", (session_id,)
            ).fetchone()
            if actor is None or conversation is None or conversation[0] is None or str(conversation[0]) != actor_user_id:
                raise ConversationAccessDenied("conversation access is denied")
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY message_id DESC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {"role": role, "content": content, "created_at": created_at}
            for role, content, created_at in reversed(rows)
        ]

    def delete_conversations_for_owner(self, actor_user_id: str, session_ids: list[str]) -> None:
        """Delete all requested conversations only when each belongs to the active actor."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.session_id
                FROM conversations AS c
                JOIN users AS u ON u.user_id = c.owner_user_id
                WHERE c.session_id = ANY(%s::uuid[])
                  AND c.owner_user_id = %s
                  AND u.is_active = TRUE
                FOR UPDATE
                """,
                (session_ids, actor_user_id),
            ).fetchall()
            if len(rows) != len(session_ids):
                raise ConversationAccessDenied("conversation access is denied")
            conn.execute(
                """
                DELETE FROM conversations
                WHERE session_id = ANY(%s::uuid[]) AND owner_user_id = %s
                """,
                (session_ids, actor_user_id),
            )

    def append_for_owner(self, owner_user_id: str, session_id: str, role: str, content: str) -> int:
        with self._connect() as conn:
            owner = conn.execute(
                "SELECT 1 FROM users WHERE user_id = %s", (owner_user_id,)
            ).fetchone()
            if owner is None:
                raise ConversationAccessDenied("owner user does not exist")
            conn.execute(
                """
                INSERT INTO conversations(session_id, revision, owner_user_id)
                VALUES (%s, 0, %s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, owner_user_id),
            )
            conversation_owner = conn.execute(
                "SELECT owner_user_id FROM conversations WHERE session_id = %s FOR UPDATE",
                (session_id,),
            ).fetchone()
            if conversation_owner is None or str(conversation_owner[0]) != owner_user_id:
                raise ConversationAccessDenied("conversation is not owned by this user")
            revision = conn.execute(
                """
                UPDATE conversations
                SET revision = revision + 1, updated_at = now()
                WHERE session_id = %s
                RETURNING revision
                """,
                (session_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO chat_messages(session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        return revision

    def load_for_user(
        self, actor_user_id: str, session_id: str, limit: int = 12
    ) -> list[dict[str, str]]:
        with self._connect() as conn:
            actor = conn.execute(
                "SELECT role FROM users WHERE user_id = %s", (actor_user_id,)
            ).fetchone()
            conversation = conn.execute(
                "SELECT owner_user_id FROM conversations WHERE session_id = %s", (session_id,)
            ).fetchone()
            if actor is None or conversation is None:
                raise ConversationAccessDenied("conversation access is denied")
            owner_user_id = conversation[0]
            is_admin = actor[0] == "admin"
            if owner_user_id is None or (str(owner_user_id) != actor_user_id and not is_admin):
                raise ConversationAccessDenied("conversation access is denied")
            if is_admin and str(owner_user_id) != actor_user_id:
                conn.execute(
                    """
                    INSERT INTO audit_events(actor_user_id, action, session_id, owner_user_id)
                    VALUES (%s, 'conversation.read.cross_owner', %s, %s)
                    """,
                    (actor_user_id, session_id, owner_user_id),
                )
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY message_id DESC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def load_for_actor(self, actor: "Actor", session_id: str, limit: int = 12) -> list[dict[str, str]]:
        return self.load_for_user(str(actor.user_id), session_id, limit)

    def is_owned_by_actor(self, actor: "Actor", session_id: str) -> bool:
        """Strict ownership check for user-scoped features; admins do not bypass it."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM conversations AS c
                JOIN users AS u ON u.user_id = c.owner_user_id
                WHERE c.session_id = %s AND c.owner_user_id = %s AND u.is_active = TRUE
                """,
                (session_id, str(actor.user_id)),
            ).fetchone()
        return row is not None

    def append_for_actor(self, actor: "Actor", session_id: str, role: str, content: str) -> int:
        actor_user_id = str(actor.user_id)
        with self._connect() as conn:
            actor_row = conn.execute(
                "SELECT role FROM users WHERE user_id = %s AND is_active = TRUE", (actor_user_id,)
            ).fetchone()
            if actor_row is None:
                raise ConversationAccessDenied("conversation access is denied")
            is_admin = actor_row[0] == "admin"
            conn.execute(
                """
                INSERT INTO conversations(session_id, revision, owner_user_id)
                VALUES (%s, 0, %s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, actor_user_id),
            )
            conversation = conn.execute(
                "SELECT owner_user_id FROM conversations WHERE session_id = %s FOR UPDATE", (session_id,)
            ).fetchone()
            if conversation is None or conversation[0] is None:
                raise ConversationAccessDenied("conversation access is denied")
            owner_user_id = str(conversation[0])
            if owner_user_id != actor_user_id and not is_admin:
                raise ConversationAccessDenied("conversation access is denied")
            if is_admin and owner_user_id != actor_user_id:
                conn.execute(
                    """
                    INSERT INTO audit_events(actor_user_id, action, session_id, owner_user_id)
                    VALUES (%s, 'conversation.write.cross_owner', %s, %s)
                    """,
                    (actor_user_id, session_id, owner_user_id),
                )
            revision = conn.execute(
                """
                UPDATE conversations SET revision = revision + 1, updated_at = now()
                WHERE session_id = %s RETURNING revision
                """,
                (session_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO chat_messages(session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        return revision

    def list_audit_events(self) -> list[dict[str, str | None]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT actor_user_id, action, session_id, owner_user_id
                FROM audit_events
                ORDER BY event_id
                """
            ).fetchall()
        return [
            {
                "actor_user_id": str(actor_user_id),
                "action": action,
                "session_id": str(session_id) if session_id is not None else None,
                "owner_user_id": str(owner_user_id) if owner_user_id is not None else None,
            }
            for actor_user_id, action, session_id, owner_user_id in rows
        ]

    def get_revision(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT revision FROM conversations WHERE session_id = %s", (session_id,)
            ).fetchone()
        return row[0] if row else 0

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise HistoryStoreUnavailable("psycopg is required when PostgreSQL history is configured.") from exc
        return psycopg.connect(self.dsn)


class CachedChatHistoryStore:
    """Use Redis as a best-effort recent-history cache for a durable store."""

    backend = "postgresql+redis-cache"

    def __init__(
        self,
        primary: HistoryStore,
        cache: HistoryCache,
        *,
        cache_max_messages: int,
    ) -> None:
        self.primary = primary
        self.cache = cache
        self.cache_max_messages = cache_max_messages

    def load(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        try:
            messages = self.cache.load(session_id, limit)
            cache_revision = self.cache.get_revision(session_id)
        except Exception:
            messages = []
            cache_revision = None
        primary_revision = self.primary.get_revision(session_id)
        if messages and cache_revision == primary_revision:
            return messages

        messages = self.primary.load(session_id, self.cache_max_messages)
        try:
            self.cache.replace(session_id, messages, primary_revision)
        except Exception:
            pass
        return messages[-limit:]

    def append(self, session_id: str, role: str, content: str) -> None:
        self.primary.append(session_id, role, content)
        try:
            primary_revision = self.primary.get_revision(session_id)
            cache_revision = self.cache.get_revision(session_id)
            if cache_revision is not None and cache_revision == primary_revision - 1:
                self.cache.append(session_id, role, content, primary_revision)
                return
            messages = self.primary.load(session_id, self.cache_max_messages)
            self.cache.replace(session_id, messages, primary_revision)
        except Exception:
            pass

    def load_for_actor(self, actor: "Actor", session_id: str, limit: int = 12) -> list[dict[str, str]]:
        # Authorization and its audit side effect must happen in the durable store first.
        self.primary.load_for_actor(actor, session_id, 1)
        return self.load(session_id, limit)

    def is_owned_by_actor(self, actor: "Actor", session_id: str) -> bool:
        return self.primary.is_owned_by_actor(actor, session_id)

    def append_for_actor(self, actor: "Actor", session_id: str, role: str, content: str) -> None:
        self.primary.append_for_actor(actor, session_id, role, content)
        try:
            primary_revision = self.primary.get_revision(session_id)
            messages = self.primary.load(session_id, self.cache_max_messages)
            self.cache.replace(session_id, messages, primary_revision)
        except Exception:
            pass

    def ping(self) -> bool:
        return self.primary.ping()


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

    def append(self, session_id: str, role: str, content: str, revision: int = 0) -> None:
        key = self._key(session_id)
        value = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        with self.client.pipeline() as pipeline:
            pipeline.rpush(key, value)
            pipeline.ltrim(key, -self.max_messages, -1)
            pipeline.expire(key, self.ttl_seconds)
            pipeline.set(self._revision_key(session_id), revision, ex=self.ttl_seconds)
            pipeline.execute()

    def replace(self, session_id: str, messages: list[dict[str, str]], revision: int = 0) -> None:
        self.clear(session_id)
        key = self._key(session_id)
        values = [json.dumps(message, ensure_ascii=False) for message in messages[-self.max_messages :]]
        with self.client.pipeline() as pipeline:
            if values:
                pipeline.rpush(key, *values)
                pipeline.ltrim(key, -self.max_messages, -1)
                pipeline.expire(key, self.ttl_seconds)
            pipeline.set(self._revision_key(session_id), revision, ex=self.ttl_seconds)
            pipeline.execute()

    def get_revision(self, session_id: str) -> int | None:
        value = self.client.get(self._revision_key(session_id))
        return int(value) if value is not None else None

    def clear(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))
        self.client.delete(self._revision_key(session_id))

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"

    def _revision_key(self, session_id: str) -> str:
        return f"{self._key(session_id)}:revision"
