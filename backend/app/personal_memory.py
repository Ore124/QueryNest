from __future__ import annotations

import uuid
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from .auth import Actor

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover
    @dataclass(frozen=True)
    class SystemMessage:  # type: ignore[no-redef]
        content: str

    @dataclass(frozen=True)
    class HumanMessage:  # type: ignore[no-redef]
        content: str


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
PERSONAL_MEMORY_TYPES = ("preference", "profile", "fact")
MAX_PERSONAL_MEMORIES_PER_TURN = 5
MAX_PERSONAL_MEMORY_KEY_CHARS = 120
MAX_PERSONAL_MEMORY_VALUE_CHARS = 500
MAX_PERSONAL_MEMORY_EVIDENCE_CHARS = 500
MAX_PERSONAL_MEMORY_TTL_DAYS = 365


class PersonalMemoryStoreUnavailable(RuntimeError):
    pass


class PersonalMemoryAccessDenied(PermissionError):
    pass


class PersonalMemoryIndex(Protocol):
    """Derived-index contract. PostgreSQL remains authoritative if this fails."""

    def upsert(self, memory: dict[str, Any]) -> None: ...

    def delete(self, memory_id: str) -> None: ...


class NoopPersonalMemoryIndex:
    def upsert(self, memory: dict[str, Any]) -> None:
        return None

    def delete(self, memory_id: str) -> None:
        return None


class InMemoryPersonalMemoryIndex:
    """Small test/dev implementation of the same derived-index contract."""

    def __init__(self) -> None:
        self.memories: dict[str, dict[str, Any]] = {}

    def upsert(self, memory: dict[str, Any]) -> None:
        self.memories[str(memory["memory_id"])] = dict(memory)

    def delete(self, memory_id: str) -> None:
        self.memories.pop(memory_id, None)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        memory = self.memories.get(memory_id)
        return dict(memory) if memory is not None else None


@dataclass(frozen=True)
class PersonalMemoryCandidate:
    memory_type: str
    key: str
    value: str
    confidence: float
    ttl_days: int | None
    evidence: str


@dataclass(frozen=True)
class PersonalMemoryWriteResult:
    extraction_accepted: bool
    source_saved: int
    indexed: int
    index_pending: int
    pending_memory_ids: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class PersonalMemoryDeleteResult:
    source_deleted: bool
    index_pending: bool
    memory_id: str


class PersonalMemoryService:
    """Best-effort extraction with a strict, user-authored personal-memory boundary."""

    def __init__(
        self,
        store: Any,
        chat_model: Any,
        *,
        index: PersonalMemoryIndex | None = None,
        default_ttl_days: int = 90,
        extraction_max_items: int = MAX_PERSONAL_MEMORIES_PER_TURN,
    ) -> None:
        self.store = store
        self.chat_model = chat_model
        self.index = index or NoopPersonalMemoryIndex()
        self.default_ttl_days = min(max(default_ttl_days, 1), MAX_PERSONAL_MEMORY_TTL_DAYS)
        self.extraction_max_items = min(max(extraction_max_items, 1), MAX_PERSONAL_MEMORIES_PER_TURN)

    def record_user_message(self, actor: Actor, owner_user_id: str, source_session_id: str, user_message: str) -> PersonalMemoryWriteResult:
        candidates, error = self._extract(user_message)
        if candidates is None:
            return PersonalMemoryWriteResult(False, 0, 0, 0, error=error)
        saved = indexed = 0
        pending: list[str] = []
        for candidate in candidates:
            try:
                memory = self.store.create_or_update(
                    actor, owner_user_id, candidate.memory_type, candidate.key, candidate.value,
                    confidence=candidate.confidence, source_session_id=source_session_id,
                    expires_at=datetime.now(timezone.utc) + timedelta(days=candidate.ttl_days or self.default_ttl_days),
                )
            except Exception:
                return PersonalMemoryWriteResult(True, saved, indexed, len(pending), tuple(pending), "storage_unavailable")
            saved += 1
            try:
                self.index.upsert(memory)
                indexed += 1
            except Exception:
                pending.append(str(memory["memory_id"]))
        return PersonalMemoryWriteResult(True, saved, indexed, len(pending), tuple(pending))

    def delete_memory(
        self, actor: Actor, memory_id: str, expected_owner_user_id: str | None = None
    ) -> PersonalMemoryDeleteResult:
        """Delete source data first; a failed derived-index deletion is observable."""
        self.store.delete_for_actor(actor, memory_id, expected_owner_user_id)
        try:
            self.index.delete(memory_id)
            return PersonalMemoryDeleteResult(True, False, memory_id)
        except Exception:
            return PersonalMemoryDeleteResult(True, True, memory_id)

    def active_for_retrieval(
        self, actor: Actor, owner_user_id: str, *, limit: int = MAX_PERSONAL_MEMORIES_PER_TURN
    ) -> list[dict[str, Any]]:
        """Return only the actor-authorized active memories from the durable source."""
        # The database expires rows before selecting them. Keep the same boundary
        # here so a future derived store cannot reintroduce expired memories.
        now = datetime.now(timezone.utc)
        memories = self.store.active_for_retrieval(
            actor, owner_user_id, limit=min(max(limit, 1), MAX_PERSONAL_MEMORIES_PER_TURN)
        )
        return [
            memory
            for memory in memories
            if memory.get("status", "active") == "active"
            and (memory.get("expires_at") is None or memory["expires_at"] > now)
        ][:MAX_PERSONAL_MEMORIES_PER_TURN]

    def _extract(self, user_message: str) -> tuple[list[PersonalMemoryCandidate] | None, str | None]:
        if _normalize_personal_text(user_message, MAX_PERSONAL_MEMORY_VALUE_CHARS) is None:
            return None, "invalid_input"
        try:
            response = self.chat_model.invoke([SystemMessage(content=_PERSONAL_MEMORY_EXTRACTION_PROMPT), HumanMessage(content=user_message)])
            candidates = parse_personal_memory_extraction(
                getattr(response, "content", None), user_message, max_items=self.extraction_max_items
            )
        except Exception:
            return None, "extraction_unavailable"
        return (candidates, None) if candidates is not None else (None, "invalid_extraction")


_PERSONAL_MEMORY_EXTRACTION_PROMPT = (
    'Extract only explicit first-person user statements. Never infer or save another person, credentials, '
    'contact details, identity documents, health, or finances. Return exactly {"memories":[...]}; every item '
    'has memory_type, key, value, confidence, and evidence; ttl_days is optional. memory_type is preference, profile, '
    'or fact; evidence is the exact user statement; confidence is 0 through 1; ttl_days, when set, is 1 through 365.'
)


def parse_personal_memory_extraction(
    content: Any, user_message: str, *, max_items: int = MAX_PERSONAL_MEMORIES_PER_TURN
) -> list[PersonalMemoryCandidate] | None:
    """Parse only the exact schema and reject unsafe or non-user-authored candidates."""
    if not isinstance(content, str):
        return None
    normalized_user = _normalize_personal_text(user_message, MAX_PERSONAL_MEMORY_VALUE_CHARS)
    if normalized_user is None:
        return None
    try:
        raw = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(raw, dict) or set(raw) != {"memories"} or not isinstance(raw["memories"], list):
        return None
    entries = raw["memories"]
    if not 1 <= max_items <= MAX_PERSONAL_MEMORIES_PER_TURN or len(entries) > max_items:
        return None
    candidates: list[PersonalMemoryCandidate] = []
    seen: set[tuple[str, str]] = set()
    required = {"memory_type", "key", "value", "confidence", "evidence"}
    allowed = required | {"ttl_days"}
    for entry in entries:
        if not isinstance(entry, dict) or not required.issubset(entry) or set(entry) - allowed:
            return None
        memory_type = entry["memory_type"]
        key = _normalize_personal_text(entry["key"], MAX_PERSONAL_MEMORY_KEY_CHARS)
        value = _normalize_personal_text(entry["value"], MAX_PERSONAL_MEMORY_VALUE_CHARS)
        evidence = _normalize_personal_text(entry["evidence"], MAX_PERSONAL_MEMORY_EVIDENCE_CHARS)
        confidence, ttl_days = entry["confidence"], entry.get("ttl_days")
        if (memory_type not in PERSONAL_MEMORY_TYPES or key is None or value is None or evidence is None
                or evidence not in normalized_user or not _is_explicit_first_person(evidence)
                or _contains_sensitive_data(f"{key} {value} {evidence}")
                or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or (ttl_days is not None and (
                    isinstance(ttl_days, bool) or not isinstance(ttl_days, int)
                    or not 1 <= ttl_days <= MAX_PERSONAL_MEMORY_TTL_DAYS
                ))):
            return None
        identity = (memory_type, key.casefold())
        if identity not in seen:
            seen.add(identity)
            candidates.append(PersonalMemoryCandidate(memory_type, key, value, float(confidence), ttl_days, evidence))
    return candidates


def _normalize_personal_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", " ", value).strip()
    return normalized if normalized and len(normalized) <= limit else None


def _is_explicit_first_person(evidence: str) -> bool:
    has_first_person = bool(re.search(r"(?:\b(?:i|me|my|mine)\b|我|本人)", evidence, re.IGNORECASE))
    refers_to_other = bool(re.search(r"(?:\b(?:he|she|they|their|his|her)\b|他|她|它|他们|她们|朋友|家人|同事)", evidence, re.IGNORECASE))
    return has_first_person and not refers_to_other


def _contains_sensitive_data(value: str) -> bool:
    return bool(re.search(
        r"(?:password|passwd|token|secret|credential|密码|密钥|令牌|phone|telephone|email|contact|手机号|电话|邮箱|联系方式|passport|id[ _-]?number|身份证|护照|health|medical|diagnos|病历|健康|诊断|bank|card number|credit card|income|财务|银行卡|信用卡|收入)",
        value, re.IGNORECASE,
    ))


class PostgresPersonalMemoryStore:
    """PostgreSQL source of truth for governed, user-scoped long-term memory."""

    backend = "postgresql"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def run_migrations(self) -> None:
        with self._connect() as conn:
            conn.execute((MIGRATIONS_DIR / "009_personal_memories.sql").read_text(encoding="utf-8"))

    def create_or_update(
        self,
        actor: Actor,
        owner_user_id: str,
        memory_type: str,
        memory_key: str,
        memory_value: str,
        *,
        confidence: float = 1.0,
        source_session_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            is_cross_owner = self._authorize(conn, actor, owner_user_id, collection=False)
            self._expire_due(conn, owner_user_id)
            if source_session_id is not None:
                source = conn.execute(
                    "SELECT owner_user_id FROM conversations WHERE session_id = %s",
                    (source_session_id,),
                ).fetchone()
                if source is None or str(source[0]) != owner_user_id:
                    raise PersonalMemoryAccessDenied("source session is not owned by memory owner")
            row = conn.execute(
                """
                INSERT INTO personal_memories(
                    memory_id, owner_user_id, memory_type, memory_key, memory_value,
                    confidence, source_session_id, expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (owner_user_id, memory_type, memory_key) WHERE status = 'active'
                DO UPDATE SET
                    memory_value = EXCLUDED.memory_value,
                    confidence = EXCLUDED.confidence,
                    source_session_id = EXCLUDED.source_session_id,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = now()
                RETURNING memory_id, owner_user_id, memory_type, memory_key, memory_value,
                    confidence, status, source_session_id, expires_at
                """,
                (
                    str(uuid.uuid4()),
                    owner_user_id,
                    memory_type,
                    memory_key,
                    memory_value,
                    confidence,
                    source_session_id,
                    expires_at,
                ),
            ).fetchone()
            if is_cross_owner:
                self._audit(conn, actor, "personal_memory.write.cross_owner", owner_user_id, source_session_id)
        return self._row_to_memory(row)

    def list_for_actor(self, actor: Actor, owner_user_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            try:
                is_cross_owner = self._authorize(conn, actor, owner_user_id, collection=True)
            except PersonalMemoryAccessDenied:
                return []
            self._expire_due(conn, owner_user_id)
            rows = conn.execute(
                """
                SELECT memory_id, owner_user_id, memory_type, memory_key, memory_value,
                    confidence, status, source_session_id, expires_at
                FROM personal_memories
                WHERE owner_user_id = %s AND status = 'active'
                ORDER BY updated_at DESC, memory_id
                """,
                (owner_user_id,),
            ).fetchall()
            if is_cross_owner:
                self._audit(conn, actor, "personal_memory.read.cross_owner", owner_user_id)
        return [self._row_to_memory(row) for row in rows]

    def get_for_actor(self, actor: Actor, memory_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = self._find_memory(conn, memory_id)
            if row is None:
                raise PersonalMemoryAccessDenied("personal memory access is denied")
            is_cross_owner = self._authorize(conn, actor, str(row[1]), collection=False)
            self._expire_due(conn, str(row[1]))
            row = self._find_memory(conn, memory_id)
            if row is None:
                raise PersonalMemoryAccessDenied("personal memory access is denied")
            if is_cross_owner:
                self._audit(conn, actor, "personal_memory.read.cross_owner", str(row[1]))
        return self._row_to_memory(row)

    def delete_for_actor(
        self, actor: Actor, memory_id: str, expected_owner_user_id: str | None = None
    ) -> None:
        with self._connect() as conn:
            row = self._find_memory(conn, memory_id)
            if row is None or row[6] != "active":
                raise PersonalMemoryAccessDenied("personal memory access is denied")
            if expected_owner_user_id is not None and str(row[1]) != expected_owner_user_id:
                raise PersonalMemoryAccessDenied("personal memory access is denied")
            is_cross_owner = self._authorize(conn, actor, str(row[1]), collection=False)
            conn.execute(
                """
                UPDATE personal_memories
                SET status = 'deleted', updated_at = now()
                WHERE memory_id = %s AND status <> 'deleted'
                """,
                (memory_id,),
            )
            if is_cross_owner:
                self._audit(conn, actor, "personal_memory.write.cross_owner", str(row[1]))

    def update_for_actor(
        self, actor: Actor, memory_id: str, changes: dict[str, Any], expected_owner_user_id: str | None = None
    ) -> dict[str, Any]:
        """Update an active memory only; source-session provenance is immutable."""
        allowed = {"memory_type", "memory_key", "memory_value", "confidence", "expires_at"}
        if not changes or set(changes) - allowed:
            raise ValueError("invalid personal memory update")
        with self._connect() as conn:
            row = self._find_memory(conn, memory_id)
            if row is None or row[6] != "active":
                raise PersonalMemoryAccessDenied("personal memory access is denied")
            if expected_owner_user_id is not None and str(row[1]) != expected_owner_user_id:
                raise PersonalMemoryAccessDenied("personal memory access is denied")
            is_cross_owner = self._authorize(conn, actor, str(row[1]), collection=False)
            assignments = ", ".join(f"{column} = %s" for column in changes)
            updated = conn.execute(
                f"""
                UPDATE personal_memories
                SET {assignments}, updated_at = now()
                WHERE memory_id = %s AND status = 'active'
                RETURNING memory_id, owner_user_id, memory_type, memory_key, memory_value,
                    confidence, status, source_session_id, expires_at
                """,
                (*changes.values(), memory_id),
            ).fetchone()
            if updated is None:
                raise PersonalMemoryAccessDenied("personal memory access is denied")
            if is_cross_owner:
                self._audit(conn, actor, "personal_memory.write.cross_owner", str(row[1]))
        return self._row_to_memory(updated)

    def active_for_retrieval(self, actor: Actor, owner_user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            try:
                is_cross_owner = self._authorize(conn, actor, owner_user_id, collection=True)
            except PersonalMemoryAccessDenied:
                return []
            self._expire_due(conn, owner_user_id)
            rows = conn.execute(
                """
                SELECT memory_id, owner_user_id, memory_type, memory_key, memory_value,
                    confidence, status, source_session_id, expires_at
                FROM personal_memories
                WHERE owner_user_id = %s AND status = 'active'
                ORDER BY updated_at DESC, memory_id
                LIMIT %s
                """,
                (owner_user_id, limit),
            ).fetchall()
            if is_cross_owner:
                self._audit(conn, actor, "personal_memory.read.cross_owner", owner_user_id)
        return [self._row_to_memory(row) for row in rows]

    def _authorize(self, conn: Any, actor: Actor, owner_user_id: str, *, collection: bool) -> bool:
        actor_row = conn.execute(
            "SELECT role FROM users WHERE user_id = %s AND is_active = TRUE", (str(actor.user_id),)
        ).fetchone()
        owner_exists = conn.execute("SELECT 1 FROM users WHERE user_id = %s", (owner_user_id,)).fetchone()
        if actor_row is None or owner_exists is None:
            raise PersonalMemoryAccessDenied("personal memory access is denied")
        if str(actor.user_id) == owner_user_id:
            return False
        if actor_row[0] == "admin":
            return True
        if collection:
            raise PersonalMemoryAccessDenied("personal memory collection is empty")
        raise PersonalMemoryAccessDenied("personal memory access is denied")

    @staticmethod
    def _expire_due(conn: Any, owner_user_id: str) -> None:
        conn.execute(
            """
            UPDATE personal_memories
            SET status = 'expired', updated_at = now()
            WHERE owner_user_id = %s AND status = 'active'
                AND expires_at IS NOT NULL AND expires_at <= now()
            """,
            (owner_user_id,),
        )

    @staticmethod
    def _audit(conn: Any, actor: Actor, action: str, owner_user_id: str, session_id: str | None = None) -> None:
        conn.execute(
            """
            INSERT INTO audit_events(actor_user_id, action, session_id, owner_user_id)
            VALUES (%s, %s, %s, %s)
            """,
            (str(actor.user_id), action, session_id, owner_user_id),
        )

    @staticmethod
    def _find_memory(conn: Any, memory_id: str) -> Any | None:
        return conn.execute(
            """
            SELECT memory_id, owner_user_id, memory_type, memory_key, memory_value,
                confidence, status, source_session_id, expires_at
            FROM personal_memories WHERE memory_id = %s
            """,
            (memory_id,),
        ).fetchone()

    @staticmethod
    def _row_to_memory(row: Any) -> dict[str, Any]:
        return {
            "memory_id": str(row[0]),
            "owner_user_id": str(row[1]),
            "memory_type": row[2],
            "key": row[3],
            "value": row[4],
            "confidence": row[5],
            "status": row[6],
            "source_session_id": str(row[7]) if row[7] is not None else None,
            "expires_at": row[8],
        }

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise PersonalMemoryStoreUnavailable(
                "psycopg is required when PostgreSQL personal memory is configured."
            ) from exc
        return psycopg.connect(self.dsn)
