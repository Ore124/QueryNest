from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - only used by minimal unit-test environments
    @dataclass(frozen=True)
    class SystemMessage:  # type: ignore[no-redef]
        content: str

    @dataclass(frozen=True)
    class HumanMessage:  # type: ignore[no-redef]
        content: str


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
MEMORY_TYPES = ("task", "constraint", "entity", "open_question")
MAX_FACTS_PER_TYPE = 4
MAX_FACTS_PER_TURN = 12
MAX_FACT_KEY_CHARS = 120
MAX_FACT_VALUE_CHARS = 500
DEFAULT_SUMMARY_BUDGET_CHARS = 1_200


class SessionMemoryStoreUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RecordTurnResult:
    """Best-effort memory update outcome, safe to emit in application telemetry."""

    success: bool
    summary_saved: bool
    facts_saved: int
    extraction_accepted: bool
    error: str | None = None


@dataclass(frozen=True)
class MemoryFact:
    memory_type: str
    key: str
    value: str
    confidence: float


class SessionMemoryService:
    """Build and persist bounded, session-scoped working memory best-effort."""

    def __init__(
        self,
        store: Any,
        chat_model: Any,
        *,
        summary_budget_chars: int = DEFAULT_SUMMARY_BUDGET_CHARS,
    ) -> None:
        if summary_budget_chars <= 0:
            raise ValueError("summary_budget_chars must be positive")
        self.store = store
        self.chat_model = chat_model
        self.summary_budget_chars = summary_budget_chars

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        prior_context: str | dict[str, Any] | None,
    ) -> RecordTurnResult:
        """Record a completed turn without allowing memory failures to fail chat."""
        session = _normalize_text(session_id, limit=128)
        user = _normalize_text(user_message, limit=MAX_FACT_VALUE_CHARS)
        assistant = _normalize_text(assistant_message, limit=MAX_FACT_VALUE_CHARS)
        if session is None or user is None or assistant is None:
            return RecordTurnResult(False, False, 0, False, "invalid_input")

        facts, extraction_error = self._extract_facts(
            user_message=user,
            assistant_message=assistant,
            prior_context=prior_context,
        )
        summary = build_session_summary(
            prior_context,
            user_message=user,
            assistant_message=assistant,
            facts=facts,
            budget_chars=self.summary_budget_chars,
        )
        try:
            self.store.save_summary(session, summary)
        except Exception:
            return RecordTurnResult(False, False, 0, bool(facts), "storage_unavailable")

        facts_saved = 0
        try:
            for sort_order, fact in enumerate(facts):
                self.store.upsert_fact(
                    session,
                    fact.key,
                    fact.value,
                    memory_type=fact.memory_type,
                    confidence=fact.confidence,
                    sort_order=sort_order,
                )
                facts_saved += 1
        except Exception:
            return RecordTurnResult(False, True, facts_saved, bool(facts), "storage_unavailable")

        return RecordTurnResult(
            True,
            True,
            facts_saved,
            extraction_error is None,
            extraction_error,
        )

    def load_context(self, session_id: str) -> dict[str, Any]:
        """Return only durable current-session memory; failures degrade to empty context."""
        try:
            summary = self.store.load_latest_summary(session_id)
            facts = self.store.load_active_facts(session_id)
            content = summary.get("content", "") if isinstance(summary, dict) else ""
            return {
                "summary": content if isinstance(content, str) else "",
                "facts": facts if isinstance(facts, list) else [],
            }
        except Exception:
            return {"summary": "", "facts": []}

    def _extract_facts(
        self,
        *,
        user_message: str,
        assistant_message: str,
        prior_context: str | dict[str, Any] | None,
    ) -> tuple[list[MemoryFact], str | None]:
        try:
            response = self.chat_model.invoke(
                [
                    SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Prior context:\n{_context_to_text(prior_context)}\n\n"
                            f"User message:\n{user_message}\n\n"
                            f"Assistant message:\n{assistant_message}"
                        )
                    ),
                ]
            )
            facts = parse_memory_extraction(getattr(response, "content", None))
        except Exception:
            return [], "extraction_unavailable"
        if facts is None:
            return [], "invalid_extraction"
        return facts, None


_EXTRACTION_SYSTEM_PROMPT = """Extract durable working memory from one completed chat turn.
Return only one JSON object, with exactly these four keys: task, constraint, entity,
open_question. Each key maps to an array of zero or more objects with exactly key,
value, confidence fields. key and value must be non-empty strings. confidence must be
a JSON number from 0 through 1. Do not infer facts that are not explicit in the turn."""


def parse_memory_extraction(content: Any) -> list[MemoryFact] | None:
    """Strictly parse the model's JSON response; invalid payloads produce no facts."""
    if not isinstance(content, str):
        return None
    try:
        raw = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(raw, dict) or set(raw) != set(MEMORY_TYPES):
        return None

    facts: list[MemoryFact] = []
    seen: set[tuple[str, str]] = set()
    for memory_type in MEMORY_TYPES:
        entries = raw[memory_type]
        if not isinstance(entries, list) or len(entries) > MAX_FACTS_PER_TYPE:
            return None
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"key", "value", "confidence"}:
                return None
            key = _normalize_text(entry["key"], limit=MAX_FACT_KEY_CHARS)
            value = _normalize_text(entry["value"], limit=MAX_FACT_VALUE_CHARS)
            confidence = entry["confidence"]
            if (
                key is None
                or value is None
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                return None
            identity = (memory_type, key.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            facts.append(MemoryFact(memory_type, key, value, float(confidence)))
            if len(facts) > MAX_FACTS_PER_TURN:
                return None
    return facts


def build_session_summary(
    prior_context: str | dict[str, Any] | None,
    *,
    user_message: str,
    assistant_message: str,
    facts: list[MemoryFact],
    budget_chars: int,
) -> str:
    """Create a deterministic character-bounded summary suitable for prompt context."""
    if budget_chars <= 0:
        raise ValueError("budget_chars must be positive")
    sections: list[str] = []
    prior = _context_to_text(prior_context)
    if prior:
        sections.append(f"Prior: {prior}")
    sections.extend((f"User: {user_message}", f"Assistant: {assistant_message}"))
    if facts:
        fact_text = "; ".join(
            f"{fact.memory_type}:{fact.key}={fact.value}" for fact in facts
        )
        sections.append(f"Working memory: {fact_text}")
    return "\n".join(sections)[:budget_chars]


def _context_to_text(context: str | dict[str, Any] | None) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return _normalize_text(context, limit=DEFAULT_SUMMARY_BUDGET_CHARS) or ""
    if isinstance(context, dict):
        try:
            return _normalize_text(
                json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                limit=DEFAULT_SUMMARY_BUDGET_CHARS,
            ) or ""
        except (TypeError, ValueError):
            return ""
    return ""


def _normalize_text(value: Any, *, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized or len(normalized) > limit:
        return None
    return normalized


class PostgresSessionMemoryStore:
    """Persist session summaries and the current structured working memory."""

    backend = "postgresql"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def run_migrations(self) -> None:
        migration = MIGRATIONS_DIR / "005_session_memory.sql"
        with self._connect() as conn:
            conn.execute(migration.read_text(encoding="utf-8"))

    def save_summary(self, session_id: str, content: str) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(session_id)
                VALUES (%s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id,),
            )
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
            row = conn.execute(
                """
                INSERT INTO session_summaries(session_id, version, content)
                SELECT %s, COALESCE(MAX(version), 0) + 1, %s
                FROM session_summaries
                WHERE session_id = %s
                RETURNING version
                """,
                (session_id, content, session_id),
            ).fetchone()
        return row[0]

    def load_latest_summary(self, session_id: str) -> dict[str, int | str] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT version, content
                FROM session_summaries
                WHERE session_id = %s
                ORDER BY version DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {"version": row[0], "content": row[1]}

    def upsert_fact(
        self,
        session_id: str,
        key: str,
        value: str,
        *,
        memory_type: str,
        confidence: float,
        sort_order: int = 0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(session_id)
                VALUES (%s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id,),
            )
            conn.execute(
                """
                INSERT INTO session_working_memory_facts(
                    session_id, memory_type, fact_key, fact_value, confidence, sort_order
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id, memory_type, fact_key) WHERE is_active DO UPDATE
                SET fact_value = EXCLUDED.fact_value,
                    confidence = EXCLUDED.confidence,
                    sort_order = EXCLUDED.sort_order,
                    updated_at = now()
                """,
                (session_id, memory_type, key, value, confidence, sort_order),
            )

    def load_active_facts(self, session_id: str) -> list[dict[str, str | float]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT fact_key, fact_value, memory_type, confidence
                FROM session_working_memory_facts
                WHERE session_id = %s AND is_active
                ORDER BY sort_order, fact_id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "key": key,
                "value": value,
                "memory_type": memory_type,
                "confidence": confidence,
            }
            for key, value, memory_type, confidence in rows
        ]

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
            raise SessionMemoryStoreUnavailable(
                "psycopg is required when PostgreSQL session memory is configured."
            ) from exc
        return psycopg.connect(self.dsn)
