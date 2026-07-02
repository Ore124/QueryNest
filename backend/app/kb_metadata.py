from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .documents import DocumentChunk, RawDocument
from .index import MILVUS_ID_FIELD, MILVUS_VECTOR_FIELD


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
SCHEMA_MIGRATIONS_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class MetadataStoreUnavailable(RuntimeError):
    pass


class ActiveIngestJobExists(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestJobState:
    job_id: str
    doc_id: str
    kb_id: str
    status: str
    worker_id: str | None
    retry_count: int
    error_message: str | None


@dataclass(frozen=True)
class IndexConsistency:
    kb_id: str
    index_version: str | None
    postgres_chunk_count: int
    milvus_vector_count: int | None
    document_statuses: dict[str, int]

    @property
    def ok(self) -> bool:
        if self.index_version is None:
            return self.postgres_chunk_count == 0
        if self.milvus_vector_count is not None and self.postgres_chunk_count != self.milvus_vector_count:
            return False
        return not any(status in self.document_statuses for status in ("failed", "parsing", "indexing"))


class PostgresMetadataStore:
    backend = "postgresql"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._advisory_lock_connections: dict[int, Any] = {}
        self._advisory_lock_connections_lock = threading.Lock()

    def run_migrations(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA_MIGRATIONS_SQL)
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                version = path.stem
                applied = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = %s",
                    (version,),
                ).fetchone()
                if applied:
                    continue
                with conn.transaction():
                    conn.execute(path.read_text(encoding="utf-8"))
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (%s) ON CONFLICT DO NOTHING",
                        (version,),
                    )

    def ensure_knowledge_base(self, kb_id: str, *, name: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_bases(kb_id, name, status)
                VALUES (%s, %s, 'active')
                ON CONFLICT (kb_id) DO UPDATE
                SET name = EXCLUDED.name,
                    updated_at = now(),
                    deleted_at = NULL
                """,
                (kb_id, name or kb_id),
            )

    def upsert_document_status(
        self,
        *,
        kb_id: str,
        doc_id: str,
        file_name: str,
        file_hash: str,
        status: str,
        parser: str,
        parser_version: str,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_bases(kb_id, name, status)
                VALUES (%s, %s, 'active')
                ON CONFLICT (kb_id) DO UPDATE
                SET updated_at = now(), deleted_at = NULL
                """,
                (kb_id, kb_id),
            )
            conn.execute(
                """
                INSERT INTO documents(
                    doc_id,
                    kb_id,
                    file_name,
                    file_hash,
                    status,
                    parser,
                    parser_version,
                    error_message,
                    metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (doc_id) DO UPDATE
                SET file_name = EXCLUDED.file_name,
                    file_hash = EXCLUDED.file_hash,
                    status = EXCLUDED.status,
                    parser = EXCLUDED.parser,
                    parser_version = EXCLUDED.parser_version,
                    error_message = EXCLUDED.error_message,
                    metadata_json = EXCLUDED.metadata_json,
                    updated_at = now(),
                    deleted_at = NULL
                """,
                (
                    doc_id,
                    kb_id,
                    file_name,
                    file_hash,
                    status,
                    parser,
                    parser_version,
                    error_message,
                    _json(metadata or {}),
                ),
            )

    def create_ingest_job(self, *, kb_id: str, doc_id: str, worker_id: str | None = None) -> str:
        job_id = str(uuid.uuid4())
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO ingest_jobs(job_id, doc_id, kb_id, status, worker_id)
                    VALUES (%s, %s, %s, 'queued', %s)
                    """,
                    (job_id, doc_id, kb_id, worker_id),
                )
        except Exception as exc:
            if _constraint_name(exc) == "idx_ingest_jobs_one_active_doc":
                raise ActiveIngestJobExists(f"active ingest job already exists for doc_id={doc_id}") from exc
            raise
        return job_id

    def get_active_ingest_job(self, doc_id: str) -> IngestJobState | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, doc_id, kb_id, status, worker_id, retry_count, error_message
                FROM ingest_jobs
                WHERE doc_id = %s
                  AND status IN ('queued', 'running', 'uploading', 'parsing', 'embedding')
                ORDER BY created_at
                LIMIT 1
                """,
                (doc_id,),
            ).fetchone()
        return _job_state(row) if row else None

    def update_ingest_job(
        self,
        job_id: str,
        *,
        status: str,
        retry_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET status = %s,
                    retry_count = COALESCE(%s, retry_count),
                    error_message = %s,
                    updated_at = now(),
                    finished_at = CASE
                        WHEN %s IN ('completed', 'failed', 'cancelled') THEN now()
                        ELSE finished_at
                    END
                WHERE job_id = %s
                """,
                (status, retry_count, error_message, status, job_id),
            )

    def fail_ingest_job(self, job_id: str, *, error_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET status = 'failed',
                    retry_count = retry_count + 1,
                    error_message = %s,
                    updated_at = now(),
                    finished_at = now()
                WHERE job_id = %s
                """,
                (error_message, job_id),
            )

    def write_index_snapshot(
        self,
        *,
        kb_id: str,
        index_version: str,
        raw_documents: list[RawDocument],
        chunks: list[DocumentChunk],
        embedding_model: str,
        chunker_version: str,
        parser_version: str,
        milvus_collection: str,
        milvus_dense_field: str = MILVUS_VECTOR_FIELD,
        milvus_sparse_field: str | None = None,
    ) -> None:
        document_records = _document_records(kb_id, raw_documents, chunks, parser_version)
        doc_id_by_source_path = {record["source_path"]: record["doc_id"] for record in document_records}
        chunk_rows = _chunk_rows(kb_id, index_version, chunks, doc_id_by_source_path)

        with self._connect() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO knowledge_bases(kb_id, name, status)
                    VALUES (%s, %s, 'active')
                    ON CONFLICT (kb_id) DO UPDATE
                    SET updated_at = now(), deleted_at = NULL
                    """,
                    (kb_id, kb_id),
                )
                conn.execute(
                    """
                    INSERT INTO index_versions(
                        kb_id,
                        index_version,
                        embedding_model,
                        chunker_version,
                        parser_version,
                        milvus_collection,
                        milvus_dense_field,
                        milvus_sparse_field,
                        status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'building')
                    ON CONFLICT (kb_id, index_version) DO UPDATE
                    SET embedding_model = EXCLUDED.embedding_model,
                        chunker_version = EXCLUDED.chunker_version,
                        parser_version = EXCLUDED.parser_version,
                        milvus_collection = EXCLUDED.milvus_collection,
                        milvus_dense_field = EXCLUDED.milvus_dense_field,
                        milvus_sparse_field = EXCLUDED.milvus_sparse_field,
                        status = 'building',
                        activated_at = NULL
                    """,
                    (
                        kb_id,
                        index_version,
                        embedding_model,
                        chunker_version,
                        parser_version,
                        milvus_collection,
                        milvus_dense_field,
                        milvus_sparse_field,
                    ),
                )
                for record in document_records:
                    conn.execute(
                        """
                        INSERT INTO documents(
                            doc_id,
                            kb_id,
                            file_name,
                            file_hash,
                            status,
                            parser,
                            parser_version,
                            error_message,
                            metadata_json
                        )
                        VALUES (%s, %s, %s, %s, 'indexed', %s, %s, NULL, %s::jsonb)
                        ON CONFLICT (doc_id) DO UPDATE
                        SET file_name = EXCLUDED.file_name,
                            file_hash = EXCLUDED.file_hash,
                            status = 'indexed',
                            parser = EXCLUDED.parser,
                            parser_version = EXCLUDED.parser_version,
                            error_message = NULL,
                            metadata_json = EXCLUDED.metadata_json,
                            updated_at = now(),
                            deleted_at = NULL
                        """,
                        (
                            record["doc_id"],
                            kb_id,
                            record["file_name"],
                            record["file_hash"],
                            record["parser"],
                            parser_version,
                            _json(record["metadata"]),
                        ),
                    )
                conn.execute(
                    "DELETE FROM chunks WHERE kb_id = %s AND index_version = %s",
                    (kb_id, index_version),
                )
                for row in chunk_rows:
                    conn.execute(
                        """
                        INSERT INTO chunks(
                            chunk_id,
                            doc_id,
                            kb_id,
                            chunk_index,
                            text,
                            page_no,
                            token_count,
                            chunk_hash,
                            metadata_json,
                            index_version
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (kb_id, index_version, chunk_id) DO UPDATE
                        SET doc_id = EXCLUDED.doc_id,
                            chunk_index = EXCLUDED.chunk_index,
                            text = EXCLUDED.text,
                            page_no = EXCLUDED.page_no,
                            token_count = EXCLUDED.token_count,
                            chunk_hash = EXCLUDED.chunk_hash,
                            metadata_json = EXCLUDED.metadata_json,
                            deleted_at = NULL
                        """,
                        (
                            row["chunk_id"],
                            row["doc_id"],
                            kb_id,
                            row["chunk_index"],
                            row["text"],
                            row["page_no"],
                            row["token_count"],
                            row["chunk_hash"],
                            _json(row["metadata"]),
                            index_version,
                        ),
                    )
                conn.execute(
                    """
                    UPDATE index_versions
                    SET status = 'superseded'
                    WHERE kb_id = %s
                      AND index_version <> %s
                      AND status = 'active'
                    """,
                    (kb_id, index_version),
                )
                conn.execute(
                    """
                    UPDATE index_versions
                    SET status = 'active',
                        activated_at = now()
                    WHERE kb_id = %s AND index_version = %s
                    """,
                    (kb_id, index_version),
                )

    def load_active_chunks(self, kb_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.text, c.metadata_json, c.index_version
                FROM chunks c
                JOIN index_versions iv
                  ON iv.kb_id = c.kb_id
                 AND iv.index_version = c.index_version
                 AND iv.status = 'active'
                WHERE c.kb_id = %s
                  AND c.deleted_at IS NULL
                ORDER BY c.doc_id, c.chunk_index, c.chunk_id
                """,
                (kb_id,),
            ).fetchall()
        chunks: list[dict[str, Any]] = []
        for row in rows:
            metadata = _metadata_value(row[2])
            metadata["chunk_id"] = row[0]
            metadata["kb_id"] = kb_id
            metadata["index_version"] = row[3]
            chunks.append({"chunk_id": row[0], "text": row[1], "metadata": metadata})
        return chunks

    def active_index_version(self, kb_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT index_version
                FROM index_versions
                WHERE kb_id = %s AND status = 'active'
                """,
                (kb_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def validate_consistency(
        self,
        *,
        kb_id: str,
        milvus_vector_count: int | None = None,
    ) -> IndexConsistency:
        with self._connect() as conn:
            version_row = conn.execute(
                """
                SELECT index_version
                FROM index_versions
                WHERE kb_id = %s AND status = 'active'
                """,
                (kb_id,),
            ).fetchone()
            index_version = str(version_row[0]) if version_row else None
            chunk_count = 0
            if index_version:
                chunk_count = int(
                    conn.execute(
                        """
                        SELECT count(*)
                        FROM chunks
                        WHERE kb_id = %s
                          AND index_version = %s
                          AND deleted_at IS NULL
                        """,
                        (kb_id, index_version),
                    ).fetchone()[0]
                )
            status_rows = conn.execute(
                """
                SELECT status, count(*)
                FROM documents
                WHERE kb_id = %s AND deleted_at IS NULL
                GROUP BY status
                """,
                (kb_id,),
            ).fetchall()
        return IndexConsistency(
            kb_id=kb_id,
            index_version=index_version,
            postgres_chunk_count=chunk_count,
            milvus_vector_count=milvus_vector_count,
            document_statuses={str(status): int(count) for status, count in status_rows},
        )

    def try_advisory_lock(self, lock_key: int) -> bool:
        conn = self._connect()
        try:
            acquired = bool(conn.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0])
        except Exception:
            conn.close()
            raise
        if not acquired:
            conn.close()
            return False
        with self._advisory_lock_connections_lock:
            self._advisory_lock_connections[lock_key] = conn
        return True

    def release_advisory_lock(self, lock_key: int) -> None:
        with self._advisory_lock_connections_lock:
            conn = self._advisory_lock_connections.pop(lock_key, None)
        if conn is None:
            return
        try:
            conn.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
        finally:
            conn.close()

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise MetadataStoreUnavailable("psycopg is required when POSTGRES_DSN is configured.") from exc
        return psycopg.connect(self.dsn)


def _document_records(
    kb_id: str,
    raw_documents: list[RawDocument],
    chunks: list[DocumentChunk],
    parser_version: str,
) -> list[dict[str, Any]]:
    by_source_path: dict[str, dict[str, Any]] = {}
    for document in raw_documents:
        source_path = document.source_path
        record = by_source_path.get(source_path)
        if record is None:
            file_hash = _file_hash(Path(source_path), document.text)
            record = {
                "doc_id": _doc_id(kb_id, file_hash),
                "source_path": source_path,
                "file_name": document.source_name,
                "file_hash": file_hash,
                "parser": document.parser,
                "metadata": {
                    "source_path": source_path,
                    "file_type": document.file_type,
                    "scenario": document.scenario,
                    "parser_version": parser_version,
                },
            }
            by_source_path[source_path] = record
    for chunk in chunks:
        source_path = str(chunk.metadata.get("source_path", ""))
        if source_path and source_path not in by_source_path:
            text = chunk.text
            file_hash = _file_hash(Path(source_path), text)
            by_source_path[source_path] = {
                "doc_id": _doc_id(kb_id, file_hash),
                "source_path": source_path,
                "file_name": str(chunk.metadata.get("source_name") or Path(source_path).name),
                "file_hash": file_hash,
                "parser": str(chunk.metadata.get("parser") or "unknown"),
                "metadata": {
                    "source_path": source_path,
                    "file_type": chunk.metadata.get("file_type"),
                    "scenario": chunk.metadata.get("scenario"),
                    "parser_version": parser_version,
                },
            }
    return list(by_source_path.values())


def _chunk_rows(
    kb_id: str,
    index_version: str,
    chunks: list[DocumentChunk],
    doc_id_by_source_path: dict[str, str],
) -> list[dict[str, Any]]:
    positions: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        source_path = str(metadata.get("source_path", ""))
        doc_id = doc_id_by_source_path.get(source_path)
        if doc_id is None:
            file_hash = _file_hash(Path(source_path), chunk.text)
            doc_id = _doc_id(kb_id, file_hash)
        chunk_index = positions.get(doc_id, 0)
        positions[doc_id] = chunk_index + 1
        metadata["kb_id"] = kb_id
        metadata["index_version"] = index_version
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": chunk.text,
                "page_no": _int_or_none(metadata.get("page")),
                "token_count": len(chunk.text.split()),
                "chunk_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                "metadata": metadata,
            }
        )
    return rows


def _file_hash(path: Path, fallback_text: str) -> str:
    digest = hashlib.sha256()
    try:
        if path.exists() and path.is_file():
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest()
    except OSError:
        pass
    digest.update(fallback_text.encode("utf-8"))
    return digest.hexdigest()


def _doc_id(kb_id: str, file_hash: str) -> str:
    return hashlib.sha1(f"{kb_id}:{file_hash}".encode("utf-8")).hexdigest()[:24]


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _metadata_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _constraint_name(exc: Exception) -> str | None:
    diag = getattr(exc, "diag", None)
    value = getattr(diag, "constraint_name", None)
    return str(value) if value is not None else None


def _job_state(row: Any) -> IngestJobState:
    return IngestJobState(
        job_id=str(row[0]),
        doc_id=str(row[1]),
        kb_id=str(row[2]),
        status=str(row[3]),
        worker_id=str(row[4]) if row[4] is not None else None,
        retry_count=int(row[5]),
        error_message=str(row[6]) if row[6] is not None else None,
    )
