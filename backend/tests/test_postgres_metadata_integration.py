from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import pytest

from app.documents import DocumentChunk, RawDocument
from app.index import HybridIndex
from app.kb_metadata import ActiveIngestJobExists, PostgresMetadataStore
from app.providers import HashEmbeddings


pytestmark = pytest.mark.integration


def _base_dsn() -> str:
    return (
        os.environ.get("POSTGRES_TEST_DSN")
        or os.environ.get("POSTGRES_DSN")
        or "postgresql://rag:rag_dev_password@127.0.0.1:5432/rag"
    )


def _schema_dsn(base_dsn: str, schema: str) -> str:
    parsed = urlparse(base_dsn)
    if parsed.scheme.startswith("postgres"):
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        params["options"] = f"-c search_path={schema}"
        params.setdefault("connect_timeout", "3")
        return urlunparse(parsed._replace(query=urlencode(params, quote_via=quote)))
    return f"{base_dsn} connect_timeout=3 options='-c search_path={schema}'"


@pytest.fixture
def psycopg_module():
    psycopg = pytest.importorskip("psycopg")
    return psycopg


@pytest.fixture
def postgres_schema(psycopg_module):
    psycopg = psycopg_module
    base_dsn = _base_dsn()
    schema = f"test_rag_{uuid.uuid4().hex}"
    try:
        with psycopg.connect(base_dsn, autocommit=True, connect_timeout=3) as conn:
            conn.execute(f'CREATE SCHEMA "{schema}"')
    except Exception as exc:
        pytest.skip(f"PostgreSQL integration database is not available: {exc}")
    try:
        yield _schema_dsn(base_dsn, schema)
    finally:
        with psycopg.connect(base_dsn, autocommit=True, connect_timeout=3) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.fixture
def pg_store(postgres_schema: str) -> PostgresMetadataStore:
    store = PostgresMetadataStore(postgres_schema)
    store.run_migrations()
    return store


def _raw_document(path: Path, *, text: str = "approval workflow source") -> RawDocument:
    path.write_text(text, encoding="utf-8")
    return RawDocument(
        text=text,
        source_path=str(path),
        source_name=path.name,
        file_type="md",
        scenario="ops",
        parser="markdown",
    )


def _chunks(path: Path, *, text_prefix: str = "postgres metadata") -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk-a",
            text=f"{text_prefix} alpha",
            metadata={
                "chunk_id": "chunk-a",
                "source_path": str(path),
                "source_name": path.name,
                "file_type": "md",
                "scenario": "ops",
                "page": 1,
                "chunk_index": 0,
                "parser": "markdown",
            },
        ),
        DocumentChunk(
            chunk_id="chunk-b",
            text=f"{text_prefix} beta",
            metadata={
                "chunk_id": "chunk-b",
                "source_path": str(path),
                "source_name": path.name,
                "file_type": "md",
                "scenario": "ops",
                "page": 2,
                "chunk_index": 1,
                "parser": "markdown",
            },
        ),
    ]


def _write_snapshot(store: PostgresMetadataStore, tmp_path: Path, *, kb_id: str, index_version: str, text_prefix: str = "postgres metadata"):
    doc_path = tmp_path / f"{kb_id}.md"
    raw_documents = [_raw_document(doc_path, text="stable source document")]
    chunks = _chunks(doc_path, text_prefix=text_prefix)
    store.write_index_snapshot(
        kb_id=kb_id,
        index_version=index_version,
        raw_documents=raw_documents,
        chunks=chunks,
        embedding_model="embedding-3",
        chunker_version="chunker-v1",
        parser_version="parser-v1",
        milvus_collection="rag_chunks_test",
    )
    return raw_documents, chunks


def test_postgres_migrations_are_repeatable(postgres_schema: str, psycopg_module):
    store = PostgresMetadataStore(postgres_schema)
    store.run_migrations()
    store.run_migrations()

    with psycopg_module.connect(postgres_schema) as conn:
        versions = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                """
            ).fetchall()
        }

    assert versions == [
        ("001_kb_metadata",),
        ("002_ingest_job_active_statuses",),
        ("003_retrieval_logs_backend_column",),
    ]
    assert {"documents", "chunks", "index_versions", "ingest_jobs"}.issubset(tables)


def test_postgres_metadata_write_read_and_update(pg_store: PostgresMetadataStore, postgres_schema: str, psycopg_module, tmp_path):
    kb_id = "kb-write-read"
    _write_snapshot(pg_store, tmp_path, kb_id=kb_id, index_version="v1", text_prefix="first")

    loaded = pg_store.load_active_chunks(kb_id)
    assert [chunk["chunk_id"] for chunk in loaded] == ["chunk-a", "chunk-b"]
    assert loaded[0]["text"] == "first alpha"
    assert pg_store.active_index_version(kb_id) == "v1"

    _write_snapshot(pg_store, tmp_path, kb_id=kb_id, index_version="v2", text_prefix="second")
    loaded = pg_store.load_active_chunks(kb_id)
    assert pg_store.active_index_version(kb_id) == "v2"
    assert loaded[0]["text"] == "second alpha"

    with psycopg_module.connect(postgres_schema) as conn:
        doc_id = conn.execute("SELECT doc_id FROM documents WHERE kb_id = %s", (kb_id,)).fetchone()[0]
        statuses = dict(conn.execute("SELECT index_version, status FROM index_versions WHERE kb_id = %s", (kb_id,)).fetchall())
        chunk_count = conn.execute(
            "SELECT count(*) FROM chunks WHERE kb_id = %s AND index_version = %s",
            (kb_id, "v2"),
        ).fetchone()[0]

    assert statuses == {"v1": "superseded", "v2": "active"}
    assert chunk_count == 2

    job_id = pg_store.create_ingest_job(kb_id=kb_id, doc_id=doc_id, worker_id="worker-a")
    pg_store.update_ingest_job(job_id, status="running", retry_count=1)
    pg_store.update_ingest_job(job_id, status="completed")

    with psycopg_module.connect(postgres_schema) as conn:
        row = conn.execute(
            "SELECT status, retry_count, finished_at IS NOT NULL FROM ingest_jobs WHERE job_id = %s",
            (job_id,),
        ).fetchone()

    assert row == ("completed", 1, True)


def test_active_index_version_constraint(pg_store: PostgresMetadataStore, postgres_schema: str, psycopg_module, tmp_path):
    kb_id = "kb-active-version"
    _write_snapshot(pg_store, tmp_path, kb_id=kb_id, index_version="v1")

    with psycopg_module.connect(postgres_schema) as conn:
        with pytest.raises(psycopg_module.errors.UniqueViolation):
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
                    status
                )
                VALUES (%s, 'manual-active', 'embedding-3', 'chunker-v1', 'parser-v1', 'rag_chunks_test', 'embedding', 'active')
                """,
                (kb_id,),
            )


def test_doc_id_allows_only_one_active_ingest_job(pg_store: PostgresMetadataStore, postgres_schema: str, psycopg_module, tmp_path):
    kb_id = "kb-ingest-job"
    _write_snapshot(pg_store, tmp_path, kb_id=kb_id, index_version="v1")

    with psycopg_module.connect(postgres_schema) as conn:
        doc_id = conn.execute("SELECT doc_id FROM documents WHERE kb_id = %s", (kb_id,)).fetchone()[0]

    first_job_id = pg_store.create_ingest_job(kb_id=kb_id, doc_id=doc_id, worker_id="worker-a")
    with pytest.raises(ActiveIngestJobExists):
        pg_store.create_ingest_job(kb_id=kb_id, doc_id=doc_id, worker_id="worker-b")

    pg_store.update_ingest_job(first_job_id, status="completed")
    second_job_id = pg_store.create_ingest_job(kb_id=kb_id, doc_id=doc_id, worker_id="worker-b")
    assert second_job_id != first_job_id


def test_hybrid_index_load_uses_postgres_and_does_not_fallback_to_local_artifacts(
    pg_store: PostgresMetadataStore,
    tmp_path,
    fake_milvus_client,
):
    kb_id = "kb-hybrid-load"
    local_index = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        kb_id=kb_id,
    )
    local_path = tmp_path / "local.md"
    local_chunks = _chunks(local_path, text_prefix="local artifact")
    local_index.build(local_chunks)

    raw_documents = [_raw_document(local_path, text="postgres source document")]
    postgres_chunks = _chunks(local_path, text_prefix="postgres source")
    pg_store.write_index_snapshot(
        kb_id=kb_id,
        index_version=local_index.index_revision,
        raw_documents=raw_documents,
        chunks=postgres_chunks,
        embedding_model="embedding-3",
        chunker_version="chunker-v1",
        parser_version="parser-v1",
        milvus_collection=local_index.milvus_collection_name,
    )

    postgres_loaded = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        metadata_store=pg_store,
        kb_id=kb_id,
    )
    assert postgres_loaded.load() is True
    assert postgres_loaded.origin == "postgresql_milvus"
    assert postgres_loaded.chunks[0]["text"] == "postgres source alpha"

    fallback_loaded = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        kb_id=kb_id,
    )
    with pytest.raises(RuntimeError, match="PostgreSQL metadata store is required"):
        fallback_loaded.load()
    assert fallback_loaded.origin == "not_loaded"
    assert fallback_loaded.chunks == []
