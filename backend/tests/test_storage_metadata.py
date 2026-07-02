from __future__ import annotations

from app.documents import DocumentChunk
from app.index import HybridIndex
from app.kb_metadata import MIGRATIONS_DIR
from app.providers import HashEmbeddings


class FakeMetadataStore:
    backend = "postgresql"

    def __init__(self, chunks):
        self._chunks = chunks

    def load_active_chunks(self, kb_id: str):
        assert kb_id == "kb-test"
        return self._chunks


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk-a",
            text="refund policy requires manager approval",
            metadata={
                "chunk_id": "chunk-a",
                "source_path": "policy.md",
                "source_name": "policy.md",
                "file_type": "md",
                "scenario": "ops",
                "chunk_index": 0,
            },
        ),
        DocumentChunk(
            chunk_id="chunk-b",
            text="incident response checks traceId first",
            metadata={
                "chunk_id": "chunk-b",
                "source_path": "runbook.md",
                "source_name": "runbook.md",
                "file_type": "md",
                "scenario": "ops",
                "chunk_index": 0,
            },
        ),
    ]


def test_metadata_migration_defines_required_tables_and_active_job_guard():
    sql = (MIGRATIONS_DIR / "001_kb_metadata.sql").read_text(encoding="utf-8")

    for table in [
        "knowledge_bases",
        "documents",
        "chunks",
        "index_versions",
        "ingest_jobs",
        "retrieval_logs",
    ]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "idx_ingest_jobs_one_active_doc" in sql
    assert "WHERE status IN ('queued', 'running')" in sql
    assert "idx_index_versions_one_active" in sql
    assert "WHERE status = 'active'" in sql


def test_hybrid_index_writes_index_version_to_milvus_rows_and_manifest(tmp_path, fake_milvus_client):
    index = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        kb_id="kb-test",
    )

    index.build(_chunks())

    rows = fake_milvus_client.collections[index.milvus_collection_name]["rows"]
    assert set(rows) == {"chunk-a", "chunk-b"}
    assert {row["kb_id"] for row in rows.values()} == {"kb-test"}
    assert {row["index_version"] for row in rows.values()} == {index.index_revision}
    assert (tmp_path / "index" / "manifest.json").exists()


def test_hybrid_index_loads_chunk_metadata_from_postgres_before_rag_index(tmp_path, fake_milvus_client):
    first = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        kb_id="kb-test",
    )
    first.build(_chunks())
    postgres_chunks = [
        {
            "chunk_id": "chunk-a",
            "text": "postgres is the metadata source",
            "metadata": {
                "chunk_id": "chunk-a",
                "source_path": "policy.md",
                "source_name": "policy.md",
                "file_type": "md",
                "scenario": "ops",
                "kb_id": "kb-test",
                "index_version": first.index_revision,
            },
        }
    ]
    (tmp_path / "index" / "chunks.jsonl").unlink()

    loaded = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        metadata_store=FakeMetadataStore(postgres_chunks),
        kb_id="kb-test",
    )

    assert loaded.load() is True
    assert loaded.origin == "postgresql"
    assert loaded.chunks[0]["text"] == "postgres is the metadata source"
    assert loaded.ready


def test_local_bm25_backend_still_supports_rollback(tmp_path, fake_milvus_client):
    index = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        bm25_backend="local",
    )
    index.build(_chunks())

    sources, debug = index.search(
        "traceId",
        scenario=None,
        dense_top_k=2,
        bm25_top_k=2,
        final_top_k=2,
        rrf_k=60,
    )

    assert sources
    assert debug["bm25_hits"]
