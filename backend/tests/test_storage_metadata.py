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
    assert "WHERE status IN ('queued', 'running', 'uploading', 'parsing', 'embedding')" in sql
    assert "idx_index_versions_one_active" in sql
    assert "WHERE status = 'active'" in sql


def test_hybrid_index_writes_versioned_rows_to_milvus_without_rag_index_artifacts(tmp_path, fake_milvus_client):
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
    assert {row["text"] for row in rows.values()} == {chunk.text for chunk in _chunks()}
    assert not (tmp_path / "index" / "chunks.jsonl").exists()
    assert not (tmp_path / "index" / "bm25.pkl").exists()


def test_hybrid_index_loads_chunk_metadata_from_postgres(tmp_path, fake_milvus_client):
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
    loaded = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        metadata_store=FakeMetadataStore(postgres_chunks),
        kb_id="kb-test",
    )

    assert loaded.load() is True
    assert loaded.origin == "postgresql_milvus"
    assert loaded.chunks[0]["text"] == "postgres is the metadata source"
    assert loaded.ready


def test_milvus_hybrid_search_returns_dense_and_bm25_rank_details(tmp_path, fake_milvus_client):
    index = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
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
    assert debug["retrieval_backend"] == "milvus_hybrid"
    assert debug["bm25_hits"]
    assert any(source.bm25_rank is not None for source in sources)
