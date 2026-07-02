from __future__ import annotations

from app.documents import DocumentChunk
from app.index import HybridIndex
from app.kb_metadata import MIGRATIONS_DIR, ActiveChunkSnapshot
from app.providers import HashEmbeddings


class FakeMetadataStore:
    backend = "postgresql"

    def __init__(self, chunks, index_version: str = "postgres-active-v1"):
        self._chunks = chunks
        self._index_version = index_version

    def load_active_snapshot(self, kb_id: str):
        assert kb_id == "kb-test"
        return ActiveChunkSnapshot(kb_id=kb_id, index_version=self._index_version, chunks=self._chunks)


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


def test_missing_rag_index_directory_does_not_block_build_or_query(tmp_path, fake_milvus_client):
    legacy_dir = tmp_path / ".rag_index"
    assert not legacy_dir.exists()
    index = HybridIndex(
        legacy_dir,
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        kb_id="kb-test",
    )

    index.build(_chunks())
    sources, debug = index.search(
        "traceId",
        scenario=None,
        dense_top_k=2,
        bm25_top_k=2,
        final_top_k=1,
        rrf_k=60,
    )

    assert sources
    assert debug["retrieval_backend"] == "milvus_hybrid"
    assert not legacy_dir.exists()


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
                "index_version": "postgres-active-v1",
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
    assert loaded.index_revision == "postgres-active-v1"
    assert loaded.chunks[0]["text"] == "postgres is the metadata source"
    assert loaded.ready


def test_hybrid_index_load_uses_postgres_active_index_version_not_content_hash(tmp_path, fake_milvus_client):
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
            "text": "same milvus row but edited postgres metadata",
            "metadata": {
                "chunk_id": "chunk-a",
                "source_path": "policy.md",
                "source_name": "policy.md",
                "file_type": "md",
                "scenario": "ops",
                "kb_id": "kb-test",
                "index_version": first.index_revision,
                "postgres_note": "does not change active index_version",
            },
        }
    ]
    loaded = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        metadata_store=FakeMetadataStore(postgres_chunks, index_version=first.index_revision),
        kb_id="kb-test",
    )

    assert loaded.load() is True
    assert loaded.index_revision == first.index_revision


def test_hybrid_index_load_rejects_dense_only_milvus_collection(tmp_path, fake_milvus_client):
    fake_milvus_client.create_collection(collection_name="rag_chunks", dimension=32)
    loaded = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        metadata_store=FakeMetadataStore(
            [
                {
                    "chunk_id": "chunk-a",
                    "text": "postgres metadata",
                    "metadata": {"chunk_id": "chunk-a", "kb_id": "kb-test", "index_version": "postgres-active-v1"},
                }
            ]
        ),
        kb_id="kb-test",
    )

    assert loaded.load() is False
    assert not loaded.ready


def test_hybrid_index_does_not_hydrate_missing_postgres_chunk_from_milvus_entity(tmp_path, fake_milvus_client):
    index = HybridIndex(
        tmp_path / "index",
        HashEmbeddings(dimensions=32),
        milvus_client=fake_milvus_client,
        metadata_store=FakeMetadataStore([], index_version=""),
        kb_id="kb-test",
    )
    index.build(_chunks())
    postgres_only_chunk = {
        "chunk_id": "postgres-only",
        "text": "postgres only metadata",
        "metadata": {"chunk_id": "postgres-only", "kb_id": "kb-test", "index_version": index.index_revision},
    }
    index.chunks = [postgres_only_chunk]
    index.chunk_by_id = {"postgres-only": postgres_only_chunk}

    sources, debug = index.search(
        "traceId",
        scenario=None,
        dense_top_k=2,
        bm25_top_k=2,
        final_top_k=2,
        rrf_k=60,
    )

    assert sources == []
    assert debug["fused_hits"] == []


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
