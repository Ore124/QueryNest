# RAG Storage Architecture Upgrade Report

## Current Architecture

Ingest flow:

```mermaid
flowchart LR
    Upload["Document upload or path"] --> Parse["DocumentParserRouter"]
    Parse --> Chunk["split_documents"]
    Chunk --> Postgres["PostgreSQL documents/chunks/index_versions"]
    Chunk --> Embed["embedding provider"]
    Embed --> Milvus["Milvus dense + BM25 sparse collection"]
```

Query flow:

```mermaid
flowchart LR
    Query["query"] --> Dense["Milvus dense search"]
    Query --> BM25["Milvus BM25 sparse search"]
    Dense --> Hybrid["Milvus hybrid_search + RRF"]
    BM25 --> Hybrid
    Postgres["PostgreSQL active chunks"] --> Hydrate["chunk text and metadata"]
    Hybrid --> Hydrate
    Hydrate --> Rerank["optional reranker"]
    Rerank --> Answer["answer with sources"]
```

## Implemented Changes

- PostgreSQL is now the chunk metadata fact source through `PostgresMetadataStore`.
- `HybridIndex.load()` requires PostgreSQL metadata and an existing Milvus collection; it no longer reads local chunk artifacts.
- `HybridIndex.build()` creates a Milvus hybrid collection with dense vector, analyzer-enabled text, BM25 sparse vector, and scalar metadata.
- Query always uses Milvus dense search, Milvus BM25 sparse search, and Milvus `hybrid_search` with RRF.
- `.rag_index/chunks.jsonl`, `.rag_index/bm25.pkl`, `rank-bm25`, and the local BM25 fallback path were removed from runtime.
- `RETRIEVAL_BM25_BACKEND` was removed. The only supported retrieval backend is Milvus hybrid.
- Whole-KB rebuild remains guarded by Redis lock when Redis is enabled, otherwise PostgreSQL advisory lock.
- Ingest requests are guarded by PostgreSQL `ingest_jobs` when `POSTGRES_DSN` is configured.

## PostgreSQL Schema

Tables:

- `knowledge_bases`: `kb_id`, `name`, `status`, `metadata_json`, timestamps, `deleted_at`.
- `documents`: `doc_id`, `kb_id`, `file_name`, `file_hash`, `status`, `parser`, `parser_version`, `error_message`, metadata and timestamps.
- `chunks`: `chunk_id`, `doc_id`, `kb_id`, `chunk_index`, `text`, `page_no`, `token_count`, `chunk_hash`, `metadata_json`, `index_version`, timestamps.
- `index_versions`: `kb_id`, `index_version`, `embedding_model`, `chunker_version`, `parser_version`, Milvus collection/field names, `status`, timestamps.
- `ingest_jobs`: `job_id`, `doc_id`, `kb_id`, `status`, `worker_id`, `retry_count`, `error_message`, timestamps.
- `retrieval_logs`: optional query log table with query hash and returned chunk ids.

Important constraints:

- One active `index_versions` row per `kb_id`.
- One active ingest job per `doc_id` where status is `queued`, `running`, `uploading`, `parsing`, or `embedding`.
- `chunks` primary key is `(kb_id, index_version, chunk_id)` for version isolation.

## Milvus Schema

Collection fields:

- `chunk_id`: `VARCHAR`, primary key.
- `kb_id`: `VARCHAR`.
- `doc_id`: `VARCHAR`.
- `index_version`: `VARCHAR`.
- `scenario`: `VARCHAR`.
- `page_no`: `INT64`.
- `chunk_index`: `INT64`.
- `text`: `VARCHAR`, `enable_analyzer=true`.
- `dense`: `FLOAT_VECTOR`.
- `sparse`: `SPARSE_FLOAT_VECTOR`, generated from the BM25 function.

Indexes:

- `dense`: `AUTOINDEX`, `COSINE`.
- `sparse`: `SPARSE_INVERTED_INDEX`, `BM25`.

## Local Startup

```powershell
docker compose up -d postgres milvus redis
cd backend
..\.venv\Scripts\python -m pip install -e .
$env:POSTGRES_DSN="postgresql://rag:rag_dev_password@127.0.0.1:5432/rag"
..\.venv\Scripts\python scripts\run_postgres_migrations.py
```

Default storage settings:

```env
POSTGRES_DSN=postgresql://rag:rag_dev_password@127.0.0.1:5432/rag
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION_NAME=rag_chunks
ARTIFACT_DIR=.rag_artifacts
```

`ARTIFACT_DIR` is not a knowledge-base fact source.

## Verification

Targeted tests:

```powershell
cd backend
New-Item -ItemType Directory -Force .pytest-tmp | Out-Null
$env:TMP=(Resolve-Path .pytest-tmp)
$env:TEMP=$env:TMP
..\.venv\Scripts\python -m pytest tests/test_storage_metadata.py tests/test_cache.py tests/test_ingestion.py tests/test_api.py tests/test_ingest_jobs.py
```

Integration tests:

```powershell
docker compose up -d postgres milvus
cd backend
New-Item -ItemType Directory -Force .pytest-tmp | Out-Null
$env:TMP=(Resolve-Path .pytest-tmp)
$env:TEMP=$env:TMP
..\.venv\Scripts\python -m pytest tests/test_postgres_metadata_integration.py
```

Milvus BM25 capability check:

```powershell
cd backend
..\.venv\Scripts\python scripts\poc_milvus_bm25.py
```

## Rollback

There is no runtime rollback to local BM25. Rollback is a git rollback to a prior commit before this architecture replacement.

## Remaining Risks

- Rebuild still drops and recreates the configured Milvus collection; no-downtime rebuild should move to versioned inserts before production rollout.
- `POSTGRES_DSN` is now required for persisted KB metadata. SQLite remains only session fallback.
- Existing local `.rag_index` artifacts are ignored and can be deleted after confirming no external tooling depends on them.
