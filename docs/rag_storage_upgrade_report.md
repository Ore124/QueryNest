# RAG Storage Architecture Upgrade Report

## Stage 0 Audit

Current ingest flow:

```mermaid
flowchart LR
    Upload["Document upload or path"] --> Parse["DocumentParserRouter"]
    Parse --> Chunk["split_documents"]
    Chunk --> Embed["embedding provider"]
    Embed --> Milvus["Milvus dense collection"]
    Chunk --> LocalChunks[".rag_index/chunks.jsonl"]
    Chunk --> LocalBM25[".rag_index/bm25.pkl"]
```

Current query flow:

```mermaid
flowchart LR
    Query["query"] --> Dense["Milvus dense search"]
    Query --> BM25["local BM25 from .rag_index/bm25.pkl"]
    Dense --> Hydrate["hydrate text/metadata from chunks.jsonl"]
    BM25 --> Fusion["RRF fusion"]
    Hydrate --> Fusion
    Fusion --> Rerank["optional reranker"]
    Rerank --> Answer["answer with sources"]
```

Important code paths:

- `backend/app/main.py`: API entry points for `/api/ingest/path`, `/api/ingest/files`, `/api/chat`.
- `backend/app/documents.py`: document discovery, parsing output normalization, chunk creation.
- `backend/app/index.py`: Milvus dense collection build/search, local BM25 build/search, RRF, rerank handoff, `.rag_index` persistence.
- `backend/app/cache.py`: Redis query/embedding cache.
- `backend/app/history.py`: Redis chat history or SQLite fallback.
- `backend/app/graph.py`: query orchestration and answer generation.

`.rag_index` was a fact source in these paths:

- Startup: `HybridIndex.load()` required both `.rag_index/bm25.pkl` and `.rag_index/chunks.jsonl`.
- Dense search hydration: Milvus returned only `chunk_id`; text and metadata came from `chunk_by_id`, loaded from `chunks.jsonl`.
- BM25 search: corpus came from `bm25.pkl` plus positional alignment with `chunks.jsonl`.
- Cache revision: retrieval cache key used the local chunk/text/metadata hash.

Multi-process consistency risks found:

- `hybrid_index`, `history_store`, `embedding_warmed`, and `embedding_warmup_lock` are process-local globals in `main.py`.
- Ingest rebuild drops and recreates the Milvus collection without a distributed lock.
- `.rag_index` rebuild deletes and rewrites local files; another process can read missing or partial artifacts.
- `bm25.pkl` and `chunks.jsonl` are not atomically swapped.
- Duplicate ingest requests can rebuild the same KB concurrently.
- Query can observe Milvus rebuilt but local chunk/BM25 files not yet rewritten.

## Implemented Changes

Stage 1 and the safe subset of Stage 2 were implemented:

- Added PostgreSQL migration SQL under `backend/app/migrations/001_kb_metadata.sql`.
- Added `PostgresMetadataStore` in `backend/app/kb_metadata.py`.
- Added default-off settings:
  - `POSTGRES_DSN`
  - `KB_ID`
  - `PARSER_VERSION`
  - `CHUNKER_VERSION`
  - `RETRIEVAL_BM25_BACKEND`
- Added PostgreSQL service to `compose.yaml`.
- Added `psycopg[binary]` dependency and packaged SQL migrations.
- `HybridIndex.load()` now prefers active chunk metadata from PostgreSQL when `POSTGRES_DSN` is configured and falls back to `.rag_index`.
- `HybridIndex.build()` writes a `manifest.json` next to `.rag_index` artifacts.
- Milvus dense rows now include dynamic scalar metadata: `chunk_id`, `doc_id`, `kb_id`, `index_version`, `text`, `page_no`, `chunk_index`.
- Local BM25 remains the default backend for rollback.

## PostgreSQL Schema

Tables added:

- `knowledge_bases`: `kb_id`, `name`, `status`, `metadata_json`, timestamps, `deleted_at`.
- `documents`: `doc_id`, `kb_id`, `file_name`, `file_hash`, `status`, `parser`, `parser_version`, `error_message`, metadata and timestamps.
- `chunks`: `chunk_id`, `doc_id`, `kb_id`, `chunk_index`, `text`, `page_no`, `token_count`, `chunk_hash`, `metadata_json`, `index_version`, timestamps.
- `index_versions`: `kb_id`, `index_version`, `embedding_model`, `chunker_version`, `parser_version`, Milvus collection/field names, `status`, timestamps.
- `ingest_jobs`: `job_id`, `doc_id`, `kb_id`, `status`, `worker_id`, `retry_count`, `error_message`, timestamps.
- `retrieval_logs`: optional query log table with query hash and returned chunk ids.

Important constraints:

- One active `index_versions` row per `kb_id`.
- One active ingest job per `doc_id` where status is `queued` or `running`.
- `chunks` primary key is `(kb_id, index_version, chunk_id)` for version isolation.

## Milvus Schema Status

Current implementation still creates the dense vector collection through `MilvusClient.create_collection(...)` with:

- Primary field: `chunk_id`
- Dense vector field: `embedding`
- Dynamic fields enabled for scalar metadata.

The collection is not yet rebuilt with a first-class BM25 schema. Milvus BM25 full-text requires a `VARCHAR` text field with analyzer enabled, a `SPARSE_FLOAT_VECTOR` field, and a BM25 function/index. Milvus hybrid search then combines dense and sparse requests with RRF or weighted rankers.

Project versions observed:

- Docker service: `milvusdb/milvus:v2.6.3`
- Declared Python client: `pymilvus>=2.5.0,<3.0.0`
- Current local Python environment: `pymilvus` is not installed, so runtime capability could not be verified locally.

## Local Startup

Start storage services:

```powershell
docker compose up -d postgres milvus redis
```

Install/update backend dependencies:

```powershell
cd backend
..\.venv\Scripts\python -m pip install -e .
```

Enable PostgreSQL metadata:

```env
POSTGRES_DSN=postgresql://rag:rag_dev_password@127.0.0.1:5432/rag
KB_ID=default
RETRIEVAL_BM25_BACKEND=local
```

Migrations run at backend startup when `POSTGRES_DSN` is configured. They can also be run explicitly:

```powershell
cd backend
$env:POSTGRES_DSN="postgresql://rag:rag_dev_password@127.0.0.1:5432/rag"
..\.venv\Scripts\python scripts\run_postgres_migrations.py
```

The SQL is idempotent and tracked by `schema_migrations`.

## Verification

Targeted tests:

```powershell
cd backend
New-Item -ItemType Directory -Force .pytest-tmp | Out-Null
$env:TMP=(Resolve-Path .pytest-tmp)
$env:TEMP=$env:TMP
..\.venv\Scripts\python -m pytest tests/test_storage_metadata.py tests/test_ingestion.py tests/test_cache.py tests/test_api.py
```

Result: 15 passed, 1 warning.

## Rollback

Set:

```env
POSTGRES_DSN=
RETRIEVAL_BM25_BACKEND=local
```

With `POSTGRES_DSN` empty, the backend uses the previous `.rag_index/chunks.jsonl` and `.rag_index/bm25.pkl` loading path. Redis and SQLite behavior is unchanged.

## Remaining Work

- Add real PostgreSQL integration tests against the compose service.
- Move ingest job creation into the request path for per-document concurrency protection.
- Add a distributed lock around whole-KB rebuilds.
- Replace delete/recreate Milvus collection with index-version-filtered inserts before enabling no-downtime rebuilds.
- Implement the Milvus BM25 collection schema only after verifying installed `pymilvus` and running Milvus support the required `FunctionType.BM25`, sparse field, and `hybrid_search` APIs.
- Add production query filters for `kb_id` and active `index_version` once the Milvus collection no longer drops old vectors on rebuild.
