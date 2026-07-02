CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_bases (
    kb_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    parser TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    error_message TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_kb_file_hash
    ON documents(kb_id, file_hash)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_documents_kb_status
    ON documents(kb_id, status);

CREATE TABLE IF NOT EXISTS index_versions (
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
    index_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    milvus_collection TEXT NOT NULL,
    milvus_dense_field TEXT NOT NULL,
    milvus_sparse_field TEXT,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at TIMESTAMPTZ,
    PRIMARY KEY (kb_id, index_version)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_index_versions_one_active
    ON index_versions(kb_id)
    WHERE status = 'active';

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT NOT NULL,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_no INTEGER,
    token_count INTEGER,
    chunk_hash TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    index_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (kb_id, index_version, chunk_id),
    FOREIGN KEY (kb_id, index_version) REFERENCES index_versions(kb_id, index_version)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc
    ON chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_kb_index_version
    ON chunks(kb_id, index_version)
    WHERE deleted_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doc_version_position
    ON chunks(doc_id, index_version, chunk_index)
    WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
    status TEXT NOT NULL,
    worker_id TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_jobs_one_active_doc
    ON ingest_jobs(doc_id)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS idx_ingest_jobs_kb_status
    ON ingest_jobs(kb_id, status);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    log_id BIGSERIAL PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
    index_version TEXT,
    query_hash TEXT NOT NULL,
    bm25_backend TEXT NOT NULL,
    dense_top_k INTEGER NOT NULL,
    bm25_top_k INTEGER NOT NULL,
    final_top_k INTEGER NOT NULL,
    result_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
