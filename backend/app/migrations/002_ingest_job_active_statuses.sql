DROP INDEX IF EXISTS idx_ingest_jobs_one_active_doc;

CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_jobs_one_active_doc
    ON ingest_jobs(doc_id)
    WHERE status IN ('queued', 'running', 'uploading', 'parsing', 'embedding');
