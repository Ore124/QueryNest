DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'retrieval_logs'
          AND column_name = 'bm25_backend'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'retrieval_logs'
          AND column_name = 'retrieval_backend'
    ) THEN
        ALTER TABLE retrieval_logs RENAME COLUMN bm25_backend TO retrieval_backend;
    END IF;
END $$;
