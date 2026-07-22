CREATE TABLE IF NOT EXISTS session_summaries (
    session_id UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    version BIGINT NOT NULL CHECK (version > 0),
    content TEXT NOT NULL CHECK (length(content) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, version)
);

CREATE TABLE IF NOT EXISTS session_working_memory_facts (
    fact_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL CHECK (
        memory_type IN ('task', 'constraint', 'entity', 'open_question')
    ),
    fact_key TEXT NOT NULL CHECK (length(fact_key) > 0),
    fact_value TEXT NOT NULL CHECK (length(fact_value) > 0),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    sort_order INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_session_working_memory_one_active_key
    ON session_working_memory_facts(session_id, memory_type, fact_key)
    WHERE is_active;

CREATE INDEX IF NOT EXISTS idx_session_working_memory_active_sort
    ON session_working_memory_facts(session_id, sort_order, fact_id)
    WHERE is_active;
