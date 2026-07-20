CREATE TABLE IF NOT EXISTS personal_memories (
    memory_id UUID PRIMARY KEY,
    owner_user_id UUID NOT NULL REFERENCES users(user_id),
    memory_type TEXT NOT NULL CHECK (memory_type IN ('preference', 'profile', 'fact')),
    memory_key TEXT NOT NULL CHECK (length(memory_key) > 0),
    memory_value TEXT NOT NULL CHECK (length(memory_value) > 0),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL CHECK (status IN ('active', 'deleted', 'expired')) DEFAULT 'active',
    source_session_id UUID REFERENCES conversations(session_id) ON DELETE SET NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_personal_memories_active_key
    ON personal_memories(owner_user_id, memory_type, memory_key)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_personal_memories_owner_active
    ON personal_memories(owner_user_id, updated_at DESC)
    WHERE status = 'active';
