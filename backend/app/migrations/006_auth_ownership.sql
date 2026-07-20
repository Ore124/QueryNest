CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('personal', 'admin')) DEFAULT 'personal',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS owner_user_id UUID REFERENCES users(user_id);

CREATE INDEX IF NOT EXISTS idx_conversations_owner_user_id
    ON conversations(owner_user_id)
    WHERE owner_user_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS audit_events (
    event_id BIGSERIAL PRIMARY KEY,
    actor_user_id UUID NOT NULL REFERENCES users(user_id),
    action TEXT NOT NULL CHECK (length(action) > 0),
    session_id UUID REFERENCES conversations(session_id) ON DELETE SET NULL,
    owner_user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_events_actor_created
    ON audit_events(actor_user_id, created_at DESC);
