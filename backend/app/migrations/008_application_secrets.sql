-- Server-only durable secrets. Values are never returned by HTTP APIs.
CREATE TABLE IF NOT EXISTS application_secrets (
    secret_name TEXT PRIMARY KEY CHECK (length(secret_name) > 0),
    secret_value TEXT NOT NULL CHECK (length(secret_value) >= 64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
