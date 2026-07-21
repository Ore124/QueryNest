ALTER TABLE users
    ADD COLUMN IF NOT EXISTS username TEXT,
    ADD COLUMN IF NOT EXISTS password_hash TEXT,
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

-- Existing ownership-only users cannot safely be assigned credentials. They
-- remain inactive until an administrator provisions a username and password.
UPDATE users
SET is_active = FALSE
WHERE username IS NULL OR password_hash IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username
    ON users(username)
    WHERE username IS NOT NULL;
