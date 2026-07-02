from __future__ import annotations

import os

from app.kb_metadata import PostgresMetadataStore


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN", "").strip()
    if not dsn:
        raise SystemExit("POSTGRES_DSN is required.")
    PostgresMetadataStore(dsn).run_migrations()
    print("PostgreSQL migrations applied.")


if __name__ == "__main__":
    main()
