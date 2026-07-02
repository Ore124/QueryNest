import pytest

from app.settings import Settings


def test_postgres_dsn_is_required_for_knowledge_base_metadata():
    with pytest.raises(ValueError, match="POSTGRES_DSN is required"):
        Settings(POSTGRES_DSN="")
