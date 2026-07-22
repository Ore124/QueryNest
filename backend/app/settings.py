from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LLM_BASE_URLS = {
    "zhipu": "https://api.z.ai/api/paas/v4/",
    "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
}
EMBEDDING_BASE_URLS = {
    "zhipu": "https://api.z.ai/api/paas/v4/",
    "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
}
RERANK_BASE_URLS = {
    "bailian": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    "zhipu": "https://api.z.ai/api/paas/v4/rerank",
}
DEFAULT_EMBEDDING_BATCH_SIZE = 64
BAILIAN_EMBEDDING_BATCH_SIZE = 10


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


load_dotenv(_project_root() / ".env")


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _provider(value: str, *, allowed: set[str], role: str) -> str:
    provider = _clean(value).lower()
    if provider not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {role} provider '{value}'. Expected one of: {choices}.")
    return provider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    zai_api_key: str = Field(default="", alias="ZAI_API_KEY")
    zai_api_base: str = Field(default="https://api.z.ai/api/paas/v4/", alias="ZAI_API_BASE")
    default_chat_model: str = Field(default="glm-5.2", alias="DEFAULT_CHAT_MODEL")
    ragas_model: str = Field(default="glm-4.7", alias="RAGAS_MODEL")
    default_embedding_model: str = Field(default="embedding-3", alias="DEFAULT_EMBEDDING_MODEL")
    embedding_dimensions: int | None = Field(default=None, alias="EMBEDDING_DIMENSIONS")
    llm_provider: str = Field(default="", alias="LLM_PROVIDER")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    embedding_provider: str = Field(default="", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="", alias="EMBEDDING_MODEL")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="", alias="EMBEDDING_BASE_URL")
    artifact_dir: Path = Field(
        default=Path(".rag_artifacts"),
        validation_alias=AliasChoices("ARTIFACT_DIR"),
    )
    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    milvus_collection_name: str = Field(default="rag_chunks", alias="MILVUS_COLLECTION_NAME")
    postgres_dsn: str = Field(default="postgresql://rag:rag_dev_password@127.0.0.1:5432/rag", alias="POSTGRES_DSN")
    kb_id: str = Field(default="default", alias="KB_ID")
    parser_version: str = Field(default="document-parsers-v1", alias="PARSER_VERSION")
    chunker_version: str = Field(default="recursive-character-v1", alias="CHUNKER_VERSION")
    default_source_path: Path = Field(default=Path(r"D:\Codex Projects\knowledge"), alias="DEFAULT_SOURCE_PATH")
    frontend_origin: str = Field(default="http://localhost:5173", alias="FRONTEND_ORIGIN")
    run_real_api_tests: bool = Field(default=False, alias="RUN_REAL_API_TESTS")
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    rerank_provider: str = Field(default="", alias="RERANK_PROVIDER")
    rerank_api_key: str = Field(default="", alias="RERANK_API_KEY")
    rerank_base_url: str = Field(default="", alias="RERANK_BASE_URL")
    rerank_api_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        alias="RERANK_API_URL",
    )
    rerank_model: str = Field(default="qwen3-vl-rerank", alias="RERANK_MODEL")
    rerank_candidate_top_k: int = Field(default=20, alias="RERANK_CANDIDATE_TOP_K")
    rerank_timeout_seconds: float = Field(default=120.0, alias="RERANK_TIMEOUT_SECONDS")
    rerank_instruct: str = Field(
        default="根据用户问题，判断候选知识库片段对回答问题的相关性，并优先返回直接包含答案依据的片段。",
        alias="RERANK_INSTRUCT",
    )
    adaptive_context_enabled: bool = Field(default=True, alias="ADAPTIVE_CONTEXT_ENABLED")
    adaptive_context_min_sources: int = Field(
        default=2, ge=1, le=20, alias="ADAPTIVE_CONTEXT_MIN_SOURCES"
    )
    adaptive_context_score_ratio: float = Field(
        default=0.5, gt=0.0, le=1.0, alias="ADAPTIVE_CONTEXT_SCORE_RATIO"
    )
    adaptive_context_max_chars: int = Field(
        default=7200, ge=1, alias="ADAPTIVE_CONTEXT_MAX_CHARS"
    )
    mineru_api_base: str = Field(default="https://mineru.net/api/v4", alias="MINERU_API_BASE")
    mineru_api_token: str = Field(default="", alias="MINERU_API_TOKEN")
    mineru_api_poll_interval_seconds: float = Field(default=3.0, alias="MINERU_API_POLL_INTERVAL_SECONDS")
    mineru_api_timeout_seconds: float = Field(default=600.0, alias="MINERU_API_TIMEOUT_SECONDS")
    mineru_api_enable_formula: bool = Field(default=True, alias="MINERU_API_ENABLE_FORMULA")
    mineru_api_enable_table: bool = Field(default=True, alias="MINERU_API_ENABLE_TABLE")
    mineru_api_is_ocr: bool = Field(default=True, alias="MINERU_API_IS_OCR")
    pdf_parse_strategy: str = Field(default="hybrid", alias="PDF_PARSE_STRATEGY")
    pdf_complex_page_ratio_threshold: float = Field(default=0.35, alias="PDF_COMPLEX_PAGE_RATIO_THRESHOLD")
    pdf_pymupdf_min_quality_score: float = Field(default=0.6, alias="PDF_PYMUPDF_MIN_QUALITY_SCORE")
    pdf_enable_mineru_fallback: bool = Field(default=True, alias="PDF_ENABLE_MINERU_FALLBACK")
    paddleocr_language: str = Field(default="ch", alias="PADDLEOCR_LANGUAGE")
    paddleocr_device: str = Field(default="cpu", alias="PADDLEOCR_DEVICE")
    redis_url: str = Field(default="", alias="REDIS_URL")
    redis_key_prefix: str = Field(default="rag:chat", alias="REDIS_KEY_PREFIX")
    redis_session_ttl_seconds: int = Field(default=604800, alias="REDIS_SESSION_TTL_SECONDS")
    history_cache_max_messages: int = Field(
        default=100,
        validation_alias=AliasChoices("HISTORY_CACHE_MAX_MESSAGES", "REDIS_HISTORY_MAX_MESSAGES"),
    )
    redis_cache_key_prefix: str = Field(default="rag:cache", alias="REDIS_CACHE_KEY_PREFIX")
    redis_cache_ttl_seconds: int = Field(default=86400, alias="REDIS_CACHE_TTL_SECONDS")
    # Optional deployment override.  When absent, startup obtains a durable
    # server-only secret from PostgreSQL after migrations have run.
    auth_jwt_secret: str = Field(default="", alias="AUTH_JWT_SECRET", validate_default=True)
    auth_jwt_issuer: str = Field(default="rag-knowledge-assistant", alias="AUTH_JWT_ISSUER")
    auth_jwt_expiry_minutes: int = Field(default=60, alias="AUTH_JWT_EXPIRY_MINUTES")
    auth_bootstrap_admin_username: str = Field(default="admin", alias="AUTH_BOOTSTRAP_ADMIN_USERNAME")
    auth_bootstrap_admin_password: str = Field(default="12345", alias="AUTH_BOOTSTRAP_ADMIN_PASSWORD")
    personal_memory_enabled: bool = Field(default=True, alias="PERSONAL_MEMORY_ENABLED")
    personal_memory_retrieval_max_items: int = Field(default=5, alias="PERSONAL_MEMORY_RETRIEVAL_MAX_ITEMS")
    personal_memory_default_ttl_days: int = Field(default=90, alias="PERSONAL_MEMORY_DEFAULT_TTL_DAYS")
    personal_memory_extraction_max_items: int = Field(default=5, alias="PERSONAL_MEMORY_EXTRACTION_MAX_ITEMS")

    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 8
    rrf_k: int = 60
    chunk_size: int = 900
    chunk_overlap: int = 160

    @field_validator("embedding_dimensions", mode="before")
    @classmethod
    def _empty_embedding_dimensions(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("postgres_dsn")
    @classmethod
    def _require_postgres_dsn(cls, value: str) -> str:
        dsn = value.strip()
        if not dsn:
            raise ValueError("POSTGRES_DSN is required. PostgreSQL is the knowledge-base metadata store.")
        return dsn

    @field_validator("auth_jwt_secret")
    @classmethod
    def _validate_configured_auth_jwt_secret(cls, value: str) -> str:
        secret = value.strip()
        if not secret:
            return ""
        insecure_values = {"change-me", "replace-with-a-secure-secret", "your-secret-here"}
        if secret.lower() in insecure_values:
            raise ValueError("AUTH_JWT_SECRET must not use a default placeholder.")
        if len(secret) < 32:
            raise ValueError("AUTH_JWT_SECRET must be at least 32 characters long.")
        return secret

    @field_validator("auth_jwt_expiry_minutes")
    @classmethod
    def _require_positive_auth_expiry(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("AUTH_JWT_EXPIRY_MINUTES must be greater than zero.")
        return value

    @field_validator("personal_memory_retrieval_max_items", "personal_memory_extraction_max_items")
    @classmethod
    def _bound_personal_memory_items(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("Personal-memory item limits must be between 1 and 5.")
        return value

    @field_validator("personal_memory_default_ttl_days")
    @classmethod
    def _bound_personal_memory_ttl(cls, value: int) -> int:
        if not 1 <= value <= 365:
            raise ValueError("PERSONAL_MEMORY_DEFAULT_TTL_DAYS must be between 1 and 365.")
        return value

    @property
    def root_dir(self) -> Path:
        return _project_root()

    @property
    def resolved_artifact_dir(self) -> Path:
        return self.artifact_dir if self.artifact_dir.is_absolute() else self.root_dir / self.artifact_dir

    @property
    def resolved_llm_provider(self) -> str:
        return _provider(_clean(self.llm_provider) or "zhipu", allowed=set(LLM_BASE_URLS), role="LLM")

    @property
    def resolved_llm_model(self) -> str:
        return _clean(self.llm_model) or self.default_chat_model

    @property
    def resolved_llm_api_key(self) -> str:
        provider = self.resolved_llm_provider
        if _clean(self.llm_api_key):
            return _clean(self.llm_api_key)
        if provider == "zhipu":
            return _clean(self.zai_api_key)
        if provider == "bailian":
            return _clean(self.dashscope_api_key)
        return ""

    @property
    def resolved_llm_base_url(self) -> str:
        provider = self.resolved_llm_provider
        if _clean(self.llm_base_url):
            return _clean(self.llm_base_url)
        if provider == "zhipu" and _clean(self.zai_api_base):
            return _clean(self.zai_api_base)
        return LLM_BASE_URLS[provider]

    @property
    def resolved_embedding_provider(self) -> str:
        provider = _clean(self.embedding_provider) or "zhipu"
        return _provider(provider, allowed=set(EMBEDDING_BASE_URLS), role="embedding")

    @property
    def resolved_embedding_model(self) -> str:
        return _clean(self.embedding_model) or self.default_embedding_model

    @property
    def resolved_embedding_api_key(self) -> str:
        provider = self.resolved_embedding_provider
        if _clean(self.embedding_api_key):
            return _clean(self.embedding_api_key)
        if provider == "zhipu":
            return _clean(self.zai_api_key)
        if provider == "bailian":
            return _clean(self.dashscope_api_key)
        return ""

    @property
    def resolved_embedding_base_url(self) -> str:
        provider = self.resolved_embedding_provider
        if _clean(self.embedding_base_url):
            return _clean(self.embedding_base_url)
        if provider == "zhipu" and _clean(self.zai_api_base):
            return _clean(self.zai_api_base)
        return EMBEDDING_BASE_URLS[provider]

    @property
    def resolved_embedding_dimensions(self) -> int | None:
        return self.embedding_dimensions

    @property
    def resolved_embedding_batch_size(self) -> int:
        if self.resolved_embedding_provider == "bailian":
            return BAILIAN_EMBEDDING_BATCH_SIZE
        return DEFAULT_EMBEDDING_BATCH_SIZE

    @property
    def resolved_rerank_provider(self) -> str:
        provider = _clean(self.rerank_provider)
        if not provider:
            provider = "bailian" if _clean(self.rerank_api_key) or _clean(self.dashscope_api_key) else "none"
        return _provider(provider, allowed={"bailian", "none", "zhipu"}, role="rerank")

    @property
    def resolved_rerank_api_key(self) -> str:
        provider = self.resolved_rerank_provider
        if provider == "none":
            return ""
        if _clean(self.rerank_api_key):
            return _clean(self.rerank_api_key)
        if provider == "bailian":
            return _clean(self.dashscope_api_key)
        if provider == "zhipu":
            return _clean(self.zai_api_key)
        return ""

    @property
    def resolved_rerank_base_url(self) -> str:
        provider = self.resolved_rerank_provider
        if provider == "none":
            return ""
        if _clean(self.rerank_base_url):
            return _clean(self.rerank_base_url)
        if provider == "bailian" and _clean(self.rerank_api_url):
            return _clean(self.rerank_api_url)
        return RERANK_BASE_URLS[provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
