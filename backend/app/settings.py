from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


load_dotenv(_project_root() / ".env")


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
    embedding_dimensions: int = Field(default=2048, alias="EMBEDDING_DIMENSIONS")
    index_dir: Path = Field(
        default=Path(".rag_index"),
        validation_alias=AliasChoices("INDEX_DIR", "FAISS_INDEX_DIR"),
    )
    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    milvus_collection_name: str = Field(default="rag_chunks", alias="MILVUS_COLLECTION_NAME")
    sqlite_path: Path = Field(default=Path(".sessions/rag_chat.sqlite3"), alias="SQLITE_PATH")
    default_source_path: Path = Field(default=Path(r"D:\Codex Projects\knowledge"), alias="DEFAULT_SOURCE_PATH")
    frontend_origin: str = Field(default="http://localhost:5173", alias="FRONTEND_ORIGIN")
    run_real_api_tests: bool = Field(default=False, alias="RUN_REAL_API_TESTS")
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
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
    mineru_command: Path = Field(default=Path(".mineru-venv/Scripts/mineru.exe"), alias="MINERU_COMMAND")
    mineru_output_dir: Path = Field(default=Path(".mineru_output"), alias="MINERU_OUTPUT_DIR")
    mineru_backend: str = Field(default="pipeline", alias="MINERU_BACKEND")
    mineru_method: str = Field(default="auto", alias="MINERU_METHOD")
    mineru_language: str = Field(default="ch", alias="MINERU_LANGUAGE")
    paddleocr_language: str = Field(default="ch", alias="PADDLEOCR_LANGUAGE")
    paddleocr_device: str = Field(default="cpu", alias="PADDLEOCR_DEVICE")
    redis_url: str = Field(default="", alias="REDIS_URL")
    redis_key_prefix: str = Field(default="rag:chat", alias="REDIS_KEY_PREFIX")
    redis_session_ttl_seconds: int = Field(default=604800, alias="REDIS_SESSION_TTL_SECONDS")
    redis_history_max_messages: int = Field(default=100, alias="REDIS_HISTORY_MAX_MESSAGES")
    redis_cache_key_prefix: str = Field(default="rag:cache", alias="REDIS_CACHE_KEY_PREFIX")
    redis_cache_ttl_seconds: int = Field(default=86400, alias="REDIS_CACHE_TTL_SECONDS")

    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 8
    rrf_k: int = 60
    chunk_size: int = 900
    chunk_overlap: int = 160

    @property
    def root_dir(self) -> Path:
        return _project_root()

    @property
    def resolved_index_dir(self) -> Path:
        return self.index_dir if self.index_dir.is_absolute() else self.root_dir / self.index_dir

    @property
    def resolved_sqlite_path(self) -> Path:
        return self.sqlite_path if self.sqlite_path.is_absolute() else self.root_dir / self.sqlite_path

    @property
    def resolved_mineru_command(self) -> Path:
        return self.mineru_command if self.mineru_command.is_absolute() else self.root_dir / self.mineru_command

    @property
    def resolved_mineru_output_dir(self) -> Path:
        return self.mineru_output_dir if self.mineru_output_dir.is_absolute() else self.root_dir / self.mineru_output_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
