from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    chunk_id: str
    text: str
    source_path: str
    source_name: str
    file_type: str
    scenario: str
    section: str | None = None
    page: int | None = None
    content_type: str = "text"
    parser: str = "text"
    table_id: str | None = None
    table_markdown: str | None = None
    table_json: str | None = None
    table_html: str | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float
    rerank_rank: int | None = None
    rerank_score: float | None = None


class ChunkView(BaseModel):
    chunk_id: str
    text: str
    source_path: str
    source_name: str
    file_type: str
    scenario: str
    section: str | None = None
    page: int | None = None
    content_type: str = "text"
    chunk_index: int | None = None


class ChunkListResponse(BaseModel):
    chunks: list[ChunkView]
    total: int
    offset: int
    limit: int


class DocumentView(BaseModel):
    source_path: str
    source_name: str
    file_type: str
    scenario: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentView]
    total: int


class DocumentDeleteResponse(BaseModel):
    source_path: str
    deleted_chunks: int
    remaining_documents: int


class IngestPathRequest(BaseModel):
    path: Path
    rebuild: bool = False
    include_images: bool = True


class IngestResponse(BaseModel):
    indexed_chunks: int
    source_documents: int
    scenarios: list[str]
    artifact_dir: str
    notices: list[str] = Field(default_factory=list)
    doc_id: str | None = None
    ingest_job_id: str | None = None
    ingest_status: str | None = None
    added_documents: int = 0
    updated_documents: int = 0
    skipped_documents: int = 0
    deleted_documents: int = 0
    moved_documents: int = 0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    scenario: str | None = None
    model: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    agentic: bool = False


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class CreatePersonalUserRequest(LoginRequest):
    @field_validator("username")
    @classmethod
    def username_must_not_be_blank(cls, value: str) -> str:
        username = value.strip()
        if not username:
            raise ValueError("username must not be blank")
        return username


class CreatedUserResponse(BaseModel):
    user_id: str
    username: str
    role: str


class AdminUserResponse(CreatedUserResponse):
    is_active: bool


class AdminConversationResponse(BaseModel):
    session_id: str
    owner_user_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationResponse(BaseModel):
    session_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationMessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime


class DeleteConversationsRequest(BaseModel):
    session_ids: list[UUID] = Field(min_length=1, max_length=100)

    @field_validator("session_ids")
    @classmethod
    def session_ids_must_be_unique(cls, session_ids: list[UUID]) -> list[UUID]:
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("session_ids must not contain duplicates")
        return session_ids


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PersonalMemoryResponse(BaseModel):
    memory_id: str
    memory_type: Literal["preference", "profile", "fact"]
    key: str
    value: str
    confidence: float = Field(ge=0, le=1)
    source_session_id: str | None = None
    expires_at: datetime | None = None


class UpdatePersonalMemoryRequest(BaseModel):
    memory_type: Literal["preference", "profile", "fact"] | None = None
    key: str | None = Field(default=None, min_length=1, max_length=120)
    value: str | None = Field(default=None, min_length=1, max_length=500)
    confidence: float | None = Field(default=None, ge=0, le=1)
    expires_at: datetime | None = None

    @field_validator("key", "value")
    @classmethod
    def text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("expires_at")
    @classmethod
    def expiry_must_be_within_retention_window(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        now = datetime.now(timezone.utc)
        if value <= now or value > now + timedelta(days=365):
            raise ValueError("expires_at must be within the next 365 days")
        return value


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]
    retrieval_debug: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    indexed_chunks: int
    default_chat_model: str
    default_embedding_model: str
    history_backend: str
    redis_connected: bool | None = None
    history_cache_connected: bool | None = None
    cache_backend: str
    cache_connected: bool | None = None
    index_origin: str
    index_build_count: int


class ModelInfo(BaseModel):
    provider: str
    model: str
    role: str
    available: bool


class ScenarioResponse(BaseModel):
    scenarios: list[str]


class WarmupResponse(BaseModel):
    status: str
    embedding_warmed: bool
