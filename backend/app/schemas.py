from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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


class IngestPathRequest(BaseModel):
    path: Path
    rebuild: bool = True
    include_images: bool = True


class IngestResponse(BaseModel):
    indexed_chunks: int
    source_documents: int
    scenarios: list[str]
    index_dir: str
    notices: list[str] = Field(default_factory=list)
    doc_id: str | None = None
    ingest_job_id: str | None = None
    ingest_status: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    scenario: str | None = None
    model: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


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
