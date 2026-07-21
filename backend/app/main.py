from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .cache import NullJsonCache, RedisJsonCache
from .auth import Actor, get_current_actor, hash_password, verify_password
from .document_parsers import DocumentParserRouter
from .documents import DocumentChunk, discover_files, load_documents, split_documents
from .graph import RagService
from .history import CachedChatHistoryStore, ConversationAccessDenied, PostgresChatHistoryStore, RedisChatHistoryStore, UsernameAlreadyExists
from .image_ocr import PaddleOcrParser
from .index import HybridIndex, MILVUS_SPARSE_FIELD
from .kb_metadata import ActiveIngestJobExists, IngestJobState, PostgresMetadataStore
from .memory import PostgresSessionMemoryStore, SessionMemoryService
from .personal_memory import NoopPersonalMemoryIndex, PersonalMemoryService, PostgresPersonalMemoryStore
from .mineru_api_client import MineruApiClient, MineruApiClientConfig
from .pdf_parse_router import PdfParseRouter, PdfParseRouterConfig
from .providers import CachedEmbeddings, get_chat_model, get_embeddings
from .rebuild_locks import KbRebuildLockBusy, PostgresAdvisoryKbRebuildLock, RedisKbRebuildLock
from .reranker import create_reranker
from .schemas import (
    ChatRequest,
    ChatResponse,
    AdminConversationResponse,
    AdminUserResponse,
    ChunkListResponse,
    ChunkView,
    CreatedUserResponse,
    CreatePersonalUserRequest,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentView,
    HealthResponse,
    IngestPathRequest,
    IngestResponse,
    LoginRequest,
    ModelInfo,
    ScenarioResponse,
    TokenResponse,
    ConversationResponse,
    ConversationMessageResponse,
    DeleteConversationsRequest,
    PersonalMemoryResponse,
    UpdatePersonalMemoryRequest,
    WarmupResponse,
)
from .settings import Settings, get_settings


logger = logging.getLogger(__name__)
settings = get_settings()
STALE_INGEST_JOB_SECONDS = 1800


def _create_history_store(settings: Settings, metadata_store: PostgresMetadataStore):
    metadata_store.run_migrations()
    settings.auth_jwt_secret = metadata_store.get_or_create_auth_jwt_secret(settings.auth_jwt_secret)
    primary = PostgresChatHistoryStore(settings.postgres_dsn)
    if not settings.redis_url:
        return primary
    cache = RedisChatHistoryStore(
        settings.redis_url,
        key_prefix=settings.redis_key_prefix,
        ttl_seconds=settings.redis_session_ttl_seconds,
        max_messages=settings.history_cache_max_messages,
    )
    return CachedChatHistoryStore(
        primary,
        cache,
        cache_max_messages=settings.history_cache_max_messages,
    )


cache_store = (
    RedisJsonCache(
        settings.redis_url,
        key_prefix=settings.redis_cache_key_prefix,
        default_ttl_seconds=settings.redis_cache_ttl_seconds,
    )
    if settings.redis_url
    else NullJsonCache()
)
metadata_store = PostgresMetadataStore(settings.postgres_dsn)
history_store = _create_history_store(settings, metadata_store)


class _MemoryExtractionChatModel:
    """Create the configured chat model only when a completed turn is extracted."""

    def invoke(self, messages: list[object]) -> object:
        return get_chat_model(
            settings,
            temperature=0.0,
            thinking=False,
            max_tokens=700,
        ).invoke(messages)


session_memory_store = PostgresSessionMemoryStore(settings.postgres_dsn)
session_memory_service = SessionMemoryService(session_memory_store, _MemoryExtractionChatModel())
personal_memory_store = PostgresPersonalMemoryStore(settings.postgres_dsn)
personal_memory_store.run_migrations()
personal_memory_service = PersonalMemoryService(
    personal_memory_store,
    _MemoryExtractionChatModel(),
    index=NoopPersonalMemoryIndex(),
    default_ttl_days=settings.personal_memory_default_ttl_days,
    extraction_max_items=settings.personal_memory_extraction_max_items,
)
rebuild_lock = (
    RedisKbRebuildLock(cache_store.client)
    if isinstance(cache_store, RedisJsonCache)
    else PostgresAdvisoryKbRebuildLock(metadata_store)
)
base_embeddings = get_embeddings(settings)
embeddings = (
    CachedEmbeddings(
        base_embeddings,
        cache=cache_store,
        model=settings.resolved_embedding_model,
        dimensions=settings.resolved_embedding_dimensions,
        ttl_seconds=settings.redis_cache_ttl_seconds,
    )
    if settings.redis_url
    else base_embeddings
)
reranker = create_reranker(settings)
hybrid_index = HybridIndex(
    settings.resolved_artifact_dir,
    embeddings,
    reranker=reranker,
    rerank_candidate_top_k=settings.rerank_candidate_top_k,
    cache=cache_store if settings.redis_url else None,
    cache_ttl_seconds=settings.redis_cache_ttl_seconds,
    milvus_uri=settings.milvus_uri,
    milvus_token=settings.milvus_token,
    milvus_collection_name=settings.milvus_collection_name,
    embedding_dimensions=settings.resolved_embedding_dimensions,
    metadata_store=metadata_store,
    kb_id=settings.kb_id,
)
hybrid_index.load()
rag_service = RagService(
    settings,
    hybrid_index,
    history_store,
    memory_service=session_memory_service,
    personal_memory_service=personal_memory_service,
)
image_ocr_parser = PaddleOcrParser(
    language=settings.paddleocr_language,
    device=settings.paddleocr_device,
)
mineru_api_client = MineruApiClient(MineruApiClientConfig.from_settings(settings))
pdf_parse_router = PdfParseRouter(PdfParseRouterConfig.from_settings(settings), mineru_api_client)
document_parsers = DocumentParserRouter(
    image_ocr_parser,
    pdf_parse_router,
)
embedding_warmed = False
embedding_warmup_lock = threading.Lock()

app = FastAPI(title="QueryNest", version="0.1.0")
_bearer_scheme = HTTPBearer(auto_error=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _primary_history_store() -> PostgresChatHistoryStore:
    return history_store.primary if isinstance(history_store, CachedChatHistoryStore) else history_store


def _current_actor(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> Actor:
    return get_current_actor(credentials, settings, _primary_history_store())


@app.on_event("startup")
def bootstrap_admin() -> None:
    username = settings.auth_bootstrap_admin_username.strip()
    password = settings.auth_bootstrap_admin_password
    if not username or not password:
        return
    primary = _primary_history_store()
    if not isinstance(primary, PostgresChatHistoryStore):
        return
    primary.bootstrap_admin(username, hash_password(password))


@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    user = _primary_history_store().find_user_by_username(request.username.strip())
    if user is None or not bool(user["is_active"]) or not verify_password(request.password, str(user["password_hash"])):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    actor = Actor(user_id=uuid.UUID(str(user["user_id"])), role=str(user["role"]))
    from .auth import create_access_token

    return TokenResponse(access_token=create_access_token(actor, settings))


@app.post("/api/admin/users", response_model=CreatedUserResponse, status_code=status.HTTP_201_CREATED)
def create_personal_user(
    request: CreatePersonalUserRequest, actor: Actor = Depends(_current_actor)
) -> CreatedUserResponse:
    if actor.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access is required.")
    try:
        user = _primary_history_store().create_personal_user(
            str(actor.user_id), request.username, hash_password(request.password)
        )
    except UsernameAlreadyExists as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists.") from exc
    return CreatedUserResponse(**user)


def _require_admin(actor: Actor) -> None:
    if actor.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access is required.")


@app.get("/api/admin/users", response_model=list[AdminUserResponse])
def list_users(actor: Actor = Depends(_current_actor)) -> list[AdminUserResponse]:
    _require_admin(actor)
    return [AdminUserResponse(**user) for user in _primary_history_store().list_users_for_admin()]


@app.get("/api/admin/conversations", response_model=list[AdminConversationResponse])
def list_admin_conversations(
    owner_user_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    actor: Actor = Depends(_current_actor),
) -> list[AdminConversationResponse]:
    _require_admin(actor)
    try:
        conversations = _primary_history_store().list_conversations_for_admin(
            str(actor.user_id), str(owner_user_id), limit
        )
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.") from exc
    return [AdminConversationResponse(**conversation) for conversation in conversations]


@app.get(
    "/api/admin/conversations/{session_id}/messages",
    response_model=list[ConversationMessageResponse],
)
def list_admin_conversation_messages(
    session_id: uuid.UUID,
    limit: int = Query(default=500, ge=1, le=1000),
    actor: Actor = Depends(_current_actor),
) -> list[ConversationMessageResponse]:
    _require_admin(actor)
    try:
        messages = _primary_history_store().load_messages_for_user(
            str(actor.user_id), str(session_id), limit
        )
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    return [ConversationMessageResponse(**message) for message in messages]


@app.get("/api/conversations", response_model=list[ConversationResponse])
def list_conversations(
    limit: int = Query(default=100, ge=1, le=500), actor: Actor = Depends(_current_actor)
) -> list[ConversationResponse]:
    try:
        conversations = _primary_history_store().list_conversations_for_user(str(actor.user_id), limit)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversations not found.") from exc
    return [ConversationResponse(**conversation) for conversation in conversations]


@app.delete("/api/conversations", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversations(
    request: DeleteConversationsRequest, actor: Actor = Depends(_current_actor)
) -> None:
    try:
        _primary_history_store().delete_conversations_for_owner(
            str(actor.user_id), [str(session_id) for session_id in request.session_ids]
        )
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc


@app.get("/api/conversations/{session_id}/messages", response_model=list[ConversationMessageResponse])
def list_conversation_messages(
    session_id: uuid.UUID,
    limit: int = Query(default=500, ge=1, le=1000),
    actor: Actor = Depends(_current_actor),
) -> list[ConversationMessageResponse]:
    try:
        messages = _primary_history_store().load_messages_for_owner(str(actor.user_id), str(session_id), limit)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from exc
    return [ConversationMessageResponse(**message) for message in messages]


def _personal_memory_response(memory: dict[str, Any]) -> PersonalMemoryResponse:
    return PersonalMemoryResponse(
        memory_id=memory["memory_id"],
        memory_type=memory["memory_type"],
        key=memory["key"],
        value=memory["value"],
        confidence=memory["confidence"],
        source_session_id=memory["source_session_id"],
        expires_at=memory["expires_at"],
    )


def _require_explicit_memory_owner(actor: Actor, owner_user_id: uuid.UUID | None) -> str | None:
    """Require an administrator's mutation target before any cross-owner storage access."""
    if actor.role == "admin" and owner_user_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    return str(owner_user_id) if owner_user_id is not None else None


@app.get("/api/memories", response_model=list[PersonalMemoryResponse])
def list_personal_memories(
    owner_user_id: uuid.UUID | None = Query(default=None), actor: Actor = Depends(_current_actor)
) -> list[PersonalMemoryResponse]:
    target_user_id = str(owner_user_id or actor.user_id)
    if owner_user_id is not None and actor.role != "admin" and target_user_id != str(actor.user_id):
        return []
    return [_personal_memory_response(memory) for memory in personal_memory_store.list_for_actor(actor, target_user_id)]


@app.patch("/api/memories/{memory_id}", response_model=PersonalMemoryResponse)
def update_personal_memory(
    memory_id: uuid.UUID, request: UpdatePersonalMemoryRequest,
    owner_user_id: uuid.UUID | None = Query(default=None), actor: Actor = Depends(_current_actor)
) -> PersonalMemoryResponse:
    changes = request.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="At least one field is required.")
    expected_owner_user_id = _require_explicit_memory_owner(actor, owner_user_id)
    column_names = {"key": "memory_key", "value": "memory_value"}
    try:
        memory = personal_memory_store.update_for_actor(
            actor, str(memory_id), {column_names.get(key, key): value for key, value in changes.items()}, expected_owner_user_id
        )
    except Exception as exc:
        from .personal_memory import PersonalMemoryAccessDenied
        if isinstance(exc, PersonalMemoryAccessDenied):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.") from exc
        raise
    return _personal_memory_response(memory)


@app.delete("/api/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_personal_memory(
    memory_id: uuid.UUID, owner_user_id: uuid.UUID | None = Query(default=None), actor: Actor = Depends(_current_actor)
) -> None:
    try:
        expected_owner_user_id = _require_explicit_memory_owner(actor, owner_user_id)
        personal_memory_service.delete_memory(actor, str(memory_id), expected_owner_user_id)
    except Exception as exc:
        from .personal_memory import PersonalMemoryAccessDenied
        if isinstance(exc, PersonalMemoryAccessDenied):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.") from exc
        raise


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    history_cache_connected = _history_cache_connected(history_store)
    return HealthResponse(
        status="ok",
        index_ready=hybrid_index.ready,
        indexed_chunks=len(hybrid_index.chunks),
        default_chat_model=settings.resolved_llm_model,
        default_embedding_model=settings.resolved_embedding_model,
        history_backend=history_store.backend,
        redis_connected=history_cache_connected,
        history_cache_connected=history_cache_connected,
        cache_backend=cache_store.backend,
        cache_connected=cache_store.ping() if cache_store.backend == "redis" else None,
        index_origin=hybrid_index.origin,
        index_build_count=hybrid_index.build_count,
    )


def _history_cache_connected(history_store: object) -> bool | None:
    if not isinstance(history_store, CachedChatHistoryStore):
        return None
    return history_store.cache.ping()


@app.get("/api/scenarios", response_model=ScenarioResponse)
def scenarios() -> ScenarioResponse:
    return ScenarioResponse(scenarios=hybrid_index.scenarios())


@app.get("/api/documents", response_model=DocumentListResponse)
def documents(scenario: str | None = None) -> DocumentListResponse:
    rows = [
        chunk
        for chunk in hybrid_index.chunks
        if not scenario or str(chunk.get("metadata", {}).get("scenario", "")) == scenario
    ]
    documents_by_path: dict[str, dict[str, Any]] = {}
    for chunk in rows:
        metadata = dict(chunk.get("metadata", {}))
        source_path = str(metadata.get("source_path") or "")
        key = source_path or str(metadata.get("source_name") or "")
        if not key:
            key = str(chunk.get("chunk_id", ""))
        record = documents_by_path.setdefault(
            key,
            {
                "source_path": source_path,
                "source_name": str(metadata.get("source_name") or key),
                "file_type": str(metadata.get("file_type") or ""),
                "scenario": str(metadata.get("scenario") or ""),
                "chunk_count": 0,
            },
        )
        record["chunk_count"] = int(record["chunk_count"]) + 1
    views = [
        DocumentView(**record)
        for record in sorted(
            documents_by_path.values(),
            key=lambda item: (str(item["source_name"]).lower(), str(item["source_path"]).lower()),
        )
    ]
    return DocumentListResponse(documents=views, total=len(views))


@app.delete("/api/documents", response_model=DocumentDeleteResponse)
def delete_document(source_path: str = Query(min_length=1)) -> DocumentDeleteResponse:
    with _rebuild_lock_context(settings.kb_id, None):
        existing = _snapshot_chunks(hybrid_index)
        deleted_chunks = [
            chunk
            for chunk in existing
            if str(chunk.metadata.get("source_path", "")) == source_path
        ]
        if not deleted_chunks:
            raise HTTPException(status_code=404, detail="Document not found.")
        remaining = [
            chunk
            for chunk in existing
            if str(chunk.metadata.get("source_path", "")) != source_path
        ]
        hybrid_index.build(remaining)
        metadata_store.write_index_snapshot(
            kb_id=settings.kb_id,
            index_version=hybrid_index.index_revision,
            raw_documents=[],
            chunks=remaining,
            embedding_model=settings.resolved_embedding_model,
            chunker_version=settings.chunker_version,
            parser_version=settings.parser_version,
            milvus_collection=settings.milvus_collection_name,
            milvus_sparse_field=MILVUS_SPARSE_FIELD,
        )
        hybrid_index.metadata_store = metadata_store
        if remaining and not hybrid_index.load():
            raise RuntimeError("Index snapshot was written but PostgreSQL/Milvus hybrid reload failed.")
    return DocumentDeleteResponse(
        source_path=source_path,
        deleted_chunks=len(deleted_chunks),
        remaining_documents=_document_group_count(remaining),
    )


@app.get("/api/chunks", response_model=ChunkListResponse)
def chunks(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    scenario: str | None = None,
    source_path: str | None = None,
) -> ChunkListResponse:
    rows = [
        chunk
        for chunk in hybrid_index.chunks
        if (not scenario or str(chunk.get("metadata", {}).get("scenario", "")) == scenario)
        and (not source_path or str(chunk.get("metadata", {}).get("source_path", "")) == source_path)
    ]
    ordered = sorted(
        rows,
        key=lambda chunk: (
            str(chunk.get("metadata", {}).get("source_name", "")),
            _int_or_none(chunk.get("metadata", {}).get("chunk_index")) or 0,
            str(chunk.get("chunk_id", "")),
        ),
    )
    page = ordered[offset : offset + limit]
    return ChunkListResponse(
        chunks=[_to_chunk_view(chunk) for chunk in page],
        total=len(ordered),
        offset=offset,
        limit=limit,
    )


@app.get("/api/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    rerank_available = settings.resolved_rerank_provider != "none" and bool(settings.resolved_rerank_api_key)
    return [
        ModelInfo(
            provider=settings.resolved_llm_provider,
            model=settings.resolved_llm_model,
            role="chat",
            available=bool(settings.resolved_llm_api_key),
        ),
        ModelInfo(
            provider=settings.resolved_embedding_provider,
            model=settings.resolved_embedding_model,
            role="embedding",
            available=bool(settings.resolved_embedding_api_key),
        ),
        ModelInfo(
            provider=settings.resolved_rerank_provider,
            model=settings.rerank_model if settings.resolved_rerank_provider != "none" else "",
            role="rerank",
            available=rerank_available,
        ),
        ModelInfo(provider="openai-compatible", model="custom-chat-model", role="chat", available=False),
        ModelInfo(provider="openai-compatible", model="custom-embedding-model", role="embedding", available=False),
    ]


@app.post("/api/warmup", response_model=WarmupResponse)
def warmup() -> WarmupResponse:
    global embedding_warmed
    if embedding_warmed or not settings.resolved_embedding_api_key:
        return WarmupResponse(status="ok", embedding_warmed=embedding_warmed)
    with embedding_warmup_lock:
        if not embedding_warmed:
            embeddings.embed_query("企业知识库检索预热")
            embedding_warmed = True
    return WarmupResponse(status="ok", embedding_warmed=True)


@app.post("/api/ingest/path", response_model=IngestResponse)
def ingest_path(request: IngestPathRequest) -> IngestResponse:
    if not request.path.exists() or not request.path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path does not exist or is not a directory: {request.path}")
    return _ingest_from_path(request.path, request.rebuild, request.include_images)


@app.post("/api/ingest/rebuild", response_model=IngestResponse)
def rebuild_ingested_sources() -> IngestResponse:
    return _rebuild_from_indexed_source_roots()


@app.post("/api/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    include_images: bool = Form(True),
) -> IngestResponse:
    upload_dir = settings.root_dir / ".uploads" / str(uuid.uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)
    mime_types: dict[str, str] = {}
    upload_filenames: list[str] = []
    for uploaded in files:
        target = _safe_upload_target(upload_dir, uploaded.filename)
        upload_filenames.append(uploaded.filename or "")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            shutil.copyfileobj(uploaded.file, stream)
        if uploaded.content_type:
            mime_types[str(target.resolve())] = uploaded.content_type
    ingest_root = _uploaded_ingest_root(upload_dir, upload_filenames)
    logger.info(
        "Ingest upload received files=%s upload_dir=%s ingest_root=%s include_images=%s",
        len(upload_filenames),
        upload_dir,
        ingest_root,
        include_images,
    )
    return _ingest_from_path(ingest_root, rebuild=False, include_images=include_images, mime_types=mime_types)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, actor: Actor = Depends(_current_actor)) -> ChatResponse:
    if not hybrid_index.ready:
        raise HTTPException(status_code=409, detail="Index is not ready. Run ingestion first.")
    try:
        return rag_service.chat(
            actor=actor,
            message=request.message,
            session_id=request.session_id,
            scenario=request.scenario,
            model=request.model,
            top_k=request.top_k,
            agentic=request.agentic,
        )
    except ConversationAccessDenied as exc:
        # Do not reveal whether another user's conversation exists.
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _safe_upload_target(upload_dir: Path, filename: str | None) -> Path:
    raw_name = (filename or "uploaded.bin").replace("\\", "/")
    relative_path = PurePosixPath(raw_name)
    invalid_chars = set('<>:"|?*')
    if relative_path.is_absolute():
        raise HTTPException(status_code=400, detail=f"Invalid uploaded filename: {filename}")
    parts = [part for part in relative_path.parts if part not in ("", ".")]
    if not parts or any(part == ".." or any(char in invalid_chars for char in part) for part in parts):
        raise HTTPException(status_code=400, detail=f"Invalid uploaded filename: {filename}")
    return upload_dir.joinpath(*parts)


def _uploaded_ingest_root(upload_dir: Path, filenames: list[str]) -> Path:
    relative_parts = [
        PurePosixPath((filename or "").replace("\\", "/")).parts
        for filename in filenames
    ]
    if not relative_parts or any(len(parts) < 2 for parts in relative_parts):
        return upload_dir
    first_parts = {parts[0] for parts in relative_parts}
    if len(first_parts) != 1:
        return upload_dir
    if not any(len(parts) > 2 for parts in relative_parts):
        return upload_dir
    return upload_dir / next(iter(first_parts))


def _to_chunk_view(chunk: dict[str, Any]) -> ChunkView:
    metadata = dict(chunk.get("metadata", {}))
    return ChunkView(
        chunk_id=str(chunk.get("chunk_id", "")),
        text=str(chunk.get("text", "")),
        source_path=str(metadata.get("source_path", "")),
        source_name=str(metadata.get("source_name", "")),
        file_type=str(metadata.get("file_type", "")),
        scenario=str(metadata.get("scenario", "")),
        section=metadata.get("section"),
        page=_int_or_none(metadata.get("page")),
        content_type=str(metadata.get("content_type", "text")),
        chunk_index=_int_or_none(metadata.get("chunk_index")),
    )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class IngestChangeCounts:
    added_documents: int = 0
    updated_documents: int = 0
    skipped_documents: int = 0
    deleted_documents: int = 0
    moved_documents: int = 0

    @property
    def changed_documents(self) -> int:
        return self.added_documents + self.updated_documents + self.deleted_documents + self.moved_documents

    def as_response_fields(self) -> dict[str, int]:
        return {
            "added_documents": self.added_documents,
            "updated_documents": self.updated_documents,
            "skipped_documents": self.skipped_documents,
            "deleted_documents": self.deleted_documents,
            "moved_documents": self.moved_documents,
        }


def _snapshot_chunks(index: HybridIndex) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=str(row["chunk_id"]),
            text=str(row["text"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in index.chunks
    ]


def _document_identity(chunk: DocumentChunk) -> tuple[str, str]:
    return (
        str(chunk.metadata.get("source_root", "")),
        str(chunk.metadata.get("source_key", "")),
    )


def _document_groups(chunks: list[DocumentChunk]) -> dict[tuple[str, str], list[DocumentChunk]]:
    groups: dict[tuple[str, str], list[DocumentChunk]] = {}
    for chunk in chunks:
        groups.setdefault(_document_identity(chunk), []).append(chunk)
    return groups


def _document_group_count(chunks: list[DocumentChunk]) -> int:
    return len(_document_groups(chunks))


def _file_hash(chunks: list[DocumentChunk]) -> str:
    return str(chunks[0].metadata.get("file_hash", "")) if chunks else ""


def _append_uploaded_chunks(
    existing: list[DocumentChunk], incoming: list[DocumentChunk]
) -> tuple[list[DocumentChunk], IngestChangeCounts]:
    """Append uploaded documents whose content is not already indexed."""
    existing_hashes = {
        _file_hash(document_chunks)
        for document_chunks in _document_groups(existing).values()
    }
    counts = IngestChangeCounts()
    accepted_chunks: list[DocumentChunk] = []
    seen_hashes = set(existing_hashes)
    for document_chunks in _document_groups(incoming).values():
        file_hash = _file_hash(document_chunks)
        if file_hash in seen_hashes:
            counts.skipped_documents += 1
            continue
        seen_hashes.add(file_hash)
        accepted_chunks.extend(document_chunks)
        counts.added_documents += 1
    return existing + accepted_chunks, counts


def _sync_snapshot(
    existing: list[DocumentChunk],
    incoming: list[DocumentChunk],
    source_root: str,
) -> tuple[list[DocumentChunk], IngestChangeCounts]:
    """Create the next snapshot for one authoritative directory source."""
    existing_groups = _document_groups(existing)
    incoming_groups = _document_groups(incoming)
    existing_root_groups = {
        key: chunks for key, chunks in existing_groups.items() if key[0] == source_root and key[1]
    }
    incoming_root_groups = {
        key: chunks for key, chunks in incoming_groups.items() if key[0] == source_root and key[1]
    }
    counts = IngestChangeCounts()
    next_chunks = [
        chunk
        for chunk in existing
        if not (_document_identity(chunk)[0] == source_root and _document_identity(chunk)[1])
    ]
    incoming_keys = set(incoming_root_groups)
    moved_keys: set[tuple[str, str]] = set()

    for key, new_chunks in incoming_root_groups.items():
        old_chunks = existing_root_groups.get(key)
        if old_chunks is not None:
            if _file_hash(old_chunks) == _file_hash(new_chunks):
                next_chunks.extend(old_chunks)
                counts.skipped_documents += 1
            else:
                next_chunks.extend(new_chunks)
                counts.updated_documents += 1
            continue

        moved_key = next(
            (
                old_key
                for old_key, old_chunks in existing_root_groups.items()
                if old_key not in incoming_keys
                and old_key not in moved_keys
                and _file_hash(old_chunks) == _file_hash(new_chunks)
            ),
            None,
        )
        next_chunks.extend(new_chunks)
        if moved_key is None:
            counts.added_documents += 1
        else:
            moved_keys.add(moved_key)
            counts.moved_documents += 1

    for key in existing_root_groups:
        if key not in incoming_keys and key not in moved_keys:
            counts.deleted_documents += 1
    return next_chunks, counts


def _ingest_from_path(
    path: Path,
    rebuild: bool,
    include_images: bool,
    mime_types: dict[str, str] | None = None,
) -> IngestResponse:
    job_state: IngestJobState | None = None
    source = _ingest_source(path, include_images=include_images, mime_types=mime_types)
    try:
        job_state = _claim_ingest_job(source)
        if job_state.status != "queued":
            return _active_ingest_response(job_state)
        metadata_store.update_ingest_job(job_state.job_id, status="parsing")
        metadata_store.upsert_document_status(
            kb_id=settings.kb_id,
            doc_id=source["doc_id"],
            file_name=source["file_name"],
            file_hash=source["file_hash"],
            status="parsing",
            parser="ingest-request",
            parser_version=settings.parser_version,
            metadata=source["metadata"],
        )
        raw_documents = (
            load_documents(path, document_parsers, include_images=include_images, mime_types=mime_types)
            if mime_types is not None
            else load_documents(path, document_parsers, include_images=include_images)
        )
        chunks = split_documents(raw_documents, settings.chunk_size, settings.chunk_overlap)
        logger.info(
            "Ingest parsed path=%s doc_id=%s raw_documents=%s chunks=%s include_images=%s",
            path,
            source["doc_id"],
            len(raw_documents),
            len(chunks),
            include_images,
        )
        if not chunks and (rebuild or mime_types is not None):
            raise HTTPException(status_code=400, detail="No supported documents found.")
        metadata_store.update_ingest_job(job_state.job_id, status="embedding")
        metadata_store.upsert_document_status(
            kb_id=settings.kb_id,
            doc_id=source["doc_id"],
            file_name=source["file_name"],
            file_hash=source["file_hash"],
            status="embedding",
            parser="ingest-request",
            parser_version=settings.parser_version,
            metadata=source["metadata"],
        )
        try:
            with _rebuild_lock_context(settings.kb_id, job_state):
                index_chunks = chunks
                change_counts = IngestChangeCounts()
                if not rebuild:
                    existing_chunks = _snapshot_chunks(hybrid_index)
                    if mime_types is None:
                        index_chunks, change_counts = _sync_snapshot(
                            existing_chunks,
                            chunks,
                            str(path.resolve()),
                        )
                    else:
                        index_chunks, change_counts = _append_uploaded_chunks(existing_chunks, chunks)
                else:
                    change_counts = IngestChangeCounts(added_documents=_document_group_count(chunks))

                if not rebuild and not change_counts.changed_documents:
                    metadata_store.upsert_document_status(
                        kb_id=settings.kb_id,
                        doc_id=source["doc_id"],
                        file_name=source["file_name"],
                        file_hash=source["file_hash"],
                        status="indexed",
                        parser="ingest-request",
                        parser_version=settings.parser_version,
                        metadata=source["metadata"],
                    )
                    metadata_store.update_ingest_job(job_state.job_id, status="completed")
                    return IngestResponse(
                        indexed_chunks=len(hybrid_index.chunks),
                        source_documents=len(raw_documents),
                        scenarios=hybrid_index.scenarios(),
                        artifact_dir=str(settings.resolved_artifact_dir),
                        notices=_ingest_notices(raw_documents),
                        doc_id=source["doc_id"],
                        ingest_job_id=job_state.job_id,
                        ingest_status="completed",
                        **change_counts.as_response_fields(),
                    )
                hybrid_index.build(index_chunks, rebuild=rebuild)
                metadata_store.write_index_snapshot(
                    kb_id=settings.kb_id,
                    index_version=hybrid_index.index_revision,
                    raw_documents=raw_documents,
                    chunks=index_chunks,
                    embedding_model=settings.resolved_embedding_model,
                    chunker_version=settings.chunker_version,
                    parser_version=settings.parser_version,
                    milvus_collection=settings.milvus_collection_name,
                    milvus_sparse_field=MILVUS_SPARSE_FIELD,
                )
                hybrid_index.metadata_store = metadata_store
                if index_chunks and not hybrid_index.load():
                    raise RuntimeError("Index snapshot was written but PostgreSQL/Milvus hybrid reload failed.")
                metadata_store.upsert_document_status(
                    kb_id=settings.kb_id,
                    doc_id=source["doc_id"],
                    file_name=source["file_name"],
                    file_hash=source["file_hash"],
                    status="indexed",
                    parser="ingest-request",
                    parser_version=settings.parser_version,
                    metadata=source["metadata"],
                )
                metadata_store.update_ingest_job(job_state.job_id, status="completed")
                logger.info(
                    "Ingest completed path=%s doc_id=%s index_version=%s chunks=%s scenarios=%s",
                    path,
                    source["doc_id"],
                    hybrid_index.index_revision,
                    len(chunks),
                    hybrid_index.scenarios(),
                )
        except KbRebuildLockBusy as exc:
            _mark_ingest_failed(job_state, source, _safe_error_message(exc))
            return IngestResponse(
                indexed_chunks=len(hybrid_index.chunks),
                source_documents=0,
                scenarios=hybrid_index.scenarios(),
                artifact_dir=str(settings.resolved_artifact_dir),
                notices=[],
                doc_id=source["doc_id"],
                ingest_job_id=job_state.job_id,
                ingest_status="rebuild_locked",
            )
        return IngestResponse(
            indexed_chunks=len(chunks),
            source_documents=len(raw_documents),
            scenarios=hybrid_index.scenarios(),
            artifact_dir=str(settings.resolved_artifact_dir),
            notices=_ingest_notices(raw_documents),
            doc_id=source["doc_id"],
            ingest_job_id=job_state.job_id,
            ingest_status="completed",
            **change_counts.as_response_fields(),
        )
    except HTTPException as exc:
        if job_state is not None:
            _mark_ingest_failed(job_state, source, str(exc.detail))
        raise
    except Exception as exc:
        if job_state is not None:
            _mark_ingest_failed(job_state, source, _safe_error_message(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _rebuild_from_indexed_source_roots() -> IngestResponse:
    existing_chunks = _snapshot_chunks(hybrid_index)
    source_roots = sorted(
        {
            str(chunk.metadata.get("source_root", ""))
            for chunk in existing_chunks
            if chunk.metadata.get("source_root")
        }
    )
    if not source_roots:
        raise HTTPException(status_code=409, detail="No retained source roots are available for rebuild.")
    missing_roots = [source_root for source_root in source_roots if not Path(source_root).is_dir()]
    if missing_roots:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot rebuild: source root is unavailable: {missing_roots[0]}",
        )

    raw_documents = [
        document
        for source_root in source_roots
        for document in load_documents(Path(source_root), document_parsers, include_images=True)
    ]
    chunks = split_documents(raw_documents, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No supported documents found in retained source roots.")

    try:
        with _rebuild_lock_context(settings.kb_id, None):
            hybrid_index.build(chunks, rebuild=True)
            metadata_store.write_index_snapshot(
                kb_id=settings.kb_id,
                index_version=hybrid_index.index_revision,
                raw_documents=raw_documents,
                chunks=chunks,
                embedding_model=settings.resolved_embedding_model,
                chunker_version=settings.chunker_version,
                parser_version=settings.parser_version,
                milvus_collection=settings.milvus_collection_name,
                milvus_sparse_field=MILVUS_SPARSE_FIELD,
            )
            hybrid_index.metadata_store = metadata_store
            if not hybrid_index.load():
                raise RuntimeError("Index snapshot was written but PostgreSQL/Milvus hybrid reload failed.")
    except KbRebuildLockBusy:
        return IngestResponse(
            indexed_chunks=len(hybrid_index.chunks),
            source_documents=0,
            scenarios=hybrid_index.scenarios(),
            artifact_dir=str(settings.resolved_artifact_dir),
            notices=[],
            ingest_status="rebuild_locked",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        indexed_chunks=len(chunks),
        source_documents=len(raw_documents),
        scenarios=hybrid_index.scenarios(),
        artifact_dir=str(settings.resolved_artifact_dir),
        notices=_ingest_notices(raw_documents),
        added_documents=_document_group_count(chunks),
    )


def _claim_ingest_job(source: dict[str, object]) -> IngestJobState:
    metadata_store.expire_stale_ingest_jobs(
        doc_id=str(source["doc_id"]),
        timeout_seconds=STALE_INGEST_JOB_SECONDS,
    )
    active = metadata_store.get_active_ingest_job(str(source["doc_id"]))
    if active is not None:
        return active
    metadata_store.upsert_document_status(
        kb_id=settings.kb_id,
        doc_id=str(source["doc_id"]),
        file_name=str(source["file_name"]),
        file_hash=str(source["file_hash"]),
        status="queued",
        parser="ingest-request",
        parser_version=settings.parser_version,
        metadata=dict(source["metadata"]),
    )
    worker_id = _worker_id()
    try:
        job_id = metadata_store.create_ingest_job(
            kb_id=settings.kb_id,
            doc_id=str(source["doc_id"]),
            worker_id=worker_id,
        )
    except ActiveIngestJobExists:
        active = metadata_store.get_active_ingest_job(str(source["doc_id"]))
        if active is not None:
            return active
        raise
    return IngestJobState(
        job_id=job_id,
        doc_id=str(source["doc_id"]),
        kb_id=settings.kb_id,
        status="queued",
        worker_id=worker_id,
        retry_count=0,
        error_message=None,
    )


def _rebuild_lock_context(kb_id: str, job_state: IngestJobState | None):
    worker_id = job_state.worker_id if job_state and job_state.worker_id else _worker_id()
    if rebuild_lock is None:
        logger.info(
            "KB rebuild lock unavailable kb_id=%s worker_id=%s pid=%s index_version=%s status=disabled",
            kb_id,
            worker_id,
            os.getpid(),
            hybrid_index.index_revision,
        )
        return _NullRebuildLockContext()
    logger.info(
        "KB rebuild lock acquire kb_id=%s worker_id=%s pid=%s index_version=%s status=waiting backend=%s",
        kb_id,
        worker_id,
        os.getpid(),
        hybrid_index.index_revision,
        getattr(rebuild_lock, "backend", "unknown"),
    )
    return _LoggingRebuildLockContext(
        inner=rebuild_lock.acquire(kb_id=kb_id, worker_id=worker_id, timeout_seconds=900),
        kb_id=kb_id,
        worker_id=worker_id,
    )


class _NullRebuildLockContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _LoggingRebuildLockContext:
    def __init__(self, *, inner, kb_id: str, worker_id: str) -> None:
        self.inner = inner
        self.kb_id = kb_id
        self.worker_id = worker_id

    def __enter__(self):
        try:
            self.inner.__enter__()
        except KbRebuildLockBusy:
            logger.info(
                "KB rebuild lock busy kb_id=%s worker_id=%s pid=%s index_version=%s status=busy",
                self.kb_id,
                self.worker_id,
                os.getpid(),
                hybrid_index.index_revision,
            )
            raise
        logger.info(
            "KB rebuild lock acquired kb_id=%s worker_id=%s pid=%s index_version=%s status=acquired",
            self.kb_id,
            self.worker_id,
            os.getpid(),
            hybrid_index.index_revision,
        )
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            return bool(self.inner.__exit__(exc_type, exc, traceback))
        finally:
            status = "released" if exc is None else "released_after_error"
            logger.info(
                "KB rebuild lock release kb_id=%s worker_id=%s pid=%s index_version=%s status=%s",
                self.kb_id,
                self.worker_id,
                os.getpid(),
                hybrid_index.index_revision,
                status,
            )


def _worker_id() -> str:
    return f"worker-{uuid.uuid4().hex[:12]}"


def _active_ingest_response(job_state: IngestJobState) -> IngestResponse:
    return IngestResponse(
        indexed_chunks=len(hybrid_index.chunks),
        source_documents=0,
        scenarios=hybrid_index.scenarios(),
        artifact_dir=str(settings.resolved_artifact_dir),
        notices=[],
        doc_id=job_state.doc_id,
        ingest_job_id=job_state.job_id,
        ingest_status=job_state.status,
    )


def _mark_ingest_failed(job_state: IngestJobState, source: dict[str, object], message: str) -> None:
    metadata_store.upsert_document_status(
        kb_id=settings.kb_id,
        doc_id=str(source["doc_id"]),
        file_name=str(source["file_name"]),
        file_hash=str(source["file_hash"]),
        status="failed",
        parser="ingest-request",
        parser_version=settings.parser_version,
        metadata=dict(source["metadata"]),
        error_message=message,
    )
    metadata_store.fail_ingest_job(job_state.job_id, error_message=message)


def _ingest_source(path: Path, *, include_images: bool, mime_types: dict[str, str] | None = None) -> dict[str, object]:
    fingerprint = _source_fingerprint(path, include_images=include_images, mime_types=mime_types)
    doc_id = hashlib.sha1(f"{settings.kb_id}:{fingerprint}".encode("utf-8")).hexdigest()[:24]
    return {
        "doc_id": doc_id,
        "file_name": path.name,
        "file_hash": fingerprint,
        "metadata": {
            "source_path": str(path.resolve()),
            "include_images": include_images,
            "kind": "ingest_request",
        },
    }


def _source_fingerprint(path: Path, *, include_images: bool, mime_types: dict[str, str] | None = None) -> str:
    digest = hashlib.sha256()
    if not _is_upload_ingest_path(path):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
    digest.update(str(include_images).encode("utf-8"))
    digest.update(b"\0")
    try:
        files = discover_files(path, include_images=include_images, mime_types=mime_types)
    except OSError:
        files = []
    for file_path in files:
        try:
            relative = file_path.resolve().relative_to(path.resolve()).as_posix()
        except ValueError:
            relative = file_path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            with file_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            digest.update(str(file_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_upload_ingest_path(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to((settings.root_dir / ".uploads").resolve())
        return bool(relative.parts)
    except ValueError:
        return False


def _safe_error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:1000]


def _ingest_notices(raw_documents: list[object]) -> list[str]:
    notices: list[str] = []
    seen: set[str] = set()
    for document in raw_documents:
        notice = getattr(document, "parse_notice", None)
        if isinstance(notice, str) and notice and notice not in seen:
            seen.add(notice)
            notices.append(notice)
    return notices
