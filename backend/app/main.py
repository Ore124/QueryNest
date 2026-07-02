from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .cache import NullJsonCache, RedisJsonCache
from .document_parsers import DocumentParserRouter
from .documents import load_documents, split_documents
from .graph import RagService
from .history import ChatHistoryStore, RedisChatHistoryStore
from .image_ocr import PaddleOcrParser
from .index import HybridIndex
from .kb_metadata import PostgresMetadataStore
from .mineru_api_client import MineruApiClient, MineruApiClientConfig
from .pdf_parse_router import PdfParseRouter, PdfParseRouterConfig
from .providers import CachedEmbeddings, get_embeddings
from .reranker import create_reranker
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestPathRequest,
    IngestResponse,
    ModelInfo,
    ScenarioResponse,
    WarmupResponse,
)
from .settings import Settings, get_settings


settings = get_settings()
cache_store = (
    RedisJsonCache(
        settings.redis_url,
        key_prefix=settings.redis_cache_key_prefix,
        default_ttl_seconds=settings.redis_cache_ttl_seconds,
    )
    if settings.redis_url
    else NullJsonCache()
)
metadata_store = PostgresMetadataStore(settings.postgres_dsn) if settings.postgres_dsn else None
if metadata_store is not None:
    metadata_store.run_migrations()
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
    settings.resolved_index_dir,
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
    bm25_backend=settings.retrieval_bm25_backend,
)
hybrid_index.load()
history_store = (
    RedisChatHistoryStore(
        settings.redis_url,
        key_prefix=settings.redis_key_prefix,
        ttl_seconds=settings.redis_session_ttl_seconds,
        max_messages=settings.redis_history_max_messages,
    )
    if settings.redis_url
    else ChatHistoryStore(settings.resolved_sqlite_path)
)
rag_service = RagService(settings, hybrid_index, history_store)
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

app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        index_ready=hybrid_index.ready,
        indexed_chunks=len(hybrid_index.chunks),
        default_chat_model=settings.resolved_llm_model,
        default_embedding_model=settings.resolved_embedding_model,
        history_backend=history_store.backend,
        redis_connected=history_store.ping() if history_store.backend == "redis" else None,
        cache_backend=cache_store.backend,
        cache_connected=cache_store.ping() if cache_store.backend == "redis" else None,
        index_origin=hybrid_index.origin,
        index_build_count=hybrid_index.build_count,
    )


@app.get("/api/scenarios", response_model=ScenarioResponse)
def scenarios() -> ScenarioResponse:
    return ScenarioResponse(scenarios=hybrid_index.scenarios())


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


@app.post("/api/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    include_images: bool = Form(True),
) -> IngestResponse:
    upload_dir = settings.root_dir / ".uploads" / str(uuid.uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in files:
        target = _safe_upload_target(upload_dir, uploaded.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            shutil.copyfileobj(uploaded.file, stream)
    return _ingest_from_path(upload_dir, rebuild=True, include_images=include_images)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not hybrid_index.ready:
        raise HTTPException(status_code=409, detail="Index is not ready. Run ingestion first.")
    try:
        return rag_service.chat(
            message=request.message,
            session_id=request.session_id,
            scenario=request.scenario,
            model=request.model,
            top_k=request.top_k,
        )
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


def _ingest_from_path(path: Path, rebuild: bool, include_images: bool) -> IngestResponse:
    try:
        raw_documents = load_documents(path, document_parsers, include_images=include_images)
        chunks = split_documents(raw_documents, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=400, detail="No supported documents found.")
        hybrid_index.build(chunks, rebuild=rebuild)
        if metadata_store is not None:
            metadata_store.write_index_snapshot(
                kb_id=settings.kb_id,
                index_version=hybrid_index.index_revision,
                raw_documents=raw_documents,
                chunks=chunks,
                embedding_model=settings.resolved_embedding_model,
                chunker_version=settings.chunker_version,
                parser_version=settings.parser_version,
                milvus_collection=settings.milvus_collection_name,
            )
        return IngestResponse(
            indexed_chunks=len(chunks),
            source_documents=len(raw_documents),
            scenarios=hybrid_index.scenarios(),
            index_dir=str(settings.resolved_index_dir),
            notices=_ingest_notices(raw_documents),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _ingest_notices(raw_documents: list[object]) -> list[str]:
    notices: list[str] = []
    seen: set[str] = set()
    for document in raw_documents:
        notice = getattr(document, "parse_notice", None)
        if isinstance(notice, str) and notice and notice not in seen:
            seen.add(notice)
            notices.append(notice)
    return notices
