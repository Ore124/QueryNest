from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .cache import NullJsonCache, RedisJsonCache
from .document_parsers import DocumentParserRouter, MinerUParser, PaddleOcrParser
from .documents import load_documents, split_documents
from .graph import RagService
from .history import ChatHistoryStore, RedisChatHistoryStore
from .index import HybridIndex
from .providers import CachedEmbeddings, get_embeddings
from .reranker import DashScopeReranker
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
base_embeddings = get_embeddings(settings)
embeddings = (
    CachedEmbeddings(
        base_embeddings,
        cache=cache_store,
        model=settings.default_embedding_model,
        dimensions=settings.embedding_dimensions,
        ttl_seconds=settings.redis_cache_ttl_seconds,
    )
    if settings.redis_url
    else base_embeddings
)
reranker = (
    DashScopeReranker(
        api_key=settings.dashscope_api_key,
        api_url=settings.rerank_api_url,
        model=settings.rerank_model,
        timeout_seconds=settings.rerank_timeout_seconds,
        instruct=settings.rerank_instruct,
    )
    if settings.dashscope_api_key
    else None
)
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
    embedding_dimensions=settings.embedding_dimensions,
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
paddleocr_parser = PaddleOcrParser(
    language=settings.paddleocr_language,
    device=settings.paddleocr_device,
)
document_parsers = DocumentParserRouter(
    MinerUParser(
        settings.resolved_mineru_command,
        settings.resolved_mineru_output_dir,
        paddleocr_parser,
        backend=settings.mineru_backend,
        method=settings.mineru_method,
        language=settings.mineru_language,
    ),
    paddleocr_parser,
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
        default_chat_model=settings.default_chat_model,
        default_embedding_model=settings.default_embedding_model,
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
    return [
        ModelInfo(provider="zhipu", model=settings.default_chat_model, role="chat", available=bool(settings.zai_api_key)),
        ModelInfo(provider="zhipu", model=settings.default_embedding_model, role="embedding", available=bool(settings.zai_api_key)),
        ModelInfo(provider="dashscope", model=settings.rerank_model, role="rerank", available=bool(settings.dashscope_api_key)),
        ModelInfo(provider="openai-compatible", model="custom-chat-model", role="chat", available=False),
        ModelInfo(provider="openai-compatible", model="custom-embedding-model", role="embedding", available=False),
    ]


@app.post("/api/warmup", response_model=WarmupResponse)
def warmup() -> WarmupResponse:
    global embedding_warmed
    if embedding_warmed or not settings.zai_api_key:
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
async def ingest_files(files: list[UploadFile] = File(...)) -> IngestResponse:
    upload_dir = settings.root_dir / ".uploads" / str(uuid.uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in files:
        target = upload_dir / Path(uploaded.filename or "uploaded.bin").name
        with target.open("wb") as stream:
            shutil.copyfileobj(uploaded.file, stream)
    return _ingest_from_path(upload_dir, rebuild=True, include_images=True)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not hybrid_index.ready:
        raise HTTPException(status_code=409, detail="Index is not ready. Run /api/ingest/path first.")
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


def _ingest_from_path(path: Path, rebuild: bool, include_images: bool) -> IngestResponse:
    try:
        raw_documents = load_documents(path, document_parsers, include_images=include_images)
        chunks = split_documents(raw_documents, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=400, detail="No supported documents found.")
        hybrid_index.build(chunks, rebuild=rebuild)
        return IngestResponse(
            indexed_chunks=len(chunks),
            source_documents=len(raw_documents),
            scenarios=hybrid_index.scenarios(),
            index_dir=str(settings.resolved_index_dir),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
