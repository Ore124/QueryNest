from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from langchain_core.embeddings import Embeddings

from .cache import JsonCache
from .documents import DocumentChunk
from .reranker import RerankResult
from .schemas import Source


MILVUS_ID_FIELD = "chunk_id"
MILVUS_TEXT_FIELD = "text"
MILVUS_VECTOR_FIELD = "dense"
MILVUS_SPARSE_FIELD = "sparse"
MILVUS_DENSE_METRIC_TYPE = "COSINE"
MILVUS_SPARSE_METRIC_TYPE = "BM25"


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]: ...


class ChunkMetadataStore(Protocol):
    backend: str

    def load_active_chunks(self, kb_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_rank: int | None = None
    rerank_score: float | None = None


class HybridIndex:
    def __init__(
        self,
        artifact_dir: Path,
        embeddings: Embeddings,
        *,
        reranker: Reranker | None = None,
        rerank_candidate_top_k: int = 20,
        cache: JsonCache | None = None,
        cache_ttl_seconds: int = 86400,
        milvus_uri: str = "http://127.0.0.1:19530",
        milvus_token: str = "",
        milvus_collection_name: str = "rag_chunks",
        embedding_dimensions: int | None = None,
        milvus_client: Any | None = None,
        metadata_store: ChunkMetadataStore | None = None,
        kb_id: str = "default",
    ) -> None:
        self.artifact_dir = artifact_dir
        self.embeddings = embeddings
        self.reranker = reranker
        self.rerank_candidate_top_k = rerank_candidate_top_k
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self.milvus_collection_name = milvus_collection_name
        self.embedding_dimensions = embedding_dimensions
        self._milvus_client = milvus_client
        self.metadata_store = metadata_store
        self.kb_id = kb_id
        self.dense_ready = False
        self.chunks: list[dict[str, Any]] = []
        self.chunk_by_id: dict[str, dict[str, Any]] = {}
        self.origin = "not_loaded"
        self.build_count = 0
        self.index_revision = ""

    @property
    def ready(self) -> bool:
        return self.dense_ready and bool(self.chunks) and bool(self.index_revision)

    def build(self, chunks: list[DocumentChunk], rebuild: bool = True) -> None:
        del rebuild
        chunk_rows = [
            {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": dict(chunk.metadata)}
            for chunk in chunks
        ]
        index_version = self._calculate_chunks_revision(chunk_rows)
        for row in chunk_rows:
            row["metadata"]["kb_id"] = self.kb_id
            row["metadata"]["index_version"] = index_version
        self._build_hybrid_collection(chunk_rows, index_version)
        self.chunks = chunk_rows
        self.chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.index_revision = index_version
        self.origin = "postgresql_milvus"
        self.build_count += 1

    def load(self) -> bool:
        if self.metadata_store is None:
            return False
        try:
            if not self._collection_exists():
                return False
            chunks = self.metadata_store.load_active_chunks(self.kb_id)
            if not chunks:
                return False
            self._load_collection()
        except Exception:
            return False
        self.chunks = chunks
        self.chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.index_revision = self._calculate_revision()
        self.origin = "postgresql_milvus"
        return True

    def search(
        self,
        query: str,
        *,
        scenario: str | None,
        dense_top_k: int,
        bm25_top_k: int,
        final_top_k: int,
        rrf_k: int,
    ) -> tuple[list[Source], dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Index is not ready. Run ingestion first.")
        cache_payload = self._retrieval_cache_payload(
            query=query,
            scenario=scenario,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
        )
        cached = self.cache.get_json("retrieval", cache_payload) if self.cache else None
        if isinstance(cached, dict):
            try:
                cached_sources = [Source(**source) for source in cached.get("sources", [])]
                cached_debug = dict(cached.get("debug", {}))
                cached_debug["cache"] = {
                    "backend": self._cache_backend(),
                    "retrieval": "hit",
                    "embedding_query": None,
                }
                cached_debug["index_operation"] = "search_only"
                cached_debug["index_origin"] = self.origin
                cached_debug["index_build_count"] = self.build_count
                cached_debug["index_revision"] = self.index_revision
                cached_debug["timings_ms"] = {
                    "dense": 0.0,
                    "bm25": 0.0,
                    "fusion": 0.0,
                    "rerank": 0.0,
                    "total": 0.0,
                }
                return cached_sources, cached_debug
            except Exception:
                pass

        started = time.perf_counter()
        query_vector = self.embeddings.embed_query(query)
        dense_hits = self._dense_search(query_vector, dense_top_k, scenario)
        dense_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        bm25_hits = self._bm25_search(query, bm25_top_k, scenario)
        bm25_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        candidate_count = max(final_top_k, self.rerank_candidate_top_k)
        fused = self._hybrid_search(
            query=query,
            query_vector=query_vector,
            dense_top_k=max(dense_top_k, candidate_count),
            bm25_top_k=max(bm25_top_k, candidate_count),
            final_top_k=candidate_count,
            scenario=scenario,
            rrf_k=rrf_k,
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
        )
        fusion_ms = (time.perf_counter() - started) * 1000

        candidates = fused[:candidate_count]
        started = time.perf_counter()
        top_hits, rerank_error = self._rerank(query, candidates, final_top_k)
        rerank_ms = (time.perf_counter() - started) * 1000
        sources = [self._to_source(hit) for hit in top_hits]
        debug = {
            "query": query,
            "scenario": scenario,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "final_top_k": final_top_k,
            "rrf_k": rrf_k,
            "dense_hits": [hit.chunk_id for hit in dense_hits],
            "bm25_hits": [hit.chunk_id for hit in bm25_hits],
            "fused_hits": [hit.chunk_id for hit in candidates],
            "retrieval_backend": "milvus_hybrid",
            "rerank_model": self.reranker.model if self.reranker else None,
            "rerank_applied": self.reranker is not None and rerank_error is None,
            "rerank_error": rerank_error,
            "reranked_hits": [hit.chunk_id for hit in top_hits],
            "index_operation": "search_only",
            "index_origin": self.origin,
            "index_build_count": self.build_count,
            "index_revision": self.index_revision,
            "cache": {
                "backend": self._cache_backend(),
                "retrieval": "miss" if self.cache else "disabled",
                "embedding_query": self._embedding_cache_hit(),
            },
            "timings_ms": {
                "dense": round(dense_ms, 2),
                "bm25": round(bm25_ms, 2),
                "fusion": round(fusion_ms, 2),
                "rerank": round(rerank_ms, 2),
                "total": round(dense_ms + bm25_ms + fusion_ms + rerank_ms, 2),
            },
        }
        if self.cache:
            self.cache.set_json(
                "retrieval",
                cache_payload,
                {
                    "sources": [source.model_dump() for source in sources],
                    "debug": debug,
                },
                ttl_seconds=self.cache_ttl_seconds,
            )
        return sources, debug

    def scenarios(self) -> list[str]:
        return sorted({str(chunk["metadata"].get("scenario", "")) for chunk in self.chunks if chunk["metadata"].get("scenario")})

    def _dense_search(self, query_vector: list[float], top_k: int, scenario: str | None) -> list[SearchHit]:
        results = self._client().search(
            collection_name=self.milvus_collection_name,
            data=[query_vector],
            anns_field=MILVUS_VECTOR_FIELD,
            limit=max(top_k * 3, top_k),
            filter=self._filter_expr(scenario),
            output_fields=[MILVUS_ID_FIELD, MILVUS_TEXT_FIELD, "doc_id", "kb_id", "index_version"],
            search_params={"metric_type": MILVUS_DENSE_METRIC_TYPE},
        )
        return self._results_to_hits(results, rank_field="dense", top_k=top_k)

    def _bm25_search(self, query: str, top_k: int, scenario: str | None) -> list[SearchHit]:
        results = self._client().search(
            collection_name=self.milvus_collection_name,
            data=[query],
            anns_field=MILVUS_SPARSE_FIELD,
            limit=max(top_k * 3, top_k),
            filter=self._filter_expr(scenario),
            output_fields=[MILVUS_ID_FIELD, MILVUS_TEXT_FIELD, "doc_id", "kb_id", "index_version"],
            search_params={"metric_type": MILVUS_SPARSE_METRIC_TYPE},
        )
        return self._results_to_hits(results, rank_field="bm25", top_k=top_k)

    def _hybrid_search(
        self,
        *,
        query: str,
        query_vector: list[float],
        dense_top_k: int,
        bm25_top_k: int,
        final_top_k: int,
        scenario: str | None,
        rrf_k: int,
        dense_hits: list[SearchHit],
        bm25_hits: list[SearchHit],
    ) -> list[SearchHit]:
        client = self._client()
        try:
            from pymilvus import AnnSearchRequest, RRFRanker
        except ImportError as exc:
            raise RuntimeError("pymilvus is required for Milvus hybrid retrieval.") from exc
        filter_expr = self._filter_expr(scenario)
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field=MILVUS_VECTOR_FIELD,
            param={"metric_type": MILVUS_DENSE_METRIC_TYPE},
            limit=dense_top_k,
            filter=filter_expr,
        )
        bm25_req = AnnSearchRequest(
            data=[query],
            anns_field=MILVUS_SPARSE_FIELD,
            param={"metric_type": MILVUS_SPARSE_METRIC_TYPE},
            limit=bm25_top_k,
            filter=filter_expr,
        )
        results = client.hybrid_search(
            collection_name=self.milvus_collection_name,
            reqs=[dense_req, bm25_req],
            ranker=RRFRanker(k=rrf_k),
            limit=final_top_k,
            output_fields=[MILVUS_ID_FIELD, MILVUS_TEXT_FIELD, "doc_id", "kb_id", "index_version"],
        )
        dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
        bm25_by_id = {hit.chunk_id: hit for hit in bm25_hits}
        hits = self._results_to_hits(results, rank_field="hybrid", top_k=final_top_k)
        merged: list[SearchHit] = []
        for hit in hits:
            dense_hit = dense_by_id.get(hit.chunk_id)
            bm25_hit = bm25_by_id.get(hit.chunk_id)
            merged.append(
                SearchHit(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    metadata=hit.metadata,
                    dense_rank=dense_hit.dense_rank if dense_hit else None,
                    bm25_rank=bm25_hit.bm25_rank if bm25_hit else None,
                    dense_score=dense_hit.dense_score if dense_hit else None,
                    bm25_score=bm25_hit.bm25_score if bm25_hit else None,
                    rrf_score=hit.rrf_score,
                )
            )
        return merged

    def _build_hybrid_collection(self, chunks: list[dict[str, Any]], index_version: str) -> None:
        texts = [str(chunk["text"]) for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        if not vectors:
            self.dense_ready = False
            return
        dimension = self.embedding_dimensions or len(vectors[0])
        client = self._client()
        if self._collection_exists():
            client.drop_collection(collection_name=self.milvus_collection_name)
        self._create_hybrid_collection(dimension)
        client.insert(
            collection_name=self.milvus_collection_name,
            data=[
                {
                    MILVUS_ID_FIELD: str(chunk["chunk_id"]),
                    "kb_id": self.kb_id,
                    "doc_id": str(metadata.get("doc_id") or metadata.get("source_path") or ""),
                    "index_version": index_version,
                    "scenario": str(metadata.get("scenario") or ""),
                    "page_no": _int_or_none(metadata.get("page")),
                    "chunk_index": _int_or_none(metadata.get("chunk_index")),
                    MILVUS_TEXT_FIELD: str(chunk["text"]),
                    MILVUS_VECTOR_FIELD: vector,
                }
                for chunk, vector, metadata in (
                    (chunk, vector, dict(chunk.get("metadata", {})))
                    for chunk, vector in zip(chunks, vectors, strict=True)
                )
            ],
        )
        client.flush(collection_name=self.milvus_collection_name)
        self._load_collection()

    def _create_hybrid_collection(self, dimension: int) -> None:
        client = self._client()
        try:
            from pymilvus import DataType, Function, FunctionType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("pymilvus is required to create Milvus hybrid collections.") from exc
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name=MILVUS_ID_FIELD, datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="index_version", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="scenario", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="page_no", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name=MILVUS_TEXT_FIELD, datatype=DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field(field_name=MILVUS_VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name=MILVUS_SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=[MILVUS_TEXT_FIELD],
                output_field_names=[MILVUS_SPARSE_FIELD],
            )
        )
        index_params = client.prepare_index_params()
        index_params.add_index(field_name=MILVUS_VECTOR_FIELD, index_type="AUTOINDEX", metric_type=MILVUS_DENSE_METRIC_TYPE)
        index_params.add_index(field_name=MILVUS_SPARSE_FIELD, index_type="SPARSE_INVERTED_INDEX", metric_type=MILVUS_SPARSE_METRIC_TYPE)
        client.create_collection(collection_name=self.milvus_collection_name, schema=schema, index_params=index_params)

    def _client(self) -> Any:
        if self._milvus_client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise RuntimeError("pymilvus is required to use Milvus hybrid retrieval.") from exc
            kwargs: dict[str, str] = {"uri": self.milvus_uri}
            if self.milvus_token:
                kwargs["token"] = self.milvus_token
            self._milvus_client = MilvusClient(**kwargs)
        return self._milvus_client

    def _collection_exists(self) -> bool:
        client = self._client()
        if hasattr(client, "has_collection"):
            return bool(client.has_collection(collection_name=self.milvus_collection_name))
        if hasattr(client, "collection_exists"):
            return bool(client.collection_exists(collection_name=self.milvus_collection_name))
        return self.milvus_collection_name in client.list_collections()

    def _load_collection(self) -> None:
        client = self._client()
        if hasattr(client, "load_collection"):
            client.load_collection(collection_name=self.milvus_collection_name)
        self.dense_ready = True

    def vector_count(self) -> int | None:
        client = self._client()
        collection = getattr(client, "collections", {}).get(self.milvus_collection_name)
        if isinstance(collection, dict) and isinstance(collection.get("rows"), dict):
            return len(collection["rows"])
        if hasattr(client, "get_collection_stats"):
            stats = client.get_collection_stats(collection_name=self.milvus_collection_name)
            if isinstance(stats, dict) and stats.get("row_count") is not None:
                return int(stats["row_count"])
        return None

    def _filter_expr(self, scenario: str | None) -> str:
        expr = f'kb_id == "{_escape_expr(self.kb_id)}" and index_version == "{_escape_expr(self.index_revision)}"'
        if scenario:
            expr += f' and scenario == "{_escape_expr(scenario)}"'
        return expr

    def _results_to_hits(self, results: Any, *, rank_field: str, top_k: int) -> list[SearchHit]:
        rows = results[0] if results else []
        hits: list[SearchHit] = []
        for result in rows:
            chunk_id = self._search_result_chunk_id(result)
            if chunk_id is None:
                continue
            chunk = self.chunk_by_id.get(chunk_id) or self._chunk_from_entity(result, chunk_id)
            if chunk is None:
                continue
            score = self._search_result_score(result)
            rank = len(hits) + 1
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    text=str(chunk["text"]),
                    metadata=dict(chunk["metadata"]),
                    dense_rank=rank if rank_field == "dense" else None,
                    bm25_rank=rank if rank_field == "bm25" else None,
                    dense_score=score if rank_field == "dense" else None,
                    bm25_score=score if rank_field == "bm25" else None,
                    rrf_score=score if rank_field == "hybrid" and score is not None else 0.0,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def _chunk_from_entity(self, result: Any, chunk_id: str) -> dict[str, Any] | None:
        entity = result.get("entity") if isinstance(result, dict) else getattr(result, "entity", None)
        if not isinstance(entity, dict):
            return None
        text = entity.get(MILVUS_TEXT_FIELD)
        if text is None:
            return None
        metadata = {
            "chunk_id": chunk_id,
            "source_path": entity.get("doc_id", ""),
            "source_name": entity.get("doc_id", ""),
            "file_type": "",
            "scenario": entity.get("scenario", ""),
            "page": entity.get("page_no"),
            "chunk_index": entity.get("chunk_index"),
            "kb_id": entity.get("kb_id", self.kb_id),
            "index_version": entity.get("index_version", self.index_revision),
        }
        return {"chunk_id": chunk_id, "text": text, "metadata": metadata}

    @staticmethod
    def _search_result_chunk_id(result: Any) -> str | None:
        if isinstance(result, dict):
            entity = result.get("entity")
            if isinstance(entity, dict) and entity.get(MILVUS_ID_FIELD) is not None:
                return str(entity[MILVUS_ID_FIELD])
            if result.get(MILVUS_ID_FIELD) is not None:
                return str(result[MILVUS_ID_FIELD])
            if result.get("id") is not None:
                return str(result["id"])
        entity = getattr(result, "entity", None)
        if isinstance(entity, dict) and entity.get(MILVUS_ID_FIELD) is not None:
            return str(entity[MILVUS_ID_FIELD])
        value = getattr(result, MILVUS_ID_FIELD, None)
        if value is not None:
            return str(value)
        value = getattr(result, "id", None)
        return str(value) if value is not None else None

    @staticmethod
    def _search_result_score(result: Any) -> float | None:
        if isinstance(result, dict):
            value = result.get("distance", result.get("score"))
        else:
            value = getattr(result, "distance", getattr(result, "score", None))
        return float(value) if value is not None else None

    def _to_source(self, hit: SearchHit) -> Source:
        metadata = hit.metadata
        return Source(
            chunk_id=hit.chunk_id,
            text=hit.text,
            source_path=str(metadata.get("source_path", "")),
            source_name=str(metadata.get("source_name", "")),
            file_type=str(metadata.get("file_type", "")),
            scenario=str(metadata.get("scenario", "")),
            section=metadata.get("section"),
            page=metadata.get("page"),
            content_type=str(metadata.get("content_type", "text")),
            parser=str(metadata.get("parser", "text")),
            table_id=metadata.get("table_id"),
            table_markdown=metadata.get("table_markdown"),
            table_json=metadata.get("table_json"),
            table_html=metadata.get("table_html"),
            dense_rank=hit.dense_rank,
            bm25_rank=hit.bm25_rank,
            dense_score=hit.dense_score,
            bm25_score=hit.bm25_score,
            rrf_score=hit.rrf_score,
            rerank_rank=hit.rerank_rank,
            rerank_score=hit.rerank_score,
        )

    def _rerank(
        self,
        query: str,
        candidates: list[SearchHit],
        final_top_k: int,
    ) -> tuple[list[SearchHit], str | None]:
        if self.reranker is None:
            return candidates[:final_top_k], None
        try:
            results = self.reranker.rerank(
                query,
                [hit.text for hit in candidates],
                top_n=final_top_k,
            )
            reranked = [
                replace(
                    candidates[result.index],
                    rerank_rank=rank,
                    rerank_score=result.score,
                )
                for rank, result in enumerate(results, start=1)
            ]
            return reranked[:final_top_k], None
        except Exception as exc:
            return candidates[:final_top_k], f"{type(exc).__name__}: {exc}"

    def _retrieval_cache_payload(
        self,
        *,
        query: str,
        scenario: str | None,
        dense_top_k: int,
        bm25_top_k: int,
        final_top_k: int,
        rrf_k: int,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "scenario": scenario,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "final_top_k": final_top_k,
            "rrf_k": rrf_k,
            "retrieval_backend": "milvus_hybrid",
            "reranker_model": self.reranker.model if self.reranker else None,
            "rerank_candidate_top_k": self.rerank_candidate_top_k,
            "index_revision": self.index_revision,
        }

    def _calculate_revision(self) -> str:
        return self._calculate_chunks_revision(self.chunks)

    @staticmethod
    def _calculate_chunks_revision(chunks: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for chunk in sorted(chunks, key=lambda item: str(item["chunk_id"])):
            metadata = dict(chunk.get("metadata", {}))
            metadata.pop("kb_id", None)
            metadata.pop("index_version", None)
            digest.update(str(chunk["chunk_id"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk["text"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            )
            digest.update(b"\n")
        return digest.hexdigest()

    def _cache_backend(self) -> str:
        return self.cache.backend if self.cache else "disabled"

    def _embedding_cache_hit(self) -> bool | None:
        value = getattr(self.embeddings, "last_query_cache_hit", None)
        return value if isinstance(value, bool) else None


def reciprocal_rank_fusion(
    dense_hits: list[SearchHit],
    bm25_hits: list[SearchHit],
    *,
    rrf_k: int,
) -> list[SearchHit]:
    by_id: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        by_id[hit.chunk_id] = SearchHit(
            chunk_id=hit.chunk_id,
            text=hit.text,
            metadata=hit.metadata,
            dense_rank=hit.dense_rank or rank,
            bm25_rank=hit.bm25_rank,
            dense_score=hit.dense_score,
            bm25_score=hit.bm25_score,
        )
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, hit in enumerate(bm25_hits, start=1):
        existing = by_id.get(hit.chunk_id)
        if existing:
            by_id[hit.chunk_id] = SearchHit(
                chunk_id=existing.chunk_id,
                text=existing.text,
                metadata=existing.metadata,
                dense_rank=existing.dense_rank,
                bm25_rank=hit.bm25_rank or rank,
                dense_score=existing.dense_score,
                bm25_score=hit.bm25_score,
            )
        else:
            by_id[hit.chunk_id] = SearchHit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                metadata=hit.metadata,
                dense_rank=hit.dense_rank,
                bm25_rank=hit.bm25_rank or rank,
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
            )
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    fused = [
        SearchHit(
            chunk_id=hit.chunk_id,
            text=hit.text,
            metadata=hit.metadata,
            dense_rank=hit.dense_rank,
            bm25_rank=hit.bm25_rank,
            dense_score=hit.dense_score,
            bm25_score=hit.bm25_score,
            rrf_score=scores[chunk_id],
        )
        for chunk_id, hit in by_id.items()
    ]
    return sorted(fused, key=lambda hit: (-hit.rrf_score, hit.dense_rank or 10_000, hit.bm25_rank or 10_000, hit.chunk_id))


def _escape_expr(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
