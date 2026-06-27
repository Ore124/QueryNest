from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from .cache import JsonCache
from .documents import DocumentChunk
from .reranker import RerankResult
from .schemas import Source
from .tokenizer import tokenize_for_bm25


MILVUS_ID_FIELD = "chunk_id"
MILVUS_VECTOR_FIELD = "embedding"
MILVUS_METRIC_TYPE = "L2"


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]: ...


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
        index_dir: Path,
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
    ) -> None:
        self.index_dir = index_dir
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
        self.dense_ready = False
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []
        self.chunks: list[dict[str, Any]] = []
        self.chunk_by_id: dict[str, dict[str, Any]] = {}
        self.origin = "not_loaded"
        self.build_count = 0
        self.index_revision = ""

    @property
    def ready(self) -> bool:
        return self.dense_ready and self.bm25 is not None and bool(self.chunks)

    def build(self, chunks: list[DocumentChunk], rebuild: bool = True) -> None:
        if rebuild and self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        self._build_dense_collection(texts, ids)
        self.tokenized_corpus = [tokenize_for_bm25(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunks = [
            {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
            for chunk in chunks
        ]
        self.chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.index_revision = self._calculate_revision()
        self.origin = "rebuilt"
        self.build_count += 1
        self._save_bm25()
        self._save_chunks()

    def load(self) -> bool:
        bm25_path = self.index_dir / "bm25.pkl"
        chunks_path = self.index_dir / "chunks.jsonl"
        if not bm25_path.exists() or not chunks_path.exists():
            return False
        try:
            if not self._collection_exists():
                return False
            self._load_dense_collection()
        except Exception:
            return False
        with bm25_path.open("rb") as stream:
            payload = pickle.load(stream)
        self.tokenized_corpus = payload["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunks = []
        with chunks_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    self.chunks.append(json.loads(line))
        self.chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.index_revision = self._calculate_revision()
        self.origin = "milvus"
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
        dense_hits = self._dense_search(query, dense_top_k, scenario)
        dense_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        bm25_hits = self._bm25_search(query, bm25_top_k, scenario)
        bm25_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=rrf_k)
        fusion_ms = (time.perf_counter() - started) * 1000
        candidate_count = max(final_top_k, self.rerank_candidate_top_k)
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

    def _dense_search(self, query: str, top_k: int, scenario: str | None) -> list[SearchHit]:
        assert self.dense_ready
        results = self._client().search(
            collection_name=self.milvus_collection_name,
            data=[self.embeddings.embed_query(query)],
            anns_field=MILVUS_VECTOR_FIELD,
            limit=max(top_k * 3, top_k),
            output_fields=[MILVUS_ID_FIELD],
            search_params={"metric_type": MILVUS_METRIC_TYPE},
        )
        dense_results = results[0] if results else []
        hits: list[SearchHit] = []
        for result in dense_results:
            chunk_id = self._search_result_chunk_id(result)
            if chunk_id is None:
                continue
            chunk = self.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            metadata = dict(chunk["metadata"])
            if scenario and metadata.get("scenario") != scenario:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    text=chunk["text"],
                    metadata=metadata,
                    dense_rank=len(hits) + 1,
                    dense_score=self._search_result_score(result),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def _bm25_search(self, query: str, top_k: int, scenario: str | None) -> list[SearchHit]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores(tokenize_for_bm25(query))
        candidates = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        hits: list[SearchHit] = []
        for index, score in candidates:
            chunk = self.chunks[index]
            metadata = dict(chunk["metadata"])
            if scenario and metadata.get("scenario") != scenario:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    metadata=metadata,
                    bm25_rank=len(hits) + 1,
                    bm25_score=float(score),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def _build_dense_collection(self, texts: list[str], ids: list[str]) -> None:
        vectors = self.embeddings.embed_documents(texts)
        if not vectors:
            self.dense_ready = False
            return
        dimension = self.embedding_dimensions or len(vectors[0])
        client = self._client()
        if self._collection_exists():
            client.drop_collection(collection_name=self.milvus_collection_name)
        client.create_collection(
            collection_name=self.milvus_collection_name,
            dimension=dimension,
            primary_field_name=MILVUS_ID_FIELD,
            vector_field_name=MILVUS_VECTOR_FIELD,
            id_type="string",
            metric_type=MILVUS_METRIC_TYPE,
            auto_id=False,
            max_length=512,
        )
        client.insert(
            collection_name=self.milvus_collection_name,
            data=[
                {
                    MILVUS_ID_FIELD: chunk_id,
                    MILVUS_VECTOR_FIELD: vector,
                }
                for chunk_id, vector in zip(ids, vectors, strict=True)
            ],
        )
        client.flush(collection_name=self.milvus_collection_name)
        self._load_dense_collection()

    def _client(self) -> Any:
        if self._milvus_client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise RuntimeError("pymilvus is required to use Milvus dense retrieval.") from exc
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

    def _load_dense_collection(self) -> None:
        client = self._client()
        if hasattr(client, "load_collection"):
            client.load_collection(collection_name=self.milvus_collection_name)
        self.dense_ready = True

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

    def _save_bm25(self) -> None:
        with (self.index_dir / "bm25.pkl").open("wb") as stream:
            pickle.dump({"tokenized_corpus": self.tokenized_corpus}, stream)

    def _save_chunks(self) -> None:
        with (self.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as stream:
            for chunk in self.chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

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
            "reranker_model": self.reranker.model if self.reranker else None,
            "rerank_candidate_top_k": self.rerank_candidate_top_k,
            "index_revision": self.index_revision,
        }

    def _calculate_revision(self) -> str:
        digest = hashlib.sha256()
        for chunk in sorted(self.chunks, key=lambda item: str(item["chunk_id"])):
            digest.update(str(chunk["chunk_id"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk["text"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    chunk.get("metadata", {}),
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
