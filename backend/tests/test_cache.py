from __future__ import annotations

from app.cache import RedisJsonCache
from app.documents import DocumentChunk
from app.index import HybridIndex
from app.providers import CachedEmbeddings, HashEmbeddings


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int | None] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.values[key] = value
        self.expirations[key] = ex
        return True

    def ping(self) -> bool:
        return True


class CountingEmbeddings(HashEmbeddings):
    def __init__(self, dimensions: int = 64) -> None:
        super().__init__(dimensions=dimensions)
        self.query_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


def test_redis_json_cache_uses_stable_payload_keys():
    client = FakeRedis()
    cache = RedisJsonCache(
        "redis://unused",
        key_prefix="rag:cache",
        default_ttl_seconds=60,
        client=client,
    )

    cache.set_json("retrieval", {"b": 2, "a": 1}, {"ok": True})

    assert cache.get_json("retrieval", {"a": 1, "b": 2}) == {"ok": True}
    key = next(iter(client.values))
    assert key.startswith("rag:cache:retrieval:")
    assert client.expirations[key] == 60
    assert cache.ping() is True


def test_cached_embeddings_reuses_identical_query_vector():
    client = FakeRedis()
    cache = RedisJsonCache(
        "redis://unused",
        key_prefix="rag:cache",
        default_ttl_seconds=60,
        client=client,
    )
    inner = CountingEmbeddings(dimensions=64)
    embeddings = CachedEmbeddings(
        inner,
        cache=cache,
        model="embedding-3",
        dimensions=64,
        ttl_seconds=60,
    )

    first = embeddings.embed_query("same question")
    assert embeddings.last_query_cache_hit is False
    second = embeddings.embed_query("same question")

    assert first == second
    assert inner.query_calls == 1
    assert embeddings.last_query_cache_hit is True


def test_hybrid_index_retrieval_cache_skips_second_search(tmp_path, fake_milvus_client):
    client = FakeRedis()
    cache = RedisJsonCache(
        "redis://unused",
        key_prefix="rag:cache",
        default_ttl_seconds=60,
        client=client,
    )
    inner = CountingEmbeddings(dimensions=64)
    embeddings = CachedEmbeddings(
        inner,
        cache=cache,
        model="embedding-3",
        dimensions=64,
        ttl_seconds=60,
    )
    index = HybridIndex(
        tmp_path / "index",
        embeddings,
        cache=cache,
        cache_ttl_seconds=60,
        milvus_client=fake_milvus_client,
    )
    index.build(
        [
            DocumentChunk(
                chunk_id="chunk-1",
                text="接口 500 需要查看 traceId 和服务日志",
                metadata={
                    "chunk_id": "chunk-1",
                    "source_path": "api.md",
                    "source_name": "api.md",
                    "file_type": "md",
                    "scenario": "研发",
                },
            ),
            DocumentChunk(
                chunk_id="chunk-2",
                text="请假流程需要先提交审批单",
                metadata={
                    "chunk_id": "chunk-2",
                    "source_path": "hr.md",
                    "source_name": "hr.md",
                    "file_type": "md",
                    "scenario": "人事",
                },
            ),
        ]
    )
    inner.query_calls = 0

    first_sources, first_debug = index.search(
        "接口 500 怎么排查",
        scenario=None,
        dense_top_k=2,
        bm25_top_k=2,
        final_top_k=1,
        rrf_k=60,
    )
    second_sources, second_debug = index.search(
        "接口 500 怎么排查",
        scenario=None,
        dense_top_k=2,
        bm25_top_k=2,
        final_top_k=1,
        rrf_k=60,
    )

    assert [source.chunk_id for source in second_sources] == [
        source.chunk_id for source in first_sources
    ]
    assert first_debug["cache"]["retrieval"] == "miss"
    assert second_debug["cache"]["retrieval"] == "hit"
    assert second_debug["timings_ms"]["dense"] == 0.0
    assert inner.query_calls == 1
