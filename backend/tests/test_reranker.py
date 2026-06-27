import httpx

from app.index import HybridIndex, SearchHit
from app.providers import HashEmbeddings
from app.reranker import DashScopeReranker, RerankResult


def test_dashscope_reranker_parses_ranked_results(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.4},
                    ]
                }
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    reranker = DashScopeReranker(
        api_key="test-key",
        api_url="https://example.com/rerank",
        model="qwen3-vl-rerank",
        timeout_seconds=30,
        instruct="按相关性排序",
    )

    results = reranker.rerank("问题", ["片段一", "片段二"], top_n=2)

    assert results == [RerankResult(index=1, score=0.9), RerankResult(index=0, score=0.4)]
    assert captured["json"]["model"] == "qwen3-vl-rerank"
    assert captured["json"]["parameters"]["top_n"] == 2


class FakeReranker:
    model = "fake-reranker"

    def rerank(self, query, documents, top_n):
        return [RerankResult(index=1, score=0.95), RerankResult(index=0, score=0.2)]


class FailingReranker:
    model = "failing-reranker"

    def rerank(self, query, documents, top_n):
        raise RuntimeError("temporary failure")


def make_hits():
    return [
        SearchHit(chunk_id="chunk-a", text="A", metadata={"chunk_id": "chunk-a"}),
        SearchHit(chunk_id="chunk-b", text="B", metadata={"chunk_id": "chunk-b"}),
    ]


def test_hybrid_index_applies_rerank_order(tmp_path):
    index = HybridIndex(
        tmp_path,
        HashEmbeddings(dimensions=8),
        reranker=FakeReranker(),
    )

    hits, error = index._rerank("问题", make_hits(), final_top_k=2)

    assert error is None
    assert [hit.chunk_id for hit in hits] == ["chunk-b", "chunk-a"]
    assert hits[0].rerank_rank == 1
    assert hits[0].rerank_score == 0.95


def test_hybrid_index_falls_back_to_rrf_order_on_rerank_failure(tmp_path):
    index = HybridIndex(
        tmp_path,
        HashEmbeddings(dimensions=8),
        reranker=FailingReranker(),
    )

    hits, error = index._rerank("问题", make_hits(), final_top_k=2)

    assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-b"]
    assert "temporary failure" in error
