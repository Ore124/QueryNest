from pathlib import Path

from fastapi.testclient import TestClient

from app import graph as graph_module
from app.history import ChatHistoryStore
from app.index import HybridIndex
from app.graph import RagService
from app.providers import HashEmbeddings


class FakeResponse:
    content = "根据引用，接口 500 应先查看错误提示和日志。[1]"


class FakeChatModel:
    def invoke(self, messages):
        return FakeResponse()


def test_api_ingest_and_chat_with_hybrid_debug(tmp_path, monkeypatch, fake_milvus_client):
    from app import main

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "03_研发与技术").mkdir()
    (docs / "03_研发与技术" / "api.md").write_text(
        "# API接口说明文档\n\n接口 500 时，责任团队应查看错误提示、traceId 和服务日志。",
        encoding="utf-8",
    )

    test_index = HybridIndex(tmp_path / "index", HashEmbeddings(dimensions=64), milvus_client=fake_milvus_client)
    test_history = ChatHistoryStore(tmp_path / "sessions.sqlite3")
    monkeypatch.setattr(main, "hybrid_index", test_index)
    monkeypatch.setattr(main, "history_store", test_history)
    monkeypatch.setattr(main, "rag_service", RagService(main.settings, test_index, test_history))
    monkeypatch.setattr(graph_module, "get_chat_model", lambda *args, **kwargs: FakeChatModel())

    client = TestClient(main.app)
    monkeypatch.setattr(main, "embedding_warmed", True)
    warmup_response = client.post("/api/warmup")
    assert warmup_response.status_code == 200
    assert warmup_response.json()["embedding_warmed"] is True

    ingest_response = client.post(
        "/api/ingest/path",
        json={"path": str(docs), "rebuild": True, "include_images": False},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["indexed_chunks"] >= 1

    chat_response = client.post("/api/chat", json={"message": "接口 500 怎么排查？", "top_k": 3})
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["answer"]
    assert payload["sources"][0]["bm25_rank"] is not None
    assert payload["retrieval_debug"]["fused_hits"]
    assert payload["retrieval_debug"]["index_operation"] == "search_only"
    assert set(payload["retrieval_debug"]["timings_ms"]) == {
        "dense",
        "bm25",
        "fusion",
        "rerank",
        "total",
    }
    build_count = payload["retrieval_debug"]["index_build_count"]

    second_response = client.post(
        "/api/chat",
        json={"message": "再说一次", "session_id": payload["session_id"], "top_k": 3},
    )

    assert second_response.status_code == 200
    assert second_response.json()["retrieval_debug"]["index_build_count"] == build_count
