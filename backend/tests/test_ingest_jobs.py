from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app.documents import RawDocument
from app.kb_metadata import ActiveIngestJobExists, IngestJobState


class ConcurrentFakeMetadataStore:
    backend = "postgresql"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_by_doc_id: dict[str, IngestJobState] = {}
        self.documents: dict[str, str] = {}
        self.failures: list[str] = []
        self.snapshots = 0

    def get_active_ingest_job(self, doc_id: str) -> IngestJobState | None:
        with self.lock:
            return self.active_by_doc_id.get(doc_id)

    def upsert_document_status(self, *, doc_id: str, status: str, **_: object) -> None:
        with self.lock:
            self.documents[doc_id] = status

    def create_ingest_job(self, *, kb_id: str, doc_id: str, worker_id: str | None = None) -> str:
        with self.lock:
            if doc_id in self.active_by_doc_id:
                raise ActiveIngestJobExists(f"active ingest job already exists for doc_id={doc_id}")
            job_id = "job-1"
            self.active_by_doc_id[doc_id] = IngestJobState(
                job_id=job_id,
                doc_id=doc_id,
                kb_id=kb_id,
                status="queued",
                worker_id=worker_id,
                retry_count=0,
                error_message=None,
            )
            return job_id

    def update_ingest_job(self, job_id: str, *, status: str, **_: object) -> None:
        with self.lock:
            for doc_id, state in list(self.active_by_doc_id.items()):
                if state.job_id != job_id:
                    continue
                if status == "completed":
                    del self.active_by_doc_id[doc_id]
                else:
                    self.active_by_doc_id[doc_id] = IngestJobState(
                        job_id=state.job_id,
                        doc_id=state.doc_id,
                        kb_id=state.kb_id,
                        status=status,
                        worker_id=state.worker_id,
                        retry_count=state.retry_count,
                        error_message=state.error_message,
                    )
                return

    def fail_ingest_job(self, job_id: str, *, error_message: str) -> None:
        with self.lock:
            self.failures.append(error_message)
            for doc_id, state in list(self.active_by_doc_id.items()):
                if state.job_id == job_id:
                    del self.active_by_doc_id[doc_id]

    def write_index_snapshot(self, **_: object) -> None:
        with self.lock:
            self.snapshots += 1


def test_concurrent_ingest_path_requests_only_one_enters_parse_and_index(tmp_path, monkeypatch, fake_milvus_client):
    from app import main
    from app.graph import RagService
    from app.history import ChatHistoryStore
    from app.index import HybridIndex
    from app.providers import HashEmbeddings

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n\nsame ingest source", encoding="utf-8")

    store = ConcurrentFakeMetadataStore()
    index = HybridIndex(tmp_path / "index", HashEmbeddings(dimensions=32), milvus_client=fake_milvus_client)
    history = ChatHistoryStore(tmp_path / "sessions.sqlite3")
    parse_calls = 0
    parse_lock = threading.Lock()

    def slow_load_documents(path: Path, parsers: object, include_images: bool = True) -> list[RawDocument]:
        del parsers, include_images
        nonlocal parse_calls
        with parse_lock:
            parse_calls += 1
        time.sleep(0.25)
        return [
            RawDocument(
                text="same ingest source",
                source_path=str(path / "guide.md"),
                source_name="guide.md",
                file_type="md",
                scenario="docs",
                parser="markdown",
            )
        ]

    monkeypatch.setattr(main, "metadata_store", store)
    monkeypatch.setattr(main, "hybrid_index", index)
    monkeypatch.setattr(main, "history_store", history)
    monkeypatch.setattr(main, "rag_service", RagService(main.settings, index, history))
    monkeypatch.setattr(main, "load_documents", slow_load_documents)

    def post_ingest() -> dict[str, object]:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/ingest/path",
                json={"path": str(docs), "rebuild": True, "include_images": False},
            )
            assert response.status_code == 200
            return response.json()

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda _: post_ingest(), range(4)))

    assert parse_calls == 1
    assert index.build_count == 1
    assert store.snapshots == 1
    assert sum(response["ingest_status"] == "completed" for response in responses) == 1
    active_statuses = {response["ingest_status"] for response in responses if response["ingest_status"] != "completed"}
    assert active_statuses <= {"queued", "parsing", "embedding"}
