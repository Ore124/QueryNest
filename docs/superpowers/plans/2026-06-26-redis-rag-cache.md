# Redis RAG Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Redis to speed up repeated RAG queries without changing answer semantics.

**Architecture:** Add a small JSON cache abstraction backed by Redis. Wrap query embeddings with an exact-match cache and add a retrieval-result cache inside `HybridIndex.search()` keyed by query, retrieval parameters, reranker model, and index revision.

**Tech Stack:** FastAPI, LangChain embeddings, FAISS, Redis, pytest.

---

### Task 1: Cache abstraction

**Files:**
- Create: `backend/app/cache.py`
- Test: `backend/tests/test_cache.py`

- [ ] Add `RedisJsonCache` and `NullJsonCache` with stable JSON-hash keys, TTL, and fail-closed behavior.
- [ ] Verify repeated equivalent payloads produce the same key and cache misses do not raise.

### Task 2: Query embedding cache

**Files:**
- Modify: `backend/app/providers.py`
- Test: `backend/tests/test_cache.py`

- [ ] Add `CachedEmbeddings` that caches only `embed_query()` and passes `embed_documents()` through.
- [ ] Verify the second identical query does not call the wrapped embedding model.

### Task 3: Retrieval cache

**Files:**
- Modify: `backend/app/index.py`
- Test: `backend/tests/test_cache.py`

- [ ] Add an index revision hash after load/build.
- [ ] Cache final retrieval sources/debug for identical query + parameters + index revision.
- [ ] Verify a second identical `search()` returns from cache and skips dense/BM25/rerank work.

### Task 4: Wiring and docs

**Files:**
- Modify: `backend/app/settings.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/schemas.py`
- Modify: `.env.example`
- Modify: `README.md`

- [ ] Add Redis cache prefix/TTL settings.
- [ ] Wire the cache into embeddings and `HybridIndex`.
- [ ] Expose cache status in `/api/health`.
- [ ] Document what Redis now accelerates and what it still does not cache.

### Task 5: Verification

**Commands:**
- `.\.venv\Scripts\python -m pytest backend\tests -q`
- `curl http://127.0.0.1:8000/api/health`
- Send the same `/api/chat` request twice and check `retrieval_debug.cache.retrieval` changes from `miss` to `hit`.
