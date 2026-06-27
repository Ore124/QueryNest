# Redis Session History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Redis for chat session history, expose Redis/index runtime state, and make the UI clear that asking a question performs retrieval rather than rebuilding the index.

**Architecture:** Keep FAISS and BM25 persisted in `.rag_index`; Redis stores only bounded, expiring chat-message lists. The backend selects Redis when `REDIS_URL` is configured and exposes connectivity plus index origin in `/api/health`. Docker Compose runs a localhost-only Redis instance with persistent storage.

**Tech Stack:** Redis 8 Docker image, redis-py, FastAPI, React, pytest.

---

### Task 1: Redis runtime

**Files:**
- Create: `compose.yaml`
- Modify: `.env`
- Modify: `.env.example`
- Modify: `backend/pyproject.toml`

- [ ] Add a localhost-only Redis service with a persistent volume and health check.
- [ ] Add `REDIS_URL`, key prefix, message limit, and session TTL configuration.
- [ ] Install the Python Redis client and start the container.
- [ ] Verify with `docker compose exec redis redis-cli ping`, expecting `PONG`.

### Task 2: Redis-backed history

**Files:**
- Modify: `backend/app/history.py`
- Modify: `backend/app/settings.py`
- Modify: `backend/app/graph.py`
- Test: `backend/tests/test_history.py`

- [ ] Add a history-store protocol shared by SQLite and Redis implementations.
- [ ] Store each message as JSON in a Redis list, trim the list to the configured maximum, and refresh its TTL.
- [ ] Test ordered loading, trimming, expiry calls, and key names without requiring a real Redis server.

### Task 3: Runtime visibility and index diagnosis

**Files:**
- Modify: `backend/app/index.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/schemas.py`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/App.tsx`
- Test: `backend/tests/test_api.py`

- [ ] Track whether the current index was loaded from disk or rebuilt in this process.
- [ ] Add `index_operation=search_only` to chat retrieval debug.
- [ ] Expose history backend, Redis connectivity, and index origin from `/api/health`.
- [ ] Replace the misleading empty-state instruction with an index-aware message and label normal chat as retrieval rather than rebuild.
- [ ] Verify two consecutive `/api/chat` calls do not change the index build count.

### Task 4: Start and smoke test

**Files:**
- Modify: `README.md`

- [ ] Run backend tests and frontend build.
- [ ] Restart the backend, retain the existing frontend service, and verify Redis health.
- [ ] Send two questions in one session and verify Redis contains the message history while the index build count remains unchanged.
- [ ] Document startup and Redis inspection commands; do not run evaluation.
