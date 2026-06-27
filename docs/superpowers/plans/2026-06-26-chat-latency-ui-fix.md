# Chat Latency and UI State Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the rebuild-index button from showing a false spinner during chat and remove the redundant LLM query-rewrite call that adds 11–18 seconds to multi-turn questions.

**Architecture:** Split frontend ingestion and chat state. Keep the LangGraph rewrite node, but make it deterministic by combining the latest prior user question with the current follow-up instead of calling the chat model. Add dense, BM25, fusion, and rerank timings to retrieval debug for ongoing diagnosis.

**Tech Stack:** React, TypeScript, LangGraph, Python, pytest, LangSmith.

---

### Task 1: Separate frontend operation state

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] Replace the shared `busy` flag with `ingestBusy` and `chatBusy`.
- [ ] Show the spinning icon on “重建索引” only while ingestion is running.
- [ ] Disable chat submission only while chat is running, while preventing ingestion and chat from overlapping.
- [ ] Build the frontend and verify the rebuild button remains static during a chat request.

### Task 2: Remove redundant rewrite model call

**Files:**
- Modify: `backend/app/graph.py`
- Test: `backend/tests/test_graph.py`

- [ ] Test that a first-turn question remains unchanged.
- [ ] Test that a follow-up query combines the latest prior user question with the current question without calling an LLM.
- [ ] Keep answer generation unchanged so response grounding and citations remain intact.

### Task 3: Add retrieval timing

**Files:**
- Modify: `backend/app/index.py`
- Test: `backend/tests/test_api.py`

- [ ] Measure dense retrieval, BM25 retrieval, RRF fusion, and rerank independently with `perf_counter`.
- [ ] Return timings in `retrieval_debug.timings_ms`.
- [ ] Verify normal chat still reports `index_operation=search_only` and no index build.

### Task 4: Restart and verify

**Files:**
- Modify: `README.md`

- [ ] Run backend tests and frontend build.
- [ ] Restart the backend and frontend.
- [ ] Send a first-turn and follow-up question in one Redis session.
- [ ] Confirm LangSmith shows near-zero rewrite time and report total/retrieve/generate timings.
