# Qwen3-VL-Rerank 与 Chunk 检索指标实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 FAISS + BM25 + RRF 候选结果后接入 `qwen3-vl-rerank`，并用 10 题标准 chunk 集重新评估 Hit、MRR、Precision 和 Recall。

**Architecture:** RRF 负责召回最多 20 个候选，DashScope 原生 rerank API 对候选文本二次排序，最终只返回用户请求的 Top-K。Rerank 调用失败时保留 RRF 顺序并在 debug 中记录失败原因，避免在线问答不可用。

**Tech Stack:** Python 3.12、httpx、DashScope `qwen3-vl-rerank`、FAISS、BM25、RRF、pytest。

---

### Task 1: 锁定 rerank 行为

**Files:**
- Create: `backend/app/reranker.py`
- Create: `backend/tests/test_reranker.py`
- Modify: `backend/tests/test_evaluation.py`

- [ ] **Step 1: 编写 API 响应映射测试**

```python
response = {"output": {"results": [
    {"index": 1, "relevance_score": 0.91},
    {"index": 0, "relevance_score": 0.42},
]}}
```

断言原候选顺序被改为索引 `[1, 0]`，并保存 rerank rank/score。

- [ ] **Step 2: 编写失败回退测试**

模拟 API 异常，断言返回原 RRF 顺序且 debug 标记 `rerank_applied = false`。

- [ ] **Step 3: 编写多标准 chunk 指标测试**

```python
expected = {"chunk-a", "chunk-c"}
retrieved = ["chunk-a", "chunk-b", "chunk-c", "chunk-d", "chunk-e"]
```

断言 `Precision@5 = 0.4`、`Recall@5 = 1.0`。

### Task 2: 实现 DashScope reranker

**Files:**
- Create: `backend/app/reranker.py`
- Modify: `backend/app/settings.py`
- Modify: `.env.example`

- [ ] **Step 1: 实现原生 HTTP 请求**

```python
payload = {
    "model": "qwen3-vl-rerank",
    "input": {
        "query": {"text": query},
        "documents": [{"text": hit.text} for hit in hits],
    },
    "parameters": {
        "top_n": top_n,
        "return_documents": False,
        "instruct": instruct,
    },
}
```

接口：

```text
POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
```

- [ ] **Step 2: 增加配置**

```text
DASHSCOPE_API_KEY
RERANK_MODEL=qwen3-vl-rerank
RERANK_CANDIDATE_TOP_K=20
RERANK_TIMEOUT_SECONDS=120
```

### Task 3: 接入 HybridIndex

**Files:**
- Modify: `backend/app/index.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/evaluation.py`

- [ ] **Step 1: RRF 后 rerank**

先取 `max(final_top_k, rerank_candidate_top_k)` 个融合候选，再由 reranker 返回最终 Top-K。

- [ ] **Step 2: 暴露排序信息**

`Source` 新增：

```python
rerank_rank: int | None
rerank_score: float | None
```

debug 新增 `rerank_model`、`rerank_applied`、`reranked_hits`。

### Task 4: 增加 Precision / Recall 并评估

**Files:**
- Modify: `backend/app/evaluation.py`
- Modify: `backend/evaluation_data/ragas_questions_10_chunks.csv`

- [ ] **Step 1: 支持多个标准 chunk**

兼容 `expected_chunk_id` 和用分号分隔的 `expected_chunk_ids`。

- [ ] **Step 2: 计算指标**

```python
precision_at_k = relevant_retrieved / k
recall_at_k = relevant_retrieved / relevant_total
```

同时保留 Hit@K、MRR、P99。

- [ ] **Step 3: 运行 10 题**

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m app.evaluation `
  .\evaluation_data\ragas_questions_10_chunks.csv `
  --limit 10 `
  --run-retrieval `
  --retrieval-output ..\logs\retrieval_report_10_chunks_rerank.json `
  --hit-k 5 `
  --retrieval-top-k 20
```

本轮不运行 Faithfulness。

### Task 5: 验证与文档

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_HANDOFF.md`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: 运行测试**

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m pytest -q
```

- [ ] **Step 2: 构建前端**

```powershell
cd "D:\Codex Projects\RAG_project\frontend"
npm run build
```

- [ ] **Step 3: 记录 rerank 前后指标和 API 配置方式**

文档不记录真实 API Key。
