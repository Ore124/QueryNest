# RAG Project Handoff

Last updated: 2026-06-25

## Project Goal

Build a local enterprise knowledge-base RAG assistant with:

- FastAPI backend
- React/Vite frontend
- LangChain + LangGraph orchestration
- Milvus dense retrieval
- BM25 sparse retrieval
- RRF fusion
- Qwen3-VL-Rerank
- Ragas-oriented evaluation tooling
- Zhipu/Z.AI models by default

The project workspace is:

```text
D:\Codex Projects\RAG_project
```

The test knowledge base used during development was:

```text
D:\Codex Projects\企业知识库
```

## Current Implementation

Backend entrypoint:

```text
backend/app/main.py
```

Frontend entrypoint:

```text
frontend/src/App.tsx
```

Important backend modules:

- `backend/app/documents.py`: document discovery, parsing, and chunking.
- `backend/app/index.py`: Milvus + BM25 indexes, RRF fusion, and rerank integration.
- `backend/app/reranker.py`: DashScope `qwen3-vl-rerank` client.
- `backend/app/graph.py`: LangGraph RAG flow.
- `backend/app/providers.py`: Zhipu-compatible chat and embedding clients.
- `backend/app/evaluation.py`: question sampling, retrieval evaluation, optional Ragas answer evaluation.
- `backend/app/settings.py`: environment and default retrieval settings.

Implemented APIs:

- `GET /api/health`
- `GET /api/scenarios`
- `GET /api/models`
- `POST /api/ingest/path`
- `POST /api/ingest/files`
- `POST /api/chat`

## Model Configuration

Configured through local `.env`; do not commit real keys.

Current defaults:

```text
ZAI_API_BASE=https://api.z.ai/api/paas/v4/
DEFAULT_CHAT_MODEL=glm-5.2
DEFAULT_VISION_MODEL=glm-5v-turbo
DEFAULT_EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=2048
DASHSCOPE_API_KEY=...
RERANK_MODEL=qwen3-vl-rerank
RERANK_CANDIDATE_TOP_K=20
```

Notes:

- `embedding-3` was verified to return 2048-dimensional vectors.
- The embedding client uses `chunk_size=64` because the Zhipu embedding API rejects input batches larger than 64.
- `glm-5.2` is used for chat and Ragas LLM evaluation.
- `glm-5v-turbo` is used for image OCR/captioning because `glm-5.2` is a text model.
- `.env` contains a real key in the local workspace. Rotate it if this project is shared outside the local machine.

## Document Support

Supported file types:

```text
.md, .txt, .pdf, .png, .jpg, .jpeg
```

Parsing behavior:

- Markdown/TXT: read as UTF-8, fallback to UTF-8-SIG and GB18030.
- PDF: `PyMuPDF` page-level text extraction.
- Images: call `glm-5v-turbo` to generate OCR/content description, then index that generated text.

Default excludes:

```text
**/tmp/**
**/*测试问题集.md
**/RAG测试问题集.md
```

The test question set is intentionally excluded from knowledge-base ingestion.

## Chunk Strategy

Configured in `backend/app/settings.py`:

```python
chunk_size = 900
chunk_overlap = 160
```

Implemented in `backend/app/documents.py` with `RecursiveCharacterTextSplitter`.

Separator priority:

```python
["\n## ", "\n### ", "\n| ", "\n\n", "\n", "。", "；", "，", " ", ""]
```

Chunk metadata includes:

- `chunk_id`
- `source_path`
- `source_name`
- `file_type`
- `scenario`
- `section`
- `page`
- `chunk_index`

## Retrieval Strategy

Implemented retrieval pipeline:

1. Dense retrieval with Milvus and Zhipu `embedding-3`.
2. Sparse retrieval with BM25.
3. Chinese tokenization for BM25 with `jieba`; English/numbers are preserved.
4. RRF fusion:

```python
score += 1 / (rrf_k + rank)
```

5. DashScope `qwen3-vl-rerank` reorders the top 20 chunk candidates. API failures fall back to RRF order.

Default retrieval settings:

```python
dense_top_k = 20
bm25_top_k = 20
final_top_k = 8
rrf_k = 60
```

Frontend displays per-source:

- source document
- section/page
- Milvus rank
- BM25 rank
- RRF score
- rerank rank and score

## LangGraph Flow

Implemented in `backend/app/graph.py`.

Flow:

```text
rewrite question -> retrieve -> generate answer
```

Behavior:

- Rewrites multi-turn questions when history exists.
- Searches Milvus + BM25, fuses with RRF, and reranks chunk candidates.
- Generates answers with citations.
- Stores chat history in SQLite through `backend/app/history.py`.

## Current Index State

A real ingestion run was completed against:

```text
D:\Codex Projects\企业知识库
```

Observed result:

```text
indexed_chunks = 88
source_documents = 26
scenarios = ["制度与流程", "图片资产", "安全与合规", "知识库目录", "研发与技术"]
```

Index files are under:

```text
.rag_index
```

Session DB:

```text
.sessions/rag_chat.sqlite3
```

These are ignored by `.gitignore`.

## Evaluation

Evaluation code:

```text
backend/app/evaluation.py
```

Question source:

```text
D:\Codex Projects\企业知识库\00_知识库目录\RAG测试问题集.md
```

The 100-question retrieval evaluation uses:

```text
--limit 100
--sample-seed 42
--hit-k 5
--retrieval-top-k 20
```

Command:

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m app.evaluation `
  "D:\Codex Projects\企业知识库\00_知识库目录\RAG测试问题集.md" `
  --limit 100 `
  --sample-seed 42 `
  --output ..\logs\ragas_questions_100.csv `
  --run-retrieval `
  --retrieval-output ..\logs\retrieval_report_100.json `
  --hit-k 5 `
  --retrieval-top-k 20
```

Latest 100-question retrieval result:

```json
{
  "question_count": 100,
  "retrieval_top_k": 20,
  "hit_k": 5,
  "hit@5": 0.96,
  "MRR": 0.9397,
  "miss_count": 0,
  "mean_latency_ms": 538.09,
  "P99_ms": 1260.07
}
```

Metric definitions:

- `hit@5`: whether the expected source document appears in the top 5 retrieved sources.
- `MRR`: average `1 / first_relevant_rank`; zero if not found.
- `P99_ms`: 99% of retrieval requests have latency less than or equal to this value.

Important correction:

- The user clarified that the intended latency metric is `P99`, not `K99`.
- If a future request mentions unusual metric names such as `K99`, confirm before implementing.

Ragas status:

- Ragas evaluation entry exists and can generate answer records.
- The current Ragas metric list includes `answer_relevancy`, `context_precision`, and `context_recall`; Faithfulness is temporarily disabled.
- Ragas summary output is written with `--ragas-output`; it includes Chinese aliases for `上下文精确率` and `上下文召回率`.
- A 1-question Ragas smoke run completed partially:
  - `answer_relevancy`, `context_precision`, `context_recall` produced values.
  - `faithfulness` returned `nan` due to Ragas output parsing issues on the Chinese example.
- Full 100-question Ragas LLM evaluation has not been run because it is costlier and still needs prompt/parser tuning.

## Frontend

Frontend stack:

- React
- Vite
- TypeScript
- lucide-react
- react-markdown

UI behavior:

- Left panel: ingestion path, image parsing toggle, scenario/model/top-k controls.
- Center: multi-turn chat.
- Right panel: source citations and retrieval ranks.

Dev server:

```powershell
cd "D:\Codex Projects\RAG_project\frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```

Default frontend URL:

```text
http://127.0.0.1:5173
```

## Run Commands

Install backend:

```powershell
cd "D:\Codex Projects\RAG_project"
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .\backend
```

Run backend:

```powershell
cd "D:\Codex Projects\RAG_project"
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Run frontend:

```powershell
cd "D:\Codex Projects\RAG_project\frontend"
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Build frontend:

```powershell
cd "D:\Codex Projects\RAG_project\frontend"
npm run build
```

Run backend tests:

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m pytest
```

Latest test status:

```text
8 passed
```

Warnings observed:

- FastAPI/Starlette `TestClient` deprecation warning.
- Milvus service must be running before loading or rebuilding the dense index.

## Known Gaps / Next Work

Likely next tasks:

1. Improve chunk strategy with heading-aware section grouping and table-preserving chunks.
2. Decide whether to restore Faithfulness after the structured-output issue is solved.
3. Run full 100-question answer generation and Ragas evaluation after cost approval.
4. Add a frontend evaluation dashboard for retrieval metrics.
5. Add provider adapters beyond Zhipu if multi-vendor API support needs to be real, not just configured placeholders.
6. Consider incremental index update/delete; current ingestion is rebuild-first.
7. Consider moving secrets out of local `.env` before sharing.

## 2026-06-24 Structured Parsing and Chunk-Level Evaluation Update

- MinerU 3.4 `pipeline` parses PDF/DOCX/PPTX/XLSX in `.mineru-venv`.
- PaddleOCR 3.7 parses all images; no vision LLM is used.
- Tables retain Markdown, JSON, and original HTML. The rebuilt index contains 692 chunks:
  - 659 text chunks
  - 14 structured table chunks
  - 19 OCR image chunks
- The versioned evaluation set is `backend/evaluation_data/ragas_questions_10_chunks.csv`.
- Retrieval relevance is exact `expected_chunk_id`; document matches are diagnostic only.
- Latest exact-chunk retrieval result:
  - Hit@5: `0.60`
  - MRR: `0.405`
  - miss count within top 20: `1`
  - mean latency: `1200.07 ms`
  - P99: `3791.49 ms`
- RAGAS evaluator model is `glm-4.7`.
- Latest RAGAS run was affected by provider connection/structured-output failures:
  - answer relevancy: `0.7631` (10/10 valid)
  - context recall: `0.9000` (10/10 valid)
  - context precision: `0.7293` (5/10 valid)
  - faithfulness: unavailable (0/10 valid)
- Treat the incomplete RAGAS metrics as partial diagnostics, not a complete benchmark.

## 2026-06-25 Qwen3-VL-Rerank Update

- Added DashScope `qwen3-vl-rerank` after RRF fusion, using 20 candidate chunks.
- The configured key is valid on the Beijing endpoint; the Singapore endpoint returns `401 InvalidApiKey`.
- Rerank failures automatically fall back to the original RRF order.
- Retrieval evaluation now supports `expected_chunk_ids` and reports chunk-level Precision@K and Recall@K.
- Faithfulness is disabled for the current RAGAS configuration.
- Latest 10-question report: `logs/retrieval_report_10_chunks_rerank.json`.
  - Hit@5 / Recall@5: `0.70` (baseline `0.60`)
  - Precision@5: `0.14`
  - MRR: `0.631` (baseline `0.405`)
  - miss count within top 20: `1`
  - mean latency: `1941.93 ms` (baseline `1200.07 ms`)
  - P99: `2894.52 ms` (baseline `3791.49 ms`)
- Resumable RAGAS report: `logs/ragas_summary_10_chunks_rerank_glm47_checkpoint.json`.
  - evaluator: `glm-4.7`
  - context top-k: `5`
  - Answer Relevancy: `0.9009` (10/10 valid)
  - Context Recall: `1.0000` (10/10 valid)
  - Context Precision: `0.9496` (10/10 valid)
  - Faithfulness: not run
- `backend/app/ragas_checkpoint.py` reuses the answer JSONL and checkpoints each successful sample/metric. Use one worker to avoid provider 429 rate limits.
