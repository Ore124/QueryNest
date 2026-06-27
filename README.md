# RAG 知识库问答助手

本项目是一个本地可运行的企业知识库 RAG 问答助手，使用 FastAPI、LangChain、LangGraph、Milvus、BM25、RRF、Qwen3-VL-Rerank、React 和 Ragas。

项目交接和当前状态见：[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## 快速启动

```powershell
cd "D:\Codex Projects\RAG_project"
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .\backend
cd frontend
npm install
npm run build
```

后端开发服务：

```powershell
cd "D:\Codex Projects\RAG_project"
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

前端开发服务：

```powershell
cd "D:\Codex Projects\RAG_project\frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```

## Milvus 向量库、Redis 会话历史与查询缓存

Milvus 保存 dense 向量，BM25 语料和 chunk 元数据保存在本地 `.rag_index`；Redis 保存多轮对话历史，并缓存查询 embedding 与检索结果。

启动 Milvus 和 Redis：

```powershell
cd "D:\Codex Projects\RAG_project"
docker compose up -d milvus redis
docker compose exec redis redis-cli ping
curl http://127.0.0.1:9091/healthz
```

Redis 预期返回 `PONG`，Milvus healthz 预期返回 `OK`。默认配置：

```text
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION_NAME=rag_chunks
INDEX_DIR=.rag_index
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_KEY_PREFIX=rag:chat
REDIS_SESSION_TTL_SECONDS=604800
REDIS_HISTORY_MAX_MESSAGES=100
REDIS_CACHE_KEY_PREFIX=rag:cache
REDIS_CACHE_TTL_SECONDS=86400
```

Redis 现在承担两类职责：

- `rag:chat:*`：保存多轮会话历史。
- `rag:cache:*`：缓存完全相同 query 的 embedding，以及相同 query、场景、top_k、rerank 模型和索引版本下的检索结果。

检索缓存 key 包含索引内容 hash；调用 `/api/ingest/*` 重建索引后，新索引会使用新的 cache key，旧缓存不会被命中。

查看当前会话键：

```powershell
docker compose exec redis redis-cli KEYS "rag:chat:*"
```

停止 Redis：

```powershell
docker compose stop redis
```

Milvus 和 Redis 端口只绑定到本机，数据保存在 Docker volume 中。

## 索引何时重建

- 启动后端：从 Milvus collection 和 `.rag_index` 加载已有索引。
- 普通提问：只执行查询向量、Milvus、BM25、RRF 和 rerank，不重建索引。
- 点击前端“重建索引”或调用 `/api/ingest/*`：才会重新解析文档并构建索引。

`/api/health` 中：

- `index_origin=milvus` 表示从 Milvus collection 和本地 BM25/chunks 文件加载。
- `index_build_count=0` 表示当前进程没有重建过索引。
- `history_backend=redis` 和 `redis_connected=true` 表示 Redis 会话存储正常。
- `cache_backend=redis` 和 `cache_connected=true` 表示 Redis 查询缓存正常。

页面初始化还会调用 `POST /api/warmup`，预热远程 `embedding-3` 客户端。该操作只生成一个查询向量，不解析文档、不写入 Milvus，也不重建索引。

## 问答延迟说明

- 建索引与问答状态已分离：提问时只有发送按钮转圈，“重建索引”按钮不会再显示重建动画。
- 多轮问题使用最近一轮用户问题进行本地上下文拼接，不再额外调用一次 LLM 改写问题。
- 在线 RAG 回答关闭 GLM 深度思考，避免引用摘取类回答产生大量 reasoning tokens。
- `retrieval_debug.timings_ms` 分别记录 `dense`、`bm25`、`fusion` 和 `rerank` 耗时。
- 远程 embedding 偶尔仍会出现数秒级网络波动，但这是查询阶段，不是索引重建。

## 常用接口

- `GET /api/health`
- `GET /api/scenarios`
- `GET /api/models`
- `POST /api/warmup`
- `POST /api/ingest/path`
- `POST /api/ingest/files`
- `POST /api/chat`

默认从 `.env` 中读取智谱和 DashScope API 配置。`.env` 不应提交到版本控制。重排配置示例：

```text
DASHSCOPE_API_KEY=replace-with-your-key
RERANK_API_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_MODEL=qwen3-vl-rerank
RERANK_CANDIDATE_TOP_K=20
```

检索顺序为 Milvus + BM25 → RRF 融合 → `qwen3-vl-rerank`。重排接口失败时自动退回 RRF 顺序。

## LangSmith 追踪

在本地 `.env` 中配置：

```text
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=replace-with-your-key
LANGSMITH_PROJECT=rag
```

启动后，LangGraph 的 `rewrite`、`retrieve`、`generate` 等节点会自动上报到 `rag` 项目。

## 文档解析

- PDF、DOCX、PPTX、XLSX：MinerU `pipeline` 后端。
- 独立图片及 MinerU 提取的文档内图片：PaddleOCR。
- 不使用视觉大模型。
- 表格同时保留 Markdown、JSON 和原始 HTML；检索块按表格行切分并重复表头。

MinerU 使用独立环境，避免与 PaddleOCR 的 OpenCV 依赖冲突：

```powershell
cd "D:\Codex Projects\RAG_project"
python -m venv .mineru-venv
.\.mineru-venv\Scripts\python -m pip install -r .\mineru-requirements.txt
```

## 评估

当前版本使用带标准 `expected_chunk_id` 的 10 题评测集：

```text
backend/evaluation_data/ragas_questions_10_chunks.csv
```

检索指标严格按 chunk ID 计算。命中同一文档但未命中标准 chunk，不计入 Hit@K、Precision@K、Recall@K 或 MRR。评测集可用分号分隔的 `expected_chunk_ids` 标注多个相关 chunk。

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m app.evaluation `
  .\evaluation_data\ragas_questions_10_chunks.csv `
  --limit 10 `
  --output ..\logs\ragas_questions_10_chunks.csv `
  --run-retrieval `
  --retrieval-output ..\logs\retrieval_report_10_chunks.json `
  --hit-k 5 `
  --retrieval-top-k 20
```

生成答案并使用 `glm-4.7` 运行 RAGAS：

```powershell
..\.venv\Scripts\python -m app.evaluation `
  .\evaluation_data\ragas_questions_10_chunks.csv `
  --limit 10 `
  --generate-answers `
  --answers-output ..\logs\ragas_records_10_chunks.jsonl `
  --retrieval-top-k 8 `
  --run-ragas `
  --ragas-model glm-4.7 `
  --ragas-output ..\logs\ragas_summary_10_chunks.json
```

RAGAS 汇总包含各指标的 `valid_samples` 和 `metric_coverage`。当模型限流、超时或结构化输出解析失败时，不用无效样本计算平均值，并将 `complete` 标记为 `false`。

当前 RAGAS 配置暂不运行 Faithfulness，只保留 Answer Relevancy、Context Precision 和 Context Recall。

可断点续跑的 10 题 RAGAS 命令：

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m app.ragas_checkpoint `
  ..\logs\ragas_records_10_chunks_rerank.jsonl `
  --output ..\logs\ragas_summary_10_chunks_rerank_glm47_checkpoint.json `
  --model glm-4.7 `
  --context-top-k 5 `
  --workers 1 `
  --timeout 300 `
  --attempts 2
```

该命令复用已有答案和上下文，成功的“题目 × 指标”会写入检查点，下次只重试失败项。

最新 10 题 chunk 级 rerank 检索结果见 `logs/retrieval_report_10_chunks_rerank.json`：

- Hit@5 / Recall@5：`0.70`
- Precision@5：`0.14`
- MRR：`0.631`
- 平均延迟：`1941.93 ms`
- P99：`2894.52 ms`

对应的 RAGAS 结果（10/10 有效）：

- Answer Relevancy：`0.9009`
- Context Recall：`1.0000`
- Context Precision：`0.9496`
