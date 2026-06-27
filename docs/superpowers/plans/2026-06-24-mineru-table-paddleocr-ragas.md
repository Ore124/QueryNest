# MinerU、表格检索、PaddleOCR 与 RAGAS 评测实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 MinerU 解析复杂文档、使用 PaddleOCR 解析独立图片、保留表格 Markdown/JSON 结构，并对 30 个问题完成检索与 RAGAS 指标评测。

**Architecture:** 主应用通过 `MinerUParser` 调用隔离环境中的 MinerU 3.4 `pipeline` CLI，并读取 `content_list.json`，避免引入 MinerU 与 PaddleOCR 的 OpenCV 依赖冲突。独立图片只调用 PaddleOCR；表格块保存原始 HTML、标准化 JSON 和 Markdown 检索文本，切块时按行拆分并重复表头。

**Tech Stack:** Python 3.12、MinerU 3.4、PaddleOCR 3.7、PaddlePaddle 3.3、FastAPI、LangChain、FAISS、BM25、RAGAS 0.4。

---

## 文件结构

- 创建 `backend/app/document_parsers.py`：MinerU CLI、PaddleOCR、表格 HTML/JSON/Markdown 转换。
- 修改 `backend/app/documents.py`：文件路由、结构化元数据和表格安全切块。
- 修改 `backend/app/main.py`：构造解析器，不再创建视觉模型。
- 修改 `backend/app/settings.py`：MinerU 命令、输出缓存和 PaddleOCR 配置。
- 修改 `backend/app/schemas.py`：向引用结果暴露内容类型和表格结构。
- 修改 `backend/app/index.py`：映射新增元数据字段。
- 修改 `backend/app/evaluation.py`：稳定使用 RAGAS 0.4 公共指标入口并保留 30 问输出。
- 修改 `backend/pyproject.toml`：PaddleOCR 依赖和文档工具可选依赖说明。
- 修改 `.env.example`、`.gitignore`、`README.md`、`PROJECT_HANDOFF.md`：安装和运行说明。
- 修改 `frontend/src/types.ts`、`frontend/src/App.tsx`：显示表格类型，不再暴露视觉模型。
- 修改 `backend/tests/test_ingestion.py`，创建 `backend/tests/test_document_parsers.py`：覆盖 MinerU、PaddleOCR 和表格结构。

### Task 1: 用测试锁定结构化解析行为

- [ ] **Step 1: 创建 MinerU 内容列表测试**

测试夹具包含文本和表格块：

```python
content = [
    {"type": "text", "text": "审批说明", "page_idx": 0},
    {
        "type": "table",
        "table_caption": ["审批矩阵"],
        "table_body": "<table><tr><th>金额</th><th>审批人</th></tr><tr><td>1000</td><td>经理</td></tr></table>",
        "page_idx": 1,
    },
]
```

断言表格文档包含 `content_type == "table"`、Markdown 表头、JSON 行列和原始 HTML。

- [ ] **Step 2: 创建 PaddleOCR 结果归一化测试**

```python
result = {"rec_texts": ["服务器", "部署"], "rec_scores": [0.99, 0.95]}
assert paddle_result_to_text(result) == "服务器\n部署"
```

- [ ] **Step 3: 创建表格切块测试**

构造超过 `chunk_size` 的 Markdown 表格，断言每个表格块都重复表头，并保留相同 `table_id` 和 `table_json`。

- [ ] **Step 4: 运行失败测试**

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m pytest tests\test_document_parsers.py tests\test_ingestion.py -q
```

预期：新模块或新字段尚不存在，测试失败。

### Task 2: 实现 MinerU、PaddleOCR 和表格转换

- [ ] **Step 1: 实现 MinerU CLI 调用**

命令固定使用非 VLM 后端：

```python
command = [
    str(self.executable),
    "-p", str(path),
    "-o", str(output_dir),
    "-b", "pipeline",
    "-m", "auto",
    "-l", self.language,
]
```

解析完成后递归查找与输入文件同 stem 的 `*_content_list.json`。

- [ ] **Step 2: 实现表格归一化**

输出结构：

```python
{
    "title": "审批矩阵",
    "headers": ["金额", "审批人"],
    "rows": [["1000", "经理"]],
    "html": "<table>...</table>",
}
```

检索文本使用 Markdown：

```markdown
### 审批矩阵

| 金额 | 审批人 |
| --- | --- |
| 1000 | 经理 |
```

- [ ] **Step 3: 实现 PaddleOCR 延迟加载**

```python
from paddleocr import PaddleOCR

self._engine = PaddleOCR(
    lang=self.language,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
)
```

调用 `predict(str(image_path))`，只拼接 OCR 文本和置信度，不调用任何视觉大模型。

- [ ] **Step 4: 运行解析器测试**

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m pytest tests\test_document_parsers.py -q
```

预期：全部通过。

### Task 3: 接入入库和检索链路

- [ ] **Step 1: 扩展支持格式**

MinerU 路由：`.pdf`、`.docx`、`.pptx`、`.xlsx`；直接文本路由：`.md`、`.txt`、`.csv`、`.tsv`；PaddleOCR 路由：`.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`。

- [ ] **Step 2: 扩展元数据**

```python
metadata = {
    "content_type": document.content_type,
    "parser": document.parser,
    "table_id": document.table_id,
    "table_markdown": document.table_markdown,
    "table_json": document.table_json,
    "table_html": document.table_html,
}
```

- [ ] **Step 3: 表格按行切块**

普通文本继续使用 `RecursiveCharacterTextSplitter`；表格使用专门函数按行组成不超过 `chunk_size` 的块，每块重复标题、表头和分隔行。

- [ ] **Step 4: 删除视觉模型调用**

`main.py` 不再读取 `DEFAULT_VISION_MODEL`，不再构造 `ImageCaptioner`，模型接口只返回聊天和嵌入模型。

- [ ] **Step 5: 验证 API 和索引**

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m pytest -q
```

预期：全部测试通过，表格字段可从 `Source` 返回。

### Task 4: 安装隔离的文档工具运行时

- [ ] **Step 1: 安装 PaddleOCR 到主环境**

```powershell
cd "D:\Codex Projects\RAG_project"
.\.venv\Scripts\python -m pip install "paddleocr==3.7.0" "paddlepaddle==3.3.1"
```

- [ ] **Step 2: 安装 MinerU 到隔离环境**

```powershell
cd "D:\Codex Projects\RAG_project"
python -m venv .mineru-venv
.\.mineru-venv\Scripts\python -m pip install --upgrade pip
.\.mineru-venv\Scripts\python -m pip install -r .\mineru-requirements.txt
```

- [ ] **Step 3: 验证命令**

```powershell
.\.mineru-venv\Scripts\mineru.exe --help
.\.venv\Scripts\python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
```

预期：两个命令退出码均为 0。

### Task 5: 重建索引并执行 30 问评测

- [ ] **Step 1: 使用新解析器重建知识库索引**

调用 `/api/ingest/path` 或直接调用入库函数，确认 PDF 由 MinerU 处理、图片由 PaddleOCR 处理，并记录表格块数量。

- [ ] **Step 2: 固定抽样 30 问**

```powershell
cd "D:\Codex Projects\RAG_project\backend"
..\.venv\Scripts\python -m app.evaluation `
  "D:\Codex Projects\企业知识库\00_知识库目录\RAG测试问题集.md" `
  --limit 30 `
  --sample-seed 42 `
  --output ..\logs\ragas_questions_30.csv `
  --run-retrieval `
  --retrieval-output ..\logs\retrieval_report_30.json `
  --generate-answers `
  --answers-output ..\logs\ragas_records_30.jsonl `
  --run-ragas `
  --ragas-output ..\logs\ragas_summary_30.json
```

- [ ] **Step 3: 验证评测输出**

确认：

```text
question_count = 30
hit@5、MRR、mean_latency_ms、P99_ms 存在
faithfulness、answer_relevancy、context_precision、context_recall 均可序列化
```

- [ ] **Step 4: 更新文档**

README 和交接文档记录安装方式、解析边界、30 问抽样参数和实际指标，不写入 API 密钥。
