import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  BookOpen,
  Bot,
  Database,
  FileSearch,
  FileText,
  FileUp,
  FolderOpen,
  Loader2,
  MessageSquare,
  RefreshCcw,
  Send,
  Settings2,
  Sparkles,
  XCircle,
} from "lucide-react";
import { chat, getChunks, getDocuments, getHealth, getModels, getScenarios, ingestFilesWithProgress, warmup } from "./api";
import type { ChunkItem, DocumentItem, Health, Message, ModelInfo, Source } from "./types";

const SUPPORTED_UPLOAD_ACCEPT = ".md,.txt,.csv,.tsv,.xlsx,.pdf,.png,.jpg,.jpeg";
const CHUNK_PAGE_SIZE = 50;

function uploadFilePath(file: File) {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
}

function uploadSelectionLabel(files: File[]) {
  if (files.length === 0) return "未选择文件";
  const firstPath = uploadFilePath(files[0]);
  const rootName = firstPath.includes("/") ? firstPath.split("/")[0] : firstPath;
  return files.length === 1 ? rootName : `${rootName} / ${files.length} 个文件`;
}

function formatElapsed(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return minutes > 0 ? `${minutes}:${String(remainingSeconds).padStart(2, "0")}` : `${remainingSeconds}s`;
}

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [scenario, setScenario] = useState("");
  const [model, setModel] = useState("");
  const [topK, setTopK] = useState(8);
  const [agentic, setAgentic] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const [panelMode, setPanelMode] = useState<"sources" | "chunks">("sources");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentTotal, setDocumentTotal] = useState(0);
  const [selectedDocument, setSelectedDocument] = useState<DocumentItem | null>(null);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [chunkOffset, setChunkOffset] = useState(0);
  const [chunkBusy, setChunkBusy] = useState(false);
  const [chunkError, setChunkError] = useState("");
  const [ingestBusy, setIngestBusy] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestPhase, setIngestPhase] = useState("");
  const [ingestElapsedSeconds, setIngestElapsedSeconds] = useState(0);
  const [chatBusy, setChatBusy] = useState(false);
  const [status, setStatus] = useState("等待连接后端");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const activeIngestRequestRef = useRef<{ cancel: () => void } | null>(null);

  const chatModels = useMemo(() => models.filter((item) => item.role === "chat"), [models]);
  const selectedUploadLabel = useMemo(() => uploadSelectionLabel(selectedFiles), [selectedFiles]);

  useEffect(() => {
    void refreshState();
  }, []);

  async function refreshState() {
    setStatus("正在连接后端并预热检索模型");
    try {
      const [healthData, scenarioData, modelData] = await Promise.all([
        getHealth(),
        getScenarios(),
        getModels(),
        warmup(),
      ]);
      setHealth(healthData);
      setScenarios(scenarioData.scenarios);
      setModels(modelData);
      setModel((current) => current || healthData.default_chat_model);
      setStatus(
        healthData.index_ready
          ? `知识库已加载：${healthData.indexed_chunks} 个片段`
          : "索引未构建",
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "后端连接失败");
    }
  }

  function handleUploadSelection(event: ChangeEvent<HTMLInputElement>) {
    setSelectedFiles(Array.from(event.target.files ?? []));
    setIngestProgress(0);
    setIngestPhase("");
    setIngestElapsedSeconds(0);
    event.target.value = "";
  }

  function handleCancelIngest() {
    activeIngestRequestRef.current?.cancel();
    activeIngestRequestRef.current = null;
    setIngestPhase("已取消");
    setStatus("已取消重建索引");
  }

  async function loadDocuments() {
    if (chunkBusy) return;
    setPanelMode("chunks");
    setSelectedDocument(null);
    setChunks([]);
    setChunkTotal(0);
    setChunkOffset(0);
    setChunkBusy(true);
    setChunkError("");
    try {
      const data = await getDocuments({ scenario: scenario || undefined });
      setDocuments(data.documents);
      setDocumentTotal(data.total);
      setStatus(`已加载 ${data.total} 个入库文件`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "加载入库文件失败";
      setChunkError(message);
      setStatus(message);
    } finally {
      setChunkBusy(false);
    }
  }

  async function loadDocumentChunks(document: DocumentItem, offset = 0) {
    if (chunkBusy) return;
    setPanelMode("chunks");
    setSelectedDocument(document);
    setChunkBusy(true);
    setChunkError("");
    try {
      const data = await getChunks({
        limit: CHUNK_PAGE_SIZE,
        offset,
        scenario: scenario || undefined,
        source_path: document.source_path,
      });
      setChunks(data.chunks);
      setChunkTotal(data.total);
      setChunkOffset(data.offset);
      setStatus(`已加载 ${document.source_name} 的 ${data.chunks.length} / ${data.total} 个片段`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "加载片段失败";
      setChunkError(message);
      setStatus(message);
    } finally {
      setChunkBusy(false);
    }
  }

  async function refreshIndexedDocuments(nextScenarios: string[]) {
    const nextScenario = scenario && nextScenarios.includes(scenario) ? scenario : "";
    if (nextScenario !== scenario) {
      setScenario(nextScenario);
    }
    setSelectedSources([]);
    setSelectedDocument(null);
    setChunks([]);
    setChunkTotal(0);
    setChunkOffset(0);
    setChunkError("");
    try {
      const data = await getDocuments({ scenario: nextScenario || undefined });
      setDocuments(data.documents);
      setDocumentTotal(data.total);
    } catch (error) {
      const message = error instanceof Error ? error.message : "加载入库文件失败";
      setDocuments([]);
      setDocumentTotal(0);
      setChunkError(message);
    }
  }

  async function handleIngest() {
    if (chatBusy || ingestBusy) return;
    if (selectedFiles.length === 0) {
      setStatus("请先选择文件或文件夹");
      return;
    }
    setIngestBusy(true);
    setIngestProgress(5);
    setIngestElapsedSeconds(0);
    setIngestPhase("上传文件");
    setStatus("正在上传并重建知识库索引");
    const progressTimer = window.setInterval(() => {
      setIngestElapsedSeconds((current) => current + 1);
      setIngestPhase("解析文档并写入索引");
      setIngestProgress((current) => Math.min(Math.max(current, 42) + 2, 92));
    }, 1000);
    try {
      const ingestRequest = ingestFilesWithProgress(selectedFiles, true, (progress) => {
        setIngestPhase(progress >= 40 ? "解析并写入索引" : "上传文件");
        setIngestProgress((current) => Math.max(current, progress));
      });
      activeIngestRequestRef.current = ingestRequest;
      const result = await ingestRequest.promise;
      window.clearInterval(progressTimer);
      if (result.ingest_status && result.ingest_status !== "completed") {
        setIngestPhase(result.ingest_status === "rebuild_locked" ? "已有重建任务" : "正在处理");
        setIngestProgress(result.ingest_status === "rebuild_locked" ? 0 : 92);
        await refreshState();
        setStatus(
          result.ingest_status === "rebuild_locked"
            ? "已有重建任务正在运行，本次文件夹未写入索引"
            : `入库任务状态：${result.ingest_status}`,
        );
        return;
      }
      setIngestPhase("完成");
      setIngestProgress(100);
      setScenarios(result.scenarios);
      await refreshState();
      await refreshIndexedDocuments(result.scenarios);
      setStatus(`完成入库：${result.indexed_chunks} 个片段，${result.source_documents} 个文档单元`);
      await new Promise((resolve) => window.setTimeout(resolve, 350));
    } catch (error) {
      window.clearInterval(progressTimer);
      setStatus(error instanceof Error ? error.message : "入库失败");
    } finally {
      activeIngestRequestRef.current = null;
      setIngestBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const question = input.trim();
    if (!question || chatBusy || ingestBusy) return;
    setInput("");
    setChatBusy(true);
    setMessages((items) => [...items, { role: "user", content: question }]);
    try {
      const response = await chat({
        message: question,
        session_id: sessionId,
        scenario: scenario || undefined,
        model: model || undefined,
        top_k: topK,
        agentic,
      });
      setSessionId(response.session_id);
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          retrieval_debug: response.retrieval_debug,
        },
      ]);
      setSelectedSources(response.sources);
      setPanelMode("sources");
      setStatus(`检索完成：返回 ${response.sources.length} 条引用，未重建索引`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "问答失败";
      setMessages((items) => [...items, { role: "assistant", content: message }]);
      setStatus(message);
    } finally {
      setChatBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <header className="brand">
          <Sparkles size={22} />
          <div>
            <h1>QueryNest</h1>
            <p>知识库问答工作台</p>
          </div>
        </header>

        <section className="control-section">
          <div className="section-heading">
            <Database size={16} />
            <span>知识库</span>
          </div>
          <input
            ref={fileInputRef}
            className="upload-input"
            type="file"
            accept={SUPPORTED_UPLOAD_ACCEPT}
            multiple
            onChange={handleUploadSelection}
          />
          <input
            ref={(input) => {
              folderInputRef.current = input;
              input?.setAttribute("directory", "");
              input?.setAttribute("webkitdirectory", "");
            }}
            className="upload-input"
            type="file"
            accept={SUPPORTED_UPLOAD_ACCEPT}
            multiple
            onChange={handleUploadSelection}
          />
          <div className="upload-actions">
            <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()}>
              <FileUp size={16} />
              <span>选择文件</span>
            </button>
            <button className="secondary-button" type="button" onClick={() => folderInputRef.current?.click()}>
              <FolderOpen size={16} />
              <span>选择文件夹</span>
            </button>
          </div>
          <div className="upload-summary">{selectedUploadLabel}</div>
          <button className="primary-button" onClick={handleIngest} disabled={ingestBusy || chatBusy || selectedFiles.length === 0}>
            {ingestBusy ? <Loader2 className="spin" size={16} /> : <RefreshCcw size={16} />}
            <span>重建索引</span>
          </button>
          {ingestBusy ? (
            <button className="secondary-button" type="button" onClick={handleCancelIngest}>
              <XCircle size={16} />
              <span>取消</span>
            </button>
          ) : null}
          {ingestBusy ? (
            <div className="ingest-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={ingestProgress}>
              <div className="ingest-progress-meta">
                <span>{ingestPhase}</span>
                <span>{ingestProgress >= 90 ? `已用 ${formatElapsed(ingestElapsedSeconds)}` : `${ingestProgress}%`}</span>
              </div>
              <div className={ingestProgress >= 90 ? "ingest-progress-track processing" : "ingest-progress-track"}>
                <div className="ingest-progress-bar" style={{ width: `${ingestProgress}%` }} />
              </div>
            </div>
          ) : null}
        </section>

        <section className="control-section">
          <div className="section-heading">
            <Settings2 size={16} />
            <span>问答设置</span>
          </div>
          <label>
            <span>场景</span>
            <select value={scenario} onChange={(event) => setScenario(event.target.value)}>
              <option value="">全部场景</option>
              {scenarios.map((item) => (
                <option value={item} key={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>模型</span>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {chatModels.map((item) => (
                <option value={item.model} key={`${item.provider}-${item.model}`}>
                  {item.model}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>最终引用数 {topK}</span>
            <input type="range" min="3" max="12" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={agentic} onChange={(event) => setAgentic(event.target.checked)} />
            <span>深度检索</span>
          </label>
        </section>

        <footer className="status-line">
          <span className={health?.index_ready ? "dot ready" : "dot"} />
          <span>{status}</span>
        </footer>
      </aside>

      <section className="workspace">
        <div className="chat-header">
          <div>
            <h2>QueryNest 工作台</h2>
          </div>
          <div className="metric-strip">
            <span>{health?.indexed_chunks ?? 0} chunks</span>
            <span>{scenarios.length} scenes</span>
            <span>{health?.redis_connected ? "Redis ready" : "Redis offline"}</span>
            <span>{health?.cache_connected ? "Cache ready" : "Cache offline"}</span>
            <span>{sessionId ? "session active" : "new session"}</span>
          </div>
        </div>

        <div className="message-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              <BookOpen size={34} />
              <h3>{health?.index_ready ? "索引已加载，可以直接提问" : "索引尚未构建"}</h3>
              <p>
                {health?.index_ready
                  ? "普通提问只执行检索和重排，不会重建索引。"
                  : "仅首次使用或知识库内容变化后需要点击“重建索引”。"}
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <div className="message-icon">{message.role === "user" ? <MessageSquare size={16} /> : <Bot size={16} />}</div>
                <div className="message-body">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                  {message.sources?.length ? (
                    <button
                      className="source-button"
                      onClick={() => {
                        setSelectedSources(message.sources ?? []);
                        setPanelMode("sources");
                      }}
                    >
                      <FileSearch size={15} />
                      <span>查看 {message.sources?.length ?? 0} 条引用</span>
                    </button>
                  ) : null}
                </div>
              </article>
            ))
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} rows={2} />
          <button className="send-button" disabled={chatBusy || ingestBusy || !input.trim()} title="发送">
            {chatBusy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </form>
      </section>

      <aside className="sources-panel">
        <div className="panel-header">
          <div className="section-heading">
            <FileSearch size={16} />
            <span>{panelMode === "chunks" ? "已建片段" : "引用"}</span>
          </div>
          <div className="panel-actions">
            <button
              className={panelMode === "sources" ? "panel-toggle active" : "panel-toggle"}
              type="button"
              onClick={() => setPanelMode("sources")}
            >
              引用
            </button>
            <button
              className={panelMode === "chunks" ? "panel-toggle active" : "panel-toggle"}
              type="button"
              onClick={() => void loadDocuments()}
              disabled={chunkBusy || !health?.index_ready}
            >
              {chunkBusy ? <Loader2 className="spin" size={13} /> : "片段"}
            </button>
          </div>
        </div>
        <div className="sources-list">
          {panelMode === "chunks" ? (
            selectedDocument ? (
              <ChunkListView
                document={selectedDocument}
                chunks={chunks}
                total={chunkTotal}
                offset={chunkOffset}
                pageSize={CHUNK_PAGE_SIZE}
                busy={chunkBusy}
                error={chunkError}
                onBack={() => {
                  setSelectedDocument(null);
                  setChunks([]);
                  setChunkTotal(0);
                  setChunkOffset(0);
                }}
                onPage={(nextOffset) => void loadDocumentChunks(selectedDocument, nextOffset)}
              />
            ) : (
              <DocumentListView
                documents={documents}
                total={documentTotal}
                busy={chunkBusy}
                error={chunkError}
                onSelect={(document) => void loadDocumentChunks(document, 0)}
              />
            )
          ) : (
            <>
              {selectedSources.length === 0 ? (
                <p className="muted">暂无引用。完成一次问答后会显示引用内容。</p>
              ) : (
                selectedSources.map((source, index) => <SourceView key={source.chunk_id} source={source} index={index + 1} />)
              )}
            </>
          )}
        </div>
      </aside>
    </main>
  );
}

function SourceView({ source, index }: { source: Source; index: number }) {
  const meta = [source.scenario, source.section, source.page ? `p.${source.page}` : ""].filter(Boolean).join(" / ");
  return (
    <section className="source-item">
      <div className="source-title">
        <span>[{index}]</span>
        <strong>{source.source_name}</strong>
      </div>
      {meta ? <p className="source-meta">{meta}</p> : null}
      <p className="source-text">{source.text}</p>
    </section>
  );
}

function DocumentListView({
  documents,
  total,
  busy,
  error,
  onSelect,
}: {
  documents: DocumentItem[];
  total: number;
  busy: boolean;
  error: string;
  onSelect: (document: DocumentItem) => void;
}) {
  if (error) {
    return <p className="muted">{error}</p>;
  }
  if (documents.length === 0) {
    return <p className="muted">{busy ? "正在加载入库文件..." : "暂无入库文件。"}</p>;
  }
  return (
    <>
      <div className="chunk-list-meta">已入库文件 {total} 个</div>
      {documents.map((document) => (
        <button className="document-item" type="button" key={document.source_path || document.source_name} onClick={() => onSelect(document)}>
          <FileText size={15} />
          <span className="document-item-main">
            <strong>{document.source_name}</strong>
            <span>
              {document.scenario}
              {document.file_type ? ` / ${document.file_type}` : ""}
            </span>
          </span>
          <span className="document-item-count">{document.chunk_count}</span>
        </button>
      ))}
    </>
  );
}

function ChunkListView({
  document,
  chunks,
  total,
  offset,
  pageSize,
  busy,
  error,
  onBack,
  onPage,
}: {
  document: DocumentItem;
  chunks: ChunkItem[];
  total: number;
  offset: number;
  pageSize: number;
  busy: boolean;
  error: string;
  onBack: () => void;
  onPage: (offset: number) => void;
}) {
  if (error) {
    return (
      <>
        <button className="source-button" type="button" onClick={onBack}>
          返回文件列表
        </button>
        <p className="muted">{error}</p>
      </>
    );
  }
  if (chunks.length === 0) {
    return (
      <>
        <button className="source-button" type="button" onClick={onBack}>
          返回文件列表
        </button>
        <p className="muted">{busy ? "正在加载片段..." : "暂无可查看片段。"}</p>
      </>
    );
  }
  const previousOffset = Math.max(offset - pageSize, 0);
  const nextOffset = offset + pageSize;
  return (
    <>
      <button className="source-button" type="button" onClick={onBack}>
        返回文件列表
      </button>
      <div className="chunk-list-meta">
        {document.source_name}：{offset + 1}-{Math.min(offset + chunks.length, total)} / {total}
      </div>
      {chunks.map((chunk, index) => (
        <section className="source-item" key={chunk.chunk_id}>
          <div className="source-title">
            <span>[{offset + index + 1}]</span>
            <strong>{chunk.source_name}</strong>
          </div>
          <p className="source-meta">
            {chunk.scenario} / {chunk.content_type}
            {chunk.section ? ` / ${chunk.section}` : ""}
            {chunk.page ? ` / p.${chunk.page}` : ""}
          </p>
          <p className="source-text">{chunk.text}</p>
        </section>
      ))}
      <div className="chunk-pager">
        <button className="secondary-button" type="button" onClick={() => onPage(previousOffset)} disabled={busy || offset === 0}>
          上一页
        </button>
        <button className="secondary-button" type="button" onClick={() => onPage(nextOffset)} disabled={busy || nextOffset >= total}>
          下一页
        </button>
      </div>
    </>
  );
}

export default App;
