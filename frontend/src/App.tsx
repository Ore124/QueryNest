import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  BookOpen,
  Bot,
  Database,
  FileSearch,
  Loader2,
  MessageSquare,
  RefreshCcw,
  Send,
  Settings2,
  Sparkles,
} from "lucide-react";
import { chat, getHealth, getModels, getScenarios, ingestPath, warmup } from "./api";
import type { Health, Message, ModelInfo, Source } from "./types";

const DEFAULT_PATH = "D:\\Codex Projects\\knowledge";

function indexOriginLabel(origin: string) {
  if (origin === "milvus") return "Milvus 加载";
  if (origin === "disk") return "磁盘加载";
  if (origin === "rebuilt") return "当前进程构建";
  return origin || "未知来源";
}

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [sourcePath, setSourcePath] = useState(DEFAULT_PATH);
  const [includeImages, setIncludeImages] = useState(true);
  const [scenario, setScenario] = useState("");
  const [model, setModel] = useState("");
  const [topK, setTopK] = useState(8);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("接口 500 怎么排查？");
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const [ingestBusy, setIngestBusy] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [status, setStatus] = useState("等待连接后端");

  const chatModels = useMemo(() => models.filter((item) => item.role === "chat"), [models]);

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
          ? `已从${indexOriginLabel(healthData.index_origin)} ${healthData.indexed_chunks} 个片段`
          : "索引未构建",
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "后端连接失败");
    }
  }

  async function handleIngest() {
    if (chatBusy || ingestBusy) return;
    setIngestBusy(true);
    setStatus("正在重建 Milvus + BM25 索引");
    try {
      const result = await ingestPath(sourcePath, includeImages);
      setScenarios(result.scenarios);
      await refreshState();
      setStatus(`完成入库：${result.indexed_chunks} 个片段，${result.source_documents} 个文档单元`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "入库失败");
    } finally {
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
      });
      setSessionId(response.session_id);
      setMessages((items) => [...items, { role: "assistant", content: response.answer, sources: response.sources }]);
      setSelectedSources(response.sources);
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
            <h1>RAG Assistant</h1>
            <p>MinerU + PaddleOCR + RRF</p>
          </div>
        </header>

        <section className="control-section">
          <div className="section-heading">
            <Database size={16} />
            <span>知识库</span>
          </div>
          <label>
            <span>本地路径</span>
            <input value={sourcePath} onChange={(event) => setSourcePath(event.target.value)} />
          </label>
          <label className="toggle-row">
            <input type="checkbox" checked={includeImages} onChange={(event) => setIncludeImages(event.target.checked)} />
            <span>解析图片</span>
          </label>
          <button className="primary-button" onClick={handleIngest} disabled={ingestBusy || chatBusy}>
            {ingestBusy ? <Loader2 className="spin" size={16} /> : <RefreshCcw size={16} />}
            <span>重建索引</span>
          </button>
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
        </section>

        <footer className="status-line">
          <span className={health?.index_ready ? "dot ready" : "dot"} />
          <span>{status}</span>
        </footer>
      </aside>

      <section className="workspace">
        <div className="chat-header">
          <div>
            <p>企业知识库问答</p>
            <h2>多轮检索工作台</h2>
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
                    <button className="source-button" onClick={() => setSelectedSources(message.sources ?? [])}>
                      <FileSearch size={15} />
                      <span>查看 {message.sources.length} 条引用</span>
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
        <div className="section-heading">
          <FileSearch size={16} />
          <span>引用与融合排序</span>
        </div>
        <div className="sources-list">
          {selectedSources.length === 0 ? (
            <p className="muted">暂无引用。完成一次问答后会显示检索细节。</p>
          ) : (
            selectedSources.map((source, index) => <SourceView key={source.chunk_id} source={source} index={index + 1} />)
          )}
        </div>
      </aside>
    </main>
  );
}

function SourceView({ source, index }: { source: Source; index: number }) {
  return (
    <section className="source-item">
      <div className="source-title">
        <span>[{index}]</span>
        <strong>{source.source_name}</strong>
      </div>
      <div className="rank-grid">
        <span>Milvus {source.dense_rank ?? "-"}</span>
        <span>BM25 {source.bm25_rank ?? "-"}</span>
        <span>RRF {source.rrf_score.toFixed(4)}</span>
        <span>Rerank {source.rerank_rank ?? "-"}</span>
        <span>重排分 {source.rerank_score?.toFixed(4) ?? "-"}</span>
      </div>
      <p className="source-meta">
        {source.scenario} / {source.content_type} / {source.parser}
        {source.section ? ` / ${source.section}` : ""}
        {source.page ? ` / p.${source.page}` : ""}
      </p>
      <p className="source-text">{source.text}</p>
    </section>
  );
}

export default App;
