import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Activity,
  BookOpen,
  Bot,
  CircleUserRound,
  Database,
  FileSearch,
  FileText,
  FileUp,
  FolderOpen,
  LayoutDashboard,
  Loader2,
  MessageSquare,
  PanelRightClose,
  PanelRightOpen,
  RefreshCcw,
  Search,
  Send,
  Settings2,
  Sparkles,
  XCircle,
} from "lucide-react";
import { chat, clearAccessToken, createPersonalUser, deleteConversations, deleteIndexedDocument, deletePersonalMemory, getAccessToken, getAdminConversationMessages, getAdminConversations, getAdminUsers, getChunks, getConversationMessages, getConversations, getCurrentRole, getDocuments, getHealth, getPersonalMemories, ingestFilesWithProgress, login, rebuildIndexedSources, setAccessToken, updatePersonalMemory, warmup } from "./api";
import type { AdminConversation, AdminConversationMessage, AdminUser, ChunkItem, Conversation, ConversationMessage, DocumentItem, Health, Message, PersonalMemory, Source } from "./types";

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
  const [authenticated, setAuthenticated] = useState(() => Boolean(getAccessToken()));
  const [isAdmin, setIsAdmin] = useState(() => getCurrentRole() === "admin");
  const [health, setHealth] = useState<Health | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [topK, setTopK] = useState(8);
  const [agentic, setAgentic] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationRefreshKey, setConversationRefreshKey] = useState(0);
  const [input, setInput] = useState("");
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const [panelMode, setPanelMode] = useState<"sources" | "chunks">("sources");
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [documentSearch, setDocumentSearch] = useState("");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentTotal, setDocumentTotal] = useState(0);
  const [selectedDocument, setSelectedDocument] = useState<DocumentItem | null>(null);
  const [documentDeleteMode, setDocumentDeleteMode] = useState(false);
  const [selectedDocumentPaths, setSelectedDocumentPaths] = useState<Set<string>>(() => new Set());
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [chunkOffset, setChunkOffset] = useState(0);
  const [chunkBusy, setChunkBusy] = useState(false);
  const [chunkError, setChunkError] = useState("");
  const [ingestBusy, setIngestBusy] = useState(false);
  const [ingestCancelable, setIngestCancelable] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestPhase, setIngestPhase] = useState("");
  const [ingestElapsedSeconds, setIngestElapsedSeconds] = useState(0);
  const [chatBusy, setChatBusy] = useState(false);
  const [chatStage, setChatStage] = useState("");
  const [status, setStatus] = useState("等待连接后端");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const followLatestMessageRef = useRef(true);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const activeIngestRequestRef = useRef<{ cancel: () => void } | null>(null);

  const selectedUploadLabel = useMemo(() => uploadSelectionLabel(selectedFiles), [selectedFiles]);

  useEffect(() => {
    if (authenticated) void refreshState();
  }, [authenticated]);

  useEffect(() => {
    if (!chatBusy) {
      setChatStage("");
      return;
    }
    setChatStage("正在检索知识库");
    const composeTimer = window.setTimeout(() => setChatStage("正在整理检索结果"), 1800);
    const answerTimer = window.setTimeout(() => setChatStage("正在生成回答"), 4500);
    return () => {
      window.clearTimeout(composeTimer);
      window.clearTimeout(answerTimer);
    };
  }, [chatBusy]);

  useEffect(() => {
    const list = messageListRef.current;
    if (list && followLatestMessageRef.current) list.scrollTop = list.scrollHeight;
  }, [messages, chatBusy]);

  function handleMessageListScroll() {
    const list = messageListRef.current;
    if (!list) return;
    followLatestMessageRef.current = list.scrollHeight - list.scrollTop - list.clientHeight < 48;
  }

  function handleLogout() {
    activeIngestRequestRef.current?.cancel();
    activeIngestRequestRef.current = null;
    clearAccessToken();
    setAuthenticated(false);
    setIsAdmin(false);
    setSessionId(undefined);
    setMessages([]);
    setInput("");
    setSelectedSources([]);
    setSelectedFiles([]);
    setDocuments([]);
    setChunks([]);
    setSelectedDocument(null);
    setPanelMode("sources");
  }

  if (!authenticated) {
    return <LoginScreen onAuthenticated={() => {
      setIsAdmin(getCurrentRole() === "admin");
      setAuthenticated(true);
    }} />;
  }

  async function refreshState() {
    setStatus("正在连接后端并预热检索模型");
    try {
      const [healthData] = await Promise.all([
        getHealth(),
        warmup(),
      ]);
      setHealth(healthData);
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
    setStatus("已取消增量入库");
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
      const data = await getDocuments();
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

  function handleDocumentSearch(value: string) {
    setDocumentSearch(value);
    if (!value.trim()) return;
    setInspectorOpen(true);
    setPanelMode("chunks");
    if (documents.length === 0) void loadDocuments();
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

  async function refreshIndexedDocuments() {
    setSelectedSources([]);
    setSelectedDocument(null);
    setDocumentDeleteMode(false);
    setSelectedDocumentPaths(new Set());
    setChunks([]);
    setChunkTotal(0);
    setChunkOffset(0);
    setChunkError("");
    try {
      const data = await getDocuments();
      setDocuments(data.documents);
      setDocumentTotal(data.total);
    } catch (error) {
      const message = error instanceof Error ? error.message : "加载入库文件失败";
      setDocuments([]);
      setDocumentTotal(0);
      setChunkError(message);
    }
  }

  function toggleDocumentSelection(sourcePath: string) {
    setSelectedDocumentPaths((paths) => {
      const nextPaths = new Set(paths);
      if (nextPaths.has(sourcePath)) {
        nextPaths.delete(sourcePath);
      } else {
        nextPaths.add(sourcePath);
      }
      return nextPaths;
    });
  }

  async function handleDeleteSelectedDocuments() {
    if (chunkBusy || ingestBusy) return;
    const selectedDocuments = documents.filter((document) => selectedDocumentPaths.has(document.source_path));
    if (selectedDocuments.length === 0) return;
    if (!window.confirm(`确认从知识库索引中删除选中的 ${selectedDocuments.length} 个文件吗？此操作只删除知识库索引，不会删除本地文件。`)) return;

    setChunkBusy(true);
    setChunkError("");
    try {
      let deletedChunks = 0;
      for (const document of selectedDocuments) {
        const result = await deleteIndexedDocument(document.source_path);
        deletedChunks += result.deleted_chunks;
      }
      await refreshState();
      await refreshIndexedDocuments();
      setStatus(`已从知识库索引删除 ${selectedDocuments.length} 个文件、${deletedChunks} 个片段。`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "删除已入库文档失败";
      setChunkError(message);
      setStatus(message);
    } finally {
      setChunkBusy(false);
    }
  }

  async function handleIngest() {
    if (chatBusy || ingestBusy) return;
    if (selectedFiles.length === 0) {
      setStatus("请先选择文件或文件夹");
      return;
    }
    setIngestBusy(true);
    setIngestCancelable(true);
    setIngestProgress(5);
    setIngestElapsedSeconds(0);
    setIngestPhase("上传文件");
    setStatus("正在上传并增量写入知识库");
    const progressTimer = window.setInterval(() => {
      setIngestElapsedSeconds((current) => current + 1);
      setIngestPhase("解析文档并增量写入索引");
      setIngestProgress((current) => Math.min(Math.max(current, 42) + 2, 92));
    }, 1000);
    try {
      const ingestRequest = ingestFilesWithProgress(selectedFiles, true, (progress) => {
        setIngestPhase(progress >= 40 ? "解析并增量写入索引" : "上传文件");
        setIngestProgress((current) => Math.max(current, progress));
      });
      activeIngestRequestRef.current = ingestRequest;
      const result = await ingestRequest.promise;
      window.clearInterval(progressTimer);
      if (result.ingest_status && result.ingest_status !== "completed") {
        setIngestPhase(result.ingest_status === "rebuild_locked" ? "已有增量入库任务" : "正在处理");
        setIngestProgress(result.ingest_status === "rebuild_locked" ? 0 : 92);
        await refreshState();
        setStatus(
          result.ingest_status === "rebuild_locked"
            ? "已有增量入库任务正在运行，本次文件夹未写入索引"
            : `入库任务状态：${result.ingest_status}`,
        );
        return;
      }
      setIngestPhase("完成");
      setIngestProgress(100);
      await refreshState();
      await refreshIndexedDocuments();
      setStatus(
        `完成入库：${result.indexed_chunks} 个片段，${result.source_documents} 个文档单元；新增 ${result.added_documents}、更新 ${result.updated_documents}、跳过 ${result.skipped_documents}、删除 ${result.deleted_documents}、移动 ${result.moved_documents}`,
      );
      await new Promise((resolve) => window.setTimeout(resolve, 350));
    } catch (error) {
      window.clearInterval(progressTimer);
      setStatus(error instanceof Error ? error.message : "入库失败");
    } finally {
      activeIngestRequestRef.current = null;
      setIngestCancelable(false);
      setIngestBusy(false);
    }
  }

  async function handleReingest() {
    if (chatBusy || ingestBusy) return;
    if (!window.confirm("将使用已保留的原始文件重新解析并替换整个知识库索引。继续吗？")) return;
    setIngestBusy(true);
    setIngestCancelable(false);
    setIngestProgress(5);
    setIngestElapsedSeconds(0);
    setIngestPhase("重新解析并清洗已有文件");
    setStatus("正在重新入库，当前检索索引将在完成后替换");
    try {
      const result = await rebuildIndexedSources();
      if (result.ingest_status === "rebuild_locked") {
        setStatus("已有入库任务正在运行，未启动重新入库");
        return;
      }
      setIngestPhase("完成");
      setIngestProgress(100);
      await refreshState();
      await refreshIndexedDocuments();
      setStatus(`重新入库完成：${result.indexed_chunks} 个片段，${result.source_documents} 个文档单元`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "重新入库失败";
      if (message.includes("source root is unavailable")) {
        window.alert("无法重新入库：原始文件目录已移动或删除。请重新上传缺失文件后再试；现有知识库索引未变。");
      }
      setStatus(message);
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
        top_k: topK,
        agentic,
      });
      setSessionId(response.session_id);
      setConversationRefreshKey((key) => key + 1);
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
      setStatus(`检索完成：返回 ${response.sources.length} 条引用，未触发入库`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "问答失败";
      setMessages((items) => [...items, { role: "assistant", content: message }]);
      setStatus(message);
    } finally {
      setChatBusy(false);
    }
  }

  function handleNewConversation() {
    if (chatBusy) return;
    setSessionId(undefined);
    setMessages([]);
    setInput("");
    setSelectedSources([]);
    setPanelMode("sources");
  }

  function handleLoadConversation(conversation: Conversation, conversationMessages: ConversationMessage[]) {
    followLatestMessageRef.current = true;
    setSessionId(conversation.session_id);
    setMessages(conversationMessages.map((message) => ({ role: message.role, content: message.content })));
    setInput("");
    setSelectedSources([]);
    setPanelMode("sources");
    setStatus(`已恢复会话：${conversation.title || "未命名会话"}`);
  }

  function handleConversationDeleted(deletedSessionId: string) {
    if (deletedSessionId === sessionId) handleNewConversation();
    setConversationRefreshKey((key) => key + 1);
    setStatus("会话已删除。");
  }

  return (
    <main className="enterprise-shell">
      <header className="global-header">
        <div className="global-brand">
          <Sparkles size={18} />
          <strong>QueryNest</strong>
        </div>
        <div className="workspace-context">知识运营中心</div>
        <label className="global-search">
          <Search size={16} />
          <input value={documentSearch} onChange={(event) => handleDocumentSearch(event.target.value)} onFocus={() => void loadDocuments()} placeholder="搜索已入库文档" />
        </label>
        <div className="global-status">
          <Activity size={15} />
          <span>{health?.index_ready ? "服务正常" : "正在连接"}</span>
        </div>
        <div className="account-menu">
          <button className="avatar-button" type="button" onClick={() => setAccountMenuOpen((open) => !open)} aria-label="账户菜单"><CircleUserRound size={23} /></button>
          {accountMenuOpen ? <div className="account-popover">{isAdmin ? <CreatePersonalUserForm /> : <span>个人账户</span>}</div> : null}
        </div>
        <button className="secondary-button" type="button" onClick={handleLogout}>退出登录</button>
      </header>

      <div className={inspectorOpen ? "app-shell" : "app-shell inspector-collapsed"}>
      <aside className="sidebar">
        <nav className="workspace-nav" aria-label="工作区导航">
          <div className="workspace-nav-item active">
            <LayoutDashboard size={16} />
            <span>问答工作台</span>
          </div>
        </nav>

        {isAdmin ? <AdminConversationAuditPanel /> : null}

        <PersonalConversationPanel
          activeSessionId={sessionId}
          refreshKey={conversationRefreshKey}
          busy={chatBusy}
          onNewConversation={handleNewConversation}
          onLoadConversation={handleLoadConversation}
          onConversationDeleted={handleConversationDeleted}
        />

        <PersonalMemoryPanel />

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
            <span>增量入库</span>
          </button>
          <button className="secondary-button" type="button" onClick={handleReingest} disabled={ingestBusy || chatBusy || !health?.index_ready}>
            <RefreshCcw size={16} />
            <span>重新入库</span>
          </button>
          {ingestBusy && ingestCancelable ? (
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
            <span>最多引用数 {topK}</span>
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
          <div className="metric-strip">
            <span>{health?.index_ready ? "知识库已就绪" : "知识库未就绪"}</span>
            <span>{health?.indexed_chunks ?? 0} 个片段</span>
            <span>{sessionId ? "会话进行中" : "新会话"}</span>
          </div>
        </div>

        <div className="message-list" ref={messageListRef} onScroll={handleMessageListScroll}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <BookOpen size={34} />
              <h3>{health?.index_ready ? "索引已加载，可以直接提问" : "索引尚未构建"}</h3>
              <p>
                {health?.index_ready
                  ? "普通提问只执行检索和重排，不会触发入库。"
                  : "仅首次使用或知识库内容变化后需要点击“增量入库”。"}
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
                        setInspectorOpen(true);
                      }}
                    >
                      <FileSearch size={15} />
                      <span>查看 {message.sources?.length ?? 0} 条引用</span>
                    </button>
                  ) : null}
                  <AgenticTrace retrievalDebug={message.retrieval_debug} />
                </div>
              </article>
            ))
          )}
          {chatBusy ? (
            <article className="message assistant chat-progress" aria-live="polite">
              <div className="message-icon"><Bot size={16} /></div>
              <div className="message-body">
                <Loader2 className="spin" size={16} />
                <span>{chatStage}</span>
              </div>
            </article>
          ) : null}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composer-input">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              rows={2}
            />
            <span className="active-model">当前模型：{health?.default_chat_model ?? "加载中"}</span>
          </div>
          <button className="send-button" disabled={chatBusy || ingestBusy || !input.trim()} title="发送">
            {chatBusy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </form>
      </section>

      {inspectorOpen ? <aside className="sources-panel">
        <button className="inspector-boundary-toggle" type="button" onClick={() => setInspectorOpen(false)} title="收起检查器">
          <PanelRightClose size={15} />
        </button>
        <div className="panel-header">
          <div className="section-heading">
            <FileSearch size={16} />
            <span>{panelMode === "chunks" ? `已入库文件 ${documentTotal} 个` : "引用"}</span>
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
        {panelMode === "chunks" && !selectedDocument ? (
          <div className="document-list-toolbar">
            {documentDeleteMode ? (
              <>
                <button className="secondary-button" type="button" onClick={() => { setDocumentDeleteMode(false); setSelectedDocumentPaths(new Set()); }} disabled={chunkBusy || ingestBusy}>取消</button>
                <button className="danger-button" type="button" onClick={() => void handleDeleteSelectedDocuments()} disabled={chunkBusy || ingestBusy || selectedDocumentPaths.size === 0}>删除所选（{selectedDocumentPaths.size}）</button>
              </>
            ) : (
              <button className="secondary-button" type="button" onClick={() => { setDocumentDeleteMode(true); setSelectedDocumentPaths(new Set()); }} disabled={chunkBusy || ingestBusy}>删除索引</button>
            )}
          </div>
        ) : null}
        <p className="panel-description">
          {panelMode === "chunks" ? "浏览当前知识库已建立的检索片段。" : "每次回答的引用来源会显示在这里。"}
        </p>
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
                query={documentSearch}
                busy={chunkBusy}
                ingestBusy={ingestBusy}
                error={chunkError}
                onSelect={(document) => void loadDocumentChunks(document, 0)}
                deleteMode={documentDeleteMode}
                selectedDocumentPaths={selectedDocumentPaths}
                onToggleSelection={toggleDocumentSelection}
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
      </aside> : <aside className="inspector-rail">
        <button className="inspector-boundary-toggle" type="button" onClick={() => setInspectorOpen(true)} title="打开检查器"><PanelRightOpen size={17} /></button>
      </aside>}
      </div>
    </main>
  );
}

function LoginScreen({ onAuthenticated }: { onAuthenticated: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleLogin(event: FormEvent) {
    event.preventDefault();
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await login(username.trim(), password);
      setAccessToken(result.access_token);
      onAuthenticated();
    } catch (loginError) {
      setError(loginError instanceof Error ? loginError.message : "登录失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="enterprise-shell login-shell">
      <form className="login-form" onSubmit={handleLogin}>
        <Sparkles size={28} />
        <h1>QueryNest</h1>
        <p>请使用本地账号登录</p>
        <label>用户名<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" /></label>
        <label>密码<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" /></label>
        {error ? <p className="login-error">{error}</p> : null}
        <button className="primary-button" disabled={busy || !username.trim() || !password} type="submit">
          {busy ? "登录中..." : "登录"}
        </button>
      </form>
    </main>
  );
}

function CreatePersonalUserForm() {
  const [expanded, setExpanded] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleCreate(event: FormEvent) {
    event.preventDefault();
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const created = await createPersonalUser(username.trim(), password);
      setUsername("");
      setPassword("");
      setMessage(`已创建账号：${created.username}`);
      setExpanded(false);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "创建账号失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="control-section admin-user-form">
      <div className="section-heading"><Settings2 size={16} /><span>账号管理</span></div>
      {expanded ? (
        <form onSubmit={handleCreate}>
          <label><span>个人账号用户名</span><input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="off" /></label>
          <label><span>初始密码</span><input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" /></label>
          <div className="admin-user-actions">
            <button className="primary-button" disabled={busy || !username.trim() || !password} type="submit">
              {busy ? "创建中..." : "确认创建"}
            </button>
            <button className="secondary-button" disabled={busy} type="button" onClick={() => setExpanded(false)}>取消</button>
          </div>
        </form>
      ) : (
        <button className="secondary-button" type="button" onClick={() => setExpanded(true)}>创建账号</button>
      )}
      {message ? <p className="admin-user-success">{message}</p> : null}
      {error ? <p className="login-error">{error}</p> : null}
    </section>
  );
}

function formatAuditTime(value?: string | null) {
  if (!value) return "无时间记录";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function PersonalConversationPanel({
  activeSessionId,
  refreshKey,
  busy,
  onNewConversation,
  onLoadConversation,
  onConversationDeleted,
}: {
  activeSessionId?: string;
  refreshKey: number;
  busy: boolean;
  onNewConversation: () => void;
  onLoadConversation: (conversation: Conversation, messages: ConversationMessage[]) => void;
  onConversationDeleted: (sessionId: string) => void;
}) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [expanded, setExpanded] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [deleteMode, setDeleteMode] = useState(false);
  const [selectedSessionIds, setSelectedSessionIds] = useState<Set<string>>(() => new Set());

  async function load() {
    setLoading(true);
    setError("");
    try {
      setConversations(await getConversations());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "无法加载我的会话");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, [refreshKey]);

  async function selectConversation(conversation: Conversation) {
    if (deleteMode || busy || loading || conversation.session_id === activeSessionId) return;
    setLoading(true);
    setError("");
    try {
      onLoadConversation(conversation, await getConversationMessages(conversation.session_id));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "无法加载会话内容");
    } finally {
      setLoading(false);
    }
  }

  function toggleSelection(sessionId: string) {
    setSelectedSessionIds((sessionIds) => {
      const nextSessionIds = new Set(sessionIds);
      if (nextSessionIds.has(sessionId)) nextSessionIds.delete(sessionId);
      else nextSessionIds.add(sessionId);
      return nextSessionIds;
    });
  }

  function cancelDeleteMode() {
    setDeleteMode(false);
    setSelectedSessionIds(new Set());
  }

  async function deleteSelected() {
    const selectedConversations = conversations.filter((conversation) => selectedSessionIds.has(conversation.session_id));
    if (selectedConversations.length === 0 || busy || loading) return;
    if (!window.confirm(`确认删除选中的 ${selectedConversations.length} 个会话吗？删除后无法恢复。`)) return;

    setLoading(true);
    setError("");
    try {
      const deletedSessionIds = new Set(selectedConversations.map((conversation) => conversation.session_id));
      await deleteConversations([...deletedSessionIds]);
      cancelDeleteMode();
      for (const conversation of selectedConversations) onConversationDeleted(conversation.session_id);
      await load();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "删除会话失败");
      await load();
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="control-section personal-conversations-panel">
      <div className="section-heading"><MessageSquare size={16} /><span>我的会话</span></div>
      <button className="primary-button" type="button" onClick={onNewConversation} disabled={busy}>
        新建会话
      </button>
      <button className="secondary-button" type="button" onClick={() => setExpanded((value) => !value)} disabled={loading}>
        {loading ? "加载中..." : expanded ? "收起会话列表" : "查看历史会话"}
      </button>
      {expanded ? (
        <>
          <div className="conversation-list-toolbar">
            {deleteMode ? (
              <>
                <button className="secondary-button" type="button" onClick={cancelDeleteMode} disabled={loading || busy}>取消</button>
                <button className="danger-button" type="button" onClick={() => void deleteSelected()} disabled={loading || busy || selectedSessionIds.size === 0}>删除所选（{selectedSessionIds.size}）</button>
              </>
            ) : (
              <button className="secondary-button" type="button" onClick={() => setDeleteMode(true)} disabled={loading || busy}>删除会话</button>
            )}
            <button className="source-button" type="button" onClick={() => void load()} disabled={loading || busy}>刷新会话列表</button>
          </div>
          {error ? <p className="login-error">{error}</p> : null}
          <div className="conversation-list" aria-label="我的会话列表">
            {conversations.map((conversation) => (
              deleteMode ? (
                <label className="conversation-list-item" key={conversation.session_id}>
                  <input type="checkbox" checked={selectedSessionIds.has(conversation.session_id)} onChange={() => toggleSelection(conversation.session_id)} disabled={loading || busy} />
                  <span><strong title={conversation.title ?? conversation.session_id}>{conversation.title || "未命名会话"}</strong><small>{formatAuditTime(conversation.updated_at)} · {conversation.message_count} 条消息</small></span>
                </label>
              ) : (
                <button
                  className={conversation.session_id === activeSessionId ? "conversation-list-item active" : "conversation-list-item"}
                  type="button"
                  key={conversation.session_id}
                  onClick={() => void selectConversation(conversation)}
                  disabled={loading || busy}
                >
                  <strong title={conversation.title ?? conversation.session_id}>{conversation.title || "未命名会话"}</strong>
                  <span>{formatAuditTime(conversation.updated_at)} · {conversation.message_count} 条消息</span>
                </button>
              )
            ))}
            {conversations.length === 0 && !loading ? <p className="muted">暂无历史会话。</p> : null}
          </div>
        </>
      ) : null}
    </section>
  );
}

function AdminConversationAuditPanel() {
  const [expanded, setExpanded] = useState(false);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null);
  const [conversations, setConversations] = useState<AdminConversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<AdminConversation | null>(null);
  const [messages, setMessages] = useState<AdminConversationMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function loadUsers() {
    setBusy(true);
    setError("");
    setSelectedUser(null);
    setConversations([]);
    setSelectedConversation(null);
    setMessages([]);
    try {
      setUsers(await getAdminUsers());
      setExpanded(true);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "加载用户列表失败");
    } finally {
      setBusy(false);
    }
  }

  function collapseUsers() {
    setExpanded(false);
    setSelectedUser(null);
    setConversations([]);
    setSelectedConversation(null);
    setMessages([]);
    setError("");
  }

  async function selectUser(user: AdminUser) {
    if (selectedUser?.user_id === user.user_id) {
      setSelectedUser(null);
      setConversations([]);
      setSelectedConversation(null);
      setMessages([]);
      return;
    }
    setBusy(true);
    setError("");
    setSelectedUser(user);
    setConversations([]);
    setSelectedConversation(null);
    setMessages([]);
    try {
      setConversations(await getAdminConversations(user.user_id));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "加载会话列表失败");
    } finally {
      setBusy(false);
    }
  }

  async function selectConversation(conversation: AdminConversation) {
    setBusy(true);
    setError("");
    setSelectedConversation(conversation);
    setMessages([]);
    try {
      setMessages(await getAdminConversationMessages(conversation.session_id));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "加载会话内容失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="control-section admin-audit-panel">
      <div className="section-heading"><FileSearch size={16} /><span>会话审计</span></div>
      <p className="audit-notice">仅供只读审计。查看行为会被记录，不能在此会话中继续提问。</p>
      <button className="secondary-button" type="button" onClick={() => expanded ? collapseUsers() : void loadUsers()} disabled={busy}>
        {busy ? "加载中..." : expanded ? "收起用户会话" : "查看用户会话"}
      </button>
      {expanded ? (
        <>
          <button className="source-button" type="button" onClick={() => void loadUsers()} disabled={busy}>刷新用户列表</button>
          {error ? <p className="login-error">{error}</p> : null}
          <div className="audit-list" aria-label="用户列表">
            {users.map((user) => (
              <button
                className={selectedUser?.user_id === user.user_id ? "audit-list-item active" : "audit-list-item"}
                type="button"
                key={user.user_id}
                onClick={() => void selectUser(user)}
                disabled={busy}
                title={selectedUser?.user_id === user.user_id ? "收起会话列表" : "查看会话"}
              >
                <strong>{user.username}</strong>
                <span>{user.role === "admin" ? "管理员" : "个人用户"}</span>
              </button>
            ))}
            {users.length === 0 && !busy ? <p className="muted">暂无可审计用户。</p> : null}
          </div>
          {selectedUser ? (
            <div className="audit-stage">
              <p className="audit-stage-title">{selectedUser.username} 的会话</p>
              <div className="audit-list" aria-label="会话列表">
                {conversations.map((conversation) => (
                  <button
                    className={selectedConversation?.session_id === conversation.session_id ? "audit-list-item active" : "audit-list-item"}
                    type="button"
                    key={conversation.session_id}
                    onClick={() => void selectConversation(conversation)}
                    disabled={busy}
                  >
                    <strong title={conversation.title ?? conversation.session_id}>{conversation.title || "未命名会话"}</strong>
                    <span>{formatAuditTime(conversation.updated_at ?? conversation.created_at)}{conversation.message_count === undefined ? "" : ` · ${conversation.message_count} 条消息`}</span>
                  </button>
                ))}
                {conversations.length === 0 && !busy ? <p className="muted">该用户暂无会话。</p> : null}
              </div>
            </div>
          ) : null}
          {selectedConversation ? (
            <div className="audit-stage">
              <p className="audit-stage-title">会话内容（只读）</p>
              <div className="audit-messages" aria-live="polite">
                {messages.map((message, index) => (
                  <article className={`audit-message ${message.role}`} key={`${message.role}-${index}`}>
                    <strong>{message.role === "user" ? "用户" : "助手"}</strong>
                    <p>{message.content}</p>
                    {message.created_at ? <small>{formatAuditTime(message.created_at)}</small> : null}
                  </article>
                ))}
                {messages.length === 0 && !busy ? <p className="muted">该会话暂无消息。</p> : null}
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

function PersonalMemoryPanel() {
  const [memories, setMemories] = useState<PersonalMemory[]>([]);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [editing, setEditing] = useState<PersonalMemory | null>(null);

  async function load() {
    setError("");
    try {
      setMemories(await getPersonalMemories());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "无法加载我的记忆");
    }
  }

  useEffect(() => { void load(); }, []);

  async function remove(memoryId: string) {
    setBusyId(memoryId);
    setError("");
    try {
      await deletePersonalMemory(memoryId);
      setMemories((items) => items.filter((item) => item.memory_id !== memoryId));
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "删除记忆失败");
    } finally {
      setBusyId(null);
    }
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!editing) return;
    setBusyId(editing.memory_id);
    setError("");
    try {
      const updated = await updatePersonalMemory(editing.memory_id, {
        memory_type: editing.memory_type,
        key: editing.key,
        value: editing.value,
        confidence: editing.confidence,
        expires_at: editing.expires_at || null,
      });
      setMemories((items) => items.map((item) => item.memory_id === updated.memory_id ? updated : item));
      setEditing(null);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "更新记忆失败");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <section className="control-section memory-panel">
      <div className="section-heading"><BookOpen size={16} /><span>我的记忆</span></div>
      <button className="source-button" type="button" onClick={() => void load()} disabled={busyId !== null}>刷新</button>
      {error ? <p className="login-error">{error}</p> : null}
      {memories.length === 0 ? <p className="muted">暂无可管理的记忆。</p> : null}
      <div className="memory-list">
        {memories.map((memory) => (
          <article className="memory-item" key={memory.memory_id}>
            <strong>{memory.key}</strong>
            <span className="memory-type">{memory.memory_type}</span>
            <p>{memory.value}</p>
            <small>置信度 {memory.confidence.toFixed(2)}</small>
            <small>{memory.expires_at ? `到期：${new Date(memory.expires_at).toLocaleDateString()}` : "无到期日"}</small>
            <small>{memory.source_session_id ? `来源会话：${memory.source_session_id}` : "来源会话：无"}</small>
            <div className="memory-actions">
              <button className="secondary-button" type="button" onClick={() => setEditing({ ...memory, expires_at: memory.expires_at ? memory.expires_at.slice(0, 16) : null })} disabled={busyId !== null}>编辑</button>
              <button className="secondary-button" type="button" onClick={() => void remove(memory.memory_id)} disabled={busyId !== null}>删除</button>
            </div>
          </article>
        ))}
      </div>
      {editing ? (
        <form className="memory-edit-form" onSubmit={save}>
          <label>类型<select value={editing.memory_type} onChange={(event) => setEditing({ ...editing, memory_type: event.target.value as PersonalMemory["memory_type"] })}><option value="preference">preference</option><option value="profile">profile</option><option value="fact">fact</option></select></label>
          <label>键<input value={editing.key} maxLength={120} onChange={(event) => setEditing({ ...editing, key: event.target.value })} /></label>
          <label>内容<textarea value={editing.value} maxLength={500} onChange={(event) => setEditing({ ...editing, value: event.target.value })} /></label>
          <label>置信度<input type="number" min="0" max="1" step="0.01" value={editing.confidence} onChange={(event) => setEditing({ ...editing, confidence: Number(event.target.value) })} /></label>
          <label>到期时间<input type="datetime-local" value={editing.expires_at ?? ""} onChange={(event) => setEditing({ ...editing, expires_at: event.target.value ? new Date(event.target.value).toISOString() : null })} /></label>
          <div className="memory-actions"><button className="primary-button" disabled={busyId !== null} type="submit">保存</button><button className="secondary-button" type="button" onClick={() => setEditing(null)} disabled={busyId !== null}>取消</button></div>
        </form>
      ) : null}
    </section>
  );
}

function AgenticTrace({ retrievalDebug }: { retrievalDebug?: Record<string, unknown> }) {
  const agentic = retrievalDebug?.agentic;
  if (!agentic || typeof agentic !== "object" || Array.isArray(agentic)) return null;
  const trace = agentic as Record<string, unknown>;
  const attempts = Array.isArray(trace.attempts) ? trace.attempts : [];
  const plan = trace.plan && typeof trace.plan === "object" ? trace.plan as Record<string, unknown> : undefined;
  const verification = trace.verification && typeof trace.verification === "object"
    ? trace.verification as Record<string, unknown>
    : undefined;
  const plannedQueries = Array.isArray(plan?.queries) ? plan.queries.filter((query): query is string => typeof query === "string") : [];
  const attemptedQueries = Array.isArray(trace.attempted_queries)
    ? trace.attempted_queries.filter((query): query is string => typeof query === "string")
    : plannedQueries;

  return (
    <details className="agentic-trace">
      <summary>深度检索轨迹</summary>
      {attemptedQueries.length ? <p>查询：{attemptedQueries.join("；")}</p> : <p>查询规划未返回可用子查询。</p>}
      <p>检索轮次：{attempts.length || 1}</p>
      {verification ? <p>证据验证：{verification.supported === true ? "通过" : "未通过"}</p> : null}
    </details>
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
  query,
  busy,
  ingestBusy,
  error,
  onSelect,
  deleteMode,
  selectedDocumentPaths,
  onToggleSelection,
}: {
  documents: DocumentItem[];
  total: number;
  query: string;
  busy: boolean;
  ingestBusy: boolean;
  error: string;
  onSelect: (document: DocumentItem) => void;
  deleteMode: boolean;
  selectedDocumentPaths: Set<string>;
  onToggleSelection: (sourcePath: string) => void;
}) {
  if (error) {
    return <p className="muted">{error}</p>;
  }
  const filteredDocuments = documents.filter((document) => document.source_name.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  if (documents.length === 0) {
    return <p className="muted">{busy ? "正在加载入库文件..." : "暂无入库文件。"}</p>;
  }
  return (
    <>
      {filteredDocuments.length === 0 ? <p className="muted">未找到匹配的入库文档。</p> : null}
      {filteredDocuments.map((document) => (
        <div className="document-item" key={document.source_path || document.source_name}>
          {deleteMode ? (
            <label className="document-item-main">
              <input
                type="checkbox"
                checked={selectedDocumentPaths.has(document.source_path)}
                onChange={() => onToggleSelection(document.source_path)}
                disabled={busy || ingestBusy}
              />
              <FileText size={15} />
              <span>
                <strong>{document.source_name}</strong>
                <span>
                  {document.scenario}
                  {document.file_type ? ` / ${document.file_type}` : ""}
                </span>
              </span>
            </label>
          ) : (
            <button className="document-item-main" type="button" onClick={() => onSelect(document)}>
              <FileText size={15} />
              <span>
                <strong>{document.source_name}</strong>
                <span>
                  {document.scenario}
                  {document.file_type ? ` / ${document.file_type}` : ""}
                </span>
              </span>
            </button>
          )}
          <span className="document-item-count">{document.chunk_count}</span>
        </div>
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
