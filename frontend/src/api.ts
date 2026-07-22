import type { AdminConversation, AdminConversationMessage, AdminUser, ChatResponse, ChunkListResponse, Conversation, ConversationMessage, CreatedUser, DocumentListResponse, Health, ModelInfo, PersonalMemory, TokenResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const ACCESS_TOKEN_KEY = "querynest.access_token";
const REQUEST_TIMEOUT_MS = 10_000;

type IngestResult = {
  indexed_chunks: number;
  source_documents: number;
  added_documents: number;
  updated_documents: number;
  skipped_documents: number;
  deleted_documents: number;
  moved_documents: number;
  scenarios: string[];
  artifact_dir: string;
  notices: string[];
  doc_id?: string | null;
  ingest_job_id?: string | null;
  ingest_status?: string | null;
};
type PartialIngestResult = Omit<IngestResult, "scenarios" | "notices"> & {
  scenarios?: string[];
  notices?: string[];
};
type UploadProgressHandler = (progress: number) => void;
type UploadRequest = { promise: Promise<IngestResult>; cancel: () => void };

function responseErrorMessage(body: string, fallback: string) {
  if (!body) return fallback;
  try {
    const payload = JSON.parse(body) as { detail?: unknown };
    if (typeof payload.detail === "string" && payload.detail) return payload.detail;
  } catch {
    return body;
  }
  return body;
}

async function request<T>(path: string, init?: RequestInit, timeoutMs: number | null = REQUEST_TIMEOUT_MS): Promise<T> {
  const headers = new Headers(init?.headers);
  const controller = new AbortController();
  const timeoutId = timeoutMs === null ? undefined : window.setTimeout(() => controller.abort(), timeoutMs);
  if (!(init?.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const token = getAccessToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers,
      ...init,
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(responseErrorMessage(detail, `HTTP ${response.status}`));
    }
    if (response.status === 204) return undefined as T;
    return response.json() as Promise<T>;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("请求超时，请确认后端服务已启动后重试。");
    }
    throw error;
  } finally {
    if (timeoutId !== undefined) window.clearTimeout(timeoutId);
  }
}

export function getAccessToken() {
  return sessionStorage.getItem(ACCESS_TOKEN_KEY);
}

export function setAccessToken(token: string) {
  sessionStorage.setItem(ACCESS_TOKEN_KEY, token);
}

export function clearAccessToken() {
  sessionStorage.removeItem(ACCESS_TOKEN_KEY);
}

export function getCurrentRole(): "admin" | "personal" | null {
  const payload = getAccessToken()?.split(".")[1];
  if (!payload) return null;
  try {
    const decoded = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/"))) as { role?: unknown };
    return decoded.role === "admin" || decoded.role === "personal" ? decoded.role : null;
  } catch {
    return null;
  }
}

export function login(username: string, password: string) {
  return request<TokenResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function createPersonalUser(username: string, password: string) {
  return request<CreatedUser>("/api/admin/users", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function getAdminUsers() {
  return request<AdminUser[]>("/api/admin/users");
}

export function getAdminConversations(ownerUserId: string) {
  const query = new URLSearchParams({ owner_user_id: ownerUserId });
  return request<AdminConversation[]>(`/api/admin/conversations?${query.toString()}`);
}

export function getAdminConversationMessages(sessionId: string) {
  return request<AdminConversationMessage[]>(`/api/admin/conversations/${encodeURIComponent(sessionId)}/messages`);
}

export function getConversations() {
  return request<Conversation[]>("/api/conversations");
}

export function getConversationMessages(sessionId: string) {
  return request<ConversationMessage[]>(`/api/conversations/${encodeURIComponent(sessionId)}/messages`);
}

export function deleteConversations(sessionIds: string[]) {
  return request<void>("/api/conversations", {
    method: "DELETE",
    body: JSON.stringify({ session_ids: sessionIds }),
  });
}

export function getPersonalMemories() {
  return request<PersonalMemory[]>("/api/memories");
}

export function updatePersonalMemory(memoryId: string, changes: Partial<Pick<PersonalMemory, "memory_type" | "key" | "value" | "confidence" | "expires_at">>) {
  return request<PersonalMemory>(`/api/memories/${memoryId}`, {
    method: "PATCH",
    body: JSON.stringify(changes),
  });
}

export function deletePersonalMemory(memoryId: string) {
  return request<void>(`/api/memories/${memoryId}`, { method: "DELETE" });
}

function normalizeIngestResult(result: PartialIngestResult): IngestResult {
  return {
    ...result,
    added_documents: typeof result.added_documents === "number" ? result.added_documents : 0,
    updated_documents: typeof result.updated_documents === "number" ? result.updated_documents : 0,
    skipped_documents: typeof result.skipped_documents === "number" ? result.skipped_documents : 0,
    deleted_documents: typeof result.deleted_documents === "number" ? result.deleted_documents : 0,
    moved_documents: typeof result.moved_documents === "number" ? result.moved_documents : 0,
    scenarios: Array.isArray(result.scenarios) ? result.scenarios : [],
    notices: Array.isArray(result.notices) ? result.notices : [],
  };
}

export function getHealth() {
  return request<Health>("/api/health");
}

export function getChunks(params: { limit: number; offset: number; source_path?: string }) {
  const query = new URLSearchParams({
    limit: String(params.limit),
    offset: String(params.offset),
  });
  if (params.source_path) query.set("source_path", params.source_path);
  return request<ChunkListResponse>(`/api/chunks?${query.toString()}`);
}

export function getDocuments() {
  return request<DocumentListResponse>("/api/documents");
}

export function deleteIndexedDocument(sourcePath: string) {
  const query = new URLSearchParams({ source_path: sourcePath });
  return request<{ source_path: string; deleted_chunks: number; remaining_documents: number }>(`/api/documents?${query.toString()}`, {
    method: "DELETE",
  });
}

export function getModels() {
  return request<ModelInfo[]>("/api/models");
}

export function warmup() {
  return request<{ status: string; embedding_warmed: boolean }>("/api/warmup", {
    method: "POST",
  });
}

export function ingestPath(path: string, includeImages: boolean) {
  return request<PartialIngestResult>("/api/ingest/path", {
    method: "POST",
    body: JSON.stringify({ path, rebuild: false, include_images: includeImages }),
  }).then(normalizeIngestResult);
}

export function rebuildIndexedSources() {
  return request<PartialIngestResult>("/api/ingest/rebuild", {
    method: "POST",
  }).then(normalizeIngestResult);
}

export function ingestFiles(files: File[], includeImages: boolean) {
  const formData = new FormData();
  for (const file of files) {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
    formData.append("files", file, relativePath || file.name);
  }
  formData.append("include_images", String(includeImages));
  return request<PartialIngestResult>("/api/ingest/files", {
    method: "POST",
    body: formData,
  }).then(normalizeIngestResult);
}

export function ingestFilesWithProgress(
  files: File[],
  includeImages: boolean,
  onUploadProgress: UploadProgressHandler,
): UploadRequest {
  const formData = new FormData();
  for (const file of files) {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
    formData.append("files", file, relativePath || file.name);
  }
  formData.append("include_images", String(includeImages));

  const request = new XMLHttpRequest();
  const promise = new Promise<IngestResult>((resolve, reject) => {
    request.open("POST", `${API_BASE}/api/ingest/files`);
    const token = getAccessToken();
    if (token) request.setRequestHeader("Authorization", `Bearer ${token}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onUploadProgress(Math.round((event.loaded / event.total) * 40));
      }
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(normalizeIngestResult(JSON.parse(request.responseText) as PartialIngestResult));
      } else {
        reject(new Error(responseErrorMessage(request.responseText, `HTTP ${request.status}`)));
      }
    };
    request.onerror = () => reject(new Error("上传失败"));
    request.onabort = () => reject(new Error("已取消增量入库"));
    request.send(formData);
  });
  return { promise, cancel: () => request.abort() };
}

export function chat(payload: {
  message: string;
  session_id?: string;
  model?: string;
  top_k?: number;
  agentic?: boolean;
}) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  }, null);
}
