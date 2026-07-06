import type { ChatResponse, ChunkListResponse, DocumentListResponse, Health, ModelInfo } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type IngestResult = {
  indexed_chunks: number;
  source_documents: number;
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers =
    init?.body instanceof FormData ? init?.headers : { "Content-Type": "application/json", ...(init?.headers ?? {}) };
  const response = await fetch(`${API_BASE}${path}`, {
    headers,
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(responseErrorMessage(detail, `HTTP ${response.status}`));
  }
  return response.json() as Promise<T>;
}

function normalizeIngestResult(result: PartialIngestResult): IngestResult {
  return {
    ...result,
    scenarios: Array.isArray(result.scenarios) ? result.scenarios : [],
    notices: Array.isArray(result.notices) ? result.notices : [],
  };
}

export function getHealth() {
  return request<Health>("/api/health");
}

export function getScenarios() {
  return request<{ scenarios: string[] }>("/api/scenarios");
}

export function getChunks(params: { limit: number; offset: number; scenario?: string; source_path?: string }) {
  const query = new URLSearchParams({
    limit: String(params.limit),
    offset: String(params.offset),
  });
  if (params.scenario) query.set("scenario", params.scenario);
  if (params.source_path) query.set("source_path", params.source_path);
  return request<ChunkListResponse>(`/api/chunks?${query.toString()}`);
}

export function getDocuments(params: { scenario?: string } = {}) {
  const query = new URLSearchParams();
  if (params.scenario) query.set("scenario", params.scenario);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<DocumentListResponse>(`/api/documents${suffix}`);
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
    body: JSON.stringify({ path, rebuild: true, include_images: includeImages }),
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
    request.onabort = () => reject(new Error("已取消重建索引"));
    request.send(formData);
  });
  return { promise, cancel: () => request.abort() };
}

export function chat(payload: {
  message: string;
  session_id?: string;
  scenario?: string;
  model?: string;
  top_k?: number;
  agentic?: boolean;
}) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
