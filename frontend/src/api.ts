import type { ChatResponse, Health, ModelInfo } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getHealth() {
  return request<Health>("/api/health");
}

export function getScenarios() {
  return request<{ scenarios: string[] }>("/api/scenarios");
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
  return request<{ indexed_chunks: number; source_documents: number; scenarios: string[]; index_dir: string }>(
    "/api/ingest/path",
    {
      method: "POST",
      body: JSON.stringify({ path, rebuild: true, include_images: includeImages }),
    },
  );
}

export function chat(payload: {
  message: string;
  session_id?: string;
  scenario?: string;
  model?: string;
  top_k?: number;
}) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
