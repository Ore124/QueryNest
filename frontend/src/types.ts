export type Source = {
  chunk_id: string;
  text: string;
  source_path: string;
  source_name: string;
  file_type: string;
  scenario: string;
  section?: string | null;
  page?: number | null;
  content_type: string;
  parser: string;
  table_id?: string | null;
  table_markdown?: string | null;
  table_json?: string | null;
  table_html?: string | null;
  dense_rank?: number | null;
  bm25_rank?: number | null;
  dense_score?: number | null;
  bm25_score?: number | null;
  rrf_score: number;
  rerank_rank?: number | null;
  rerank_score?: number | null;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  sources: Source[];
  retrieval_debug: Record<string, unknown>;
};

export type Health = {
  status: string;
  index_ready: boolean;
  indexed_chunks: number;
  default_chat_model: string;
  default_embedding_model: string;
  history_backend: string;
  redis_connected?: boolean | null;
  cache_backend: string;
  cache_connected?: boolean | null;
  index_origin: string;
  index_build_count: number;
};

export type ModelInfo = {
  provider: string;
  model: string;
  role: string;
  available: boolean;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};
