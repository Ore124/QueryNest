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
  retrieval_debug: RetrievalDebug;
};

export type ChunkItem = {
  chunk_id: string;
  text: string;
  source_path: string;
  source_name: string;
  file_type: string;
  scenario: string;
  section?: string | null;
  page?: number | null;
  content_type: string;
  chunk_index?: number | null;
};

export type ChunkListResponse = {
  chunks: ChunkItem[];
  total: number;
  offset: number;
  limit: number;
};

export type DocumentItem = {
  source_path: string;
  source_name: string;
  file_type: string;
  scenario: string;
  chunk_count: number;
};

export type DocumentListResponse = {
  documents: DocumentItem[];
  total: number;
};

export type RetrievalDebug = Record<string, unknown>;

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

export type TokenResponse = {
  access_token: string;
  token_type: string;
};

export type CreatedUser = {
  user_id: string;
  username: string;
  role: "personal";
};

export type AdminUser = {
  user_id: string;
  username: string;
  role: "admin" | "personal";
};

export type AdminConversation = {
  session_id: string;
  title?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  message_count?: number;
};

export type AdminConversationMessage = {
  role: "user" | "assistant";
  content: string;
  created_at?: string | null;
};

export type Conversation = {
  session_id: string;
  title?: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type ConversationMessage = {
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

export type PersonalMemory = {
  memory_id: string;
  memory_type: "preference" | "profile" | "fact";
  key: string;
  value: string;
  confidence: number;
  source_session_id?: string | null;
  expires_at?: string | null;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  retrieval_debug?: RetrievalDebug;
};
