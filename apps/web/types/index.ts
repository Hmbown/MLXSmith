// API Types

export type JSONValue =
  | string
  | number
  | boolean
  | null
  | { [key: string]: JSONValue }
  | JSONValue[];

export interface ModelInfo {
  id: string;
  path: string;
  size_bytes?: number;
  format: "mlx" | "hf" | "gguf";
  has_adapter: boolean;
  adapter_path?: string;
  metadata?: Record<string, JSONValue>;
  downloaded_at?: number;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  total: number;
  cache_dir: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  max_tokens: number;
  temperature: number;
  top_p: number;
  top_k?: number;
  stream?: boolean;
  stop?: string[];
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  choices: {
    index: number;
    delta: {
      content?: string;
    };
    finish_reason: string | null;
  }[];
}

export interface RLMTrainingMetrics {
  loss?: number;
  reward_mean?: number;
  reward_std?: number;
  kl_div?: number;
  learning_rate?: number;
}

export interface RLMState {
  status: "idle" | "running" | "paused" | "completed" | "error";
  iteration?: number;
  total_iterations?: number;
  metrics?: RLMTrainingMetrics;
  started_at?: number;
  updated_at?: number;
  error_message?: string;
}

export interface RLMHistoryEntry {
  iteration: number;
  timestamp: number;
  adapter_score?: number;
  base_score?: number;
  improvement?: number;
  metrics?: RLMTrainingMetrics;
}

export interface AdapterReloadResponse {
  ok: boolean;
  base_model: string;
  adapter_path?: string;
  message?: string;
}

export interface HFTokenResponse {
  ok: boolean;
  validated: boolean;
  username?: string;
  message: string;
  storage_method: "keyring" | "file" | "memory";
}
