// API Types

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  quantization?: string;
  architecture?: string;
  lastUsed?: string;
  path: string;
  isAdapter?: boolean;
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

export interface RLMState {
  status?: 'idle' | 'running' | 'paused' | 'error';
  currentStep?: number;
  totalSteps?: number;
  currentLoss?: number;
  bestScore?: number;
  config?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface RLMHistoryEntry {
  step: number;
  timestamp: number;
  loss?: number;
  reward?: number;
  adapter_score?: number;
  passed?: boolean;
  metrics?: Record<string, number>;
}

export interface ServerStatus {
  running: boolean;
  port: number;
  url: string;
}

export interface TrainingConfig {
  model: string;
  dataset: string;
  output: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  loraR: number;
  loraAlpha: number;
  loraDropout: number;
}

export interface HFTokenResponse {
  success: boolean;
  message?: string;
}
