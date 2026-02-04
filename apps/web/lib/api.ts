import type {
  ModelsListResponse,
  ChatCompletionRequest,
  ChatCompletionChunk,
  RLMState,
  RLMHistoryEntry,
  ChatMessage,
  AdapterReloadResponse,
  HFTokenResponse,
} from "@/types";
import type { LocalModel } from "@/lib/models";
import { toLocalModel } from "@/lib/models";
import { useSettingsStore } from "@/stores/settings";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

function resolveApiBase(): string {
  if (typeof window !== "undefined") {
    const fromStore = useSettingsStore.getState().apiUrl?.trim();
    if (fromStore) {
      return fromStore.replace(/\/+$/, "");
    }
  }
  return DEFAULT_API_BASE.replace(/\/+$/, "");
}

// Helper for fetch with error handling
async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const base = resolveApiBase();
  const res = await fetch(`${base}${url}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const error = await res.text().catch(() => "Unknown error");
    throw new Error(`API Error: ${res.status} - ${error}`);
  }

  return res.json() as Promise<T>;
}

// Models API
export async function listModels(): Promise<LocalModel[]> {
  const response = await fetchJSON<ModelsListResponse>("/internal/models/list");
  return response.models.map(toLocalModel);
}

export async function pullModel(input: {
  modelId: string;
  quantize?: boolean;
  qBits?: number;
}): Promise<{ ok: boolean; model_id: string; message?: string }> {
  const body: Record<string, unknown> = {
    model_id: input.modelId,
    quantize: input.quantize ?? false,
  };
  if (typeof input.qBits === "number") {
    body.q_bits = input.qBits;
  }

  return fetchJSON("/internal/models/pull", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function deleteModel(modelId: string): Promise<{ ok: boolean; model_id: string; message?: string }> {
  const query = new URLSearchParams({ model_id: modelId });
  return fetchJSON(`/internal/models/delete?${query.toString()}`, {
    method: "POST",
  });
}

// Chat API
export async function sendChatCompletion(
  request: ChatCompletionRequest
): Promise<{
  id: string;
  choices: { message: ChatMessage; finish_reason: string }[];
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}> {
  return fetchJSON("/v1/chat/completions", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function* streamChatCompletion(
  request: ChatCompletionRequest
): AsyncGenerator<ChatCompletionChunk, void, unknown> {
  const base = resolveApiBase();
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...request, stream: true }),
  });

  if (!res.ok) {
    const error = await res.text().catch(() => "Unknown error");
    throw new Error(`API Error: ${res.status} - ${error}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;

      const payload = line.replace("data:", "").trim();
      if (payload === "[DONE]") return;

      try {
        const chunk: ChatCompletionChunk = JSON.parse(payload);
        yield chunk;
      } catch {
        // Ignore parse errors
      }
    }
  }
}

// RLM API
export async function getRLMState(): Promise<RLMState> {
  return fetchJSON<RLMState>("/internal/rlm/state");
}

export async function getRLMHistory(): Promise<RLMHistoryEntry[]> {
  return fetchJSON<RLMHistoryEntry[]>("/internal/rlm/history");
}

// Adapter reload
export async function reloadAdapter(adapterPath?: string, reloadBase = false): Promise<AdapterReloadResponse> {
  return fetchJSON("/internal/adapter/reload", {
    method: "POST",
    body: JSON.stringify({
      adapter_path: adapterPath ?? null,
      reload_base: reloadBase,
    }),
  });
}

// Server Control API
export async function getServerStatus(): Promise<{ running: boolean }> {
  try {
    const base = resolveApiBase();
    const res = await fetch(`${base}/health`, { method: "GET" });
    return { running: res.ok };
  } catch {
    return { running: false };
  }
}

// Settings API
export async function saveHFToken(token: string, validate = true, persist = true): Promise<HFTokenResponse> {
  return fetchJSON("/internal/hf/token", {
    method: "POST",
    body: JSON.stringify({ token, validate_token: validate, persist }),
  });
}
