import type {
  ModelInfo,
  ChatCompletionRequest,
  ChatCompletionChunk,
  RLMState,
  RLMHistoryEntry,
  ChatMessage,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Helper for fetch with error handling
async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
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
export async function listModels(): Promise<ModelInfo[]> {
  return fetchJSON<ModelInfo[]>("/internal/models/list");
}

export async function pullModel(modelId: string): Promise<{ success: boolean; message: string }> {
  return fetchJSON("/internal/models/pull", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
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
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
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

// Server Control API
export async function getServerStatus(): Promise<{ running: boolean; port: number }> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    return { running: res.ok, port: 8080 };
  } catch {
    return { running: false, port: 8080 };
  }
}

export async function startServer(port: number): Promise<{ success: boolean }> {
  return fetchJSON("/internal/server/start", {
    method: "POST",
    body: JSON.stringify({ port }),
  });
}

export async function stopServer(): Promise<{ success: boolean }> {
  return fetchJSON("/internal/server/stop", {
    method: "POST",
  });
}

// Settings API
export async function saveHFToken(token: string): Promise<{ success: boolean }> {
  return fetchJSON("/internal/hf/token", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

export async function getHFToken(): Promise<{ token: string | null }> {
  return fetchJSON("/internal/hf/token");
}

export async function saveProjectPath(path: string): Promise<{ success: boolean }> {
  return fetchJSON("/internal/settings/project-path", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export async function getProjectPath(): Promise<{ path: string | null }> {
  return fetchJSON("/internal/settings/project-path");
}
