"use client";

import { useState, useCallback } from "react";
import { streamChatCompletion, sendChatCompletion } from "@/lib/api";
import { useChatStore } from "@/stores/chat";
import type { ChatCompletionRequest, ChatMessage } from "@/types";

export function useChat() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { addMessage, updateLastMessage, setStreaming } = useChatStore();

  const sendMessage = useCallback(
    async (
      sessionId: string,
      message: string,
      request: Omit<ChatCompletionRequest, "messages"> & { messages: ChatMessage[] }
    ) => {
      setError(null);

      // Add user message
      addMessage(sessionId, { role: "user", content: message });

      // Add empty assistant message
      addMessage(sessionId, { role: "assistant", content: "" });

      setIsStreaming(true);
      setStreaming(true);

      try {
        if (request.stream) {
          const stream = streamChatCompletion(request);
          let fullContent = "";

          for await (const chunk of stream) {
            const content = chunk.choices?.[0]?.delta?.content || "";
            fullContent += content;
            updateLastMessage(sessionId, content);
          }
        } else {
          const response = await sendChatCompletion(request);
          const content = response.choices?.[0]?.message?.content || "";
          updateLastMessage(sessionId, content);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
        updateLastMessage(sessionId, "\n\n[Error: Failed to get response]");
      } finally {
        setIsStreaming(false);
        setStreaming(false);
      }
    },
    [addMessage, updateLastMessage, setStreaming]
  );

  return {
    sendMessage,
    isStreaming,
    error,
  };
}
