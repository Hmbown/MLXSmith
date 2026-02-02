import { create } from "zustand";
import type { ChatMessage } from "@/types";

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  model: string;
  createdAt: number;
  updatedAt: number;
}

interface ChatState {
  sessions: ChatSession[];
  currentSessionId: string | null;
  isStreaming: boolean;

  // Actions
  createSession: (model: string) => string;
  deleteSession: (id: string) => void;
  setCurrentSession: (id: string) => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  updateLastMessage: (sessionId: string, content: string) => void;
  setStreaming: (streaming: boolean) => void;
  clearSessions: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  isStreaming: false,

  createSession: (model: string) => {
    const id = `session-${Date.now()}`;
    const newSession: ChatSession = {
      id,
      title: "New Chat",
      messages: [],
      model,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    set((state) => ({
      sessions: [newSession, ...state.sessions],
      currentSessionId: id,
    }));
    return id;
  },

  deleteSession: (id: string) => {
    set((state) => ({
      sessions: state.sessions.filter((s) => s.id !== id),
      currentSessionId:
        state.currentSessionId === id
          ? state.sessions.find((s) => s.id !== id)?.id || null
          : state.currentSessionId,
    }));
  },

  setCurrentSession: (id: string) => {
    set({ currentSessionId: id });
  },

  addMessage: (sessionId: string, message: ChatMessage) => {
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId
          ? {
              ...s,
              messages: [...s.messages, message],
              updatedAt: Date.now(),
              title:
                s.messages.length === 0 && message.role === "user"
                  ? message.content.slice(0, 30) + "..."
                  : s.title,
            }
          : s
      ),
    }));
  },

  updateLastMessage: (sessionId: string, content: string) => {
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId
          ? {
              ...s,
              messages: s.messages.map((m, i) =>
                i === s.messages.length - 1 && m.role === "assistant"
                  ? { ...m, content: m.content + content }
                  : m
              ),
            }
          : s
      ),
    }));
  },

  setStreaming: (isStreaming: boolean) => {
    set({ isStreaming });
  },

  clearSessions: () => {
    set({ sessions: [], currentSessionId: null });
  },
}));
