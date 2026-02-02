import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  // Theme
  theme: "light" | "dark" | "system";
  setTheme: (theme: "light" | "dark" | "system") => void;

  // API Configuration
  apiUrl: string;
  setApiUrl: (url: string) => void;

  // Project Settings
  projectPath: string;
  setProjectPath: (path: string) => void;

  // HuggingFace
  hfToken: string | null;
  setHfToken: (token: string | null) => void;

  // Chat Defaults
  defaultModel: string;
  setDefaultModel: (model: string) => void;
  defaultSystemPrompt: string;
  setDefaultSystemPrompt: (prompt: string) => void;
  defaultTemperature: number;
  setDefaultTemperature: (temp: number) => void;
  defaultTopP: number;
  setDefaultTopP: (topP: number) => void;
  defaultMaxTokens: number;
  setDefaultMaxTokens: (tokens: number) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      // Theme
      theme: "system",
      setTheme: (theme) => set({ theme }),

      // API Configuration
      apiUrl: "http://localhost:8080",
      setApiUrl: (apiUrl) => set({ apiUrl }),

      // Project Settings
      projectPath: "",
      setProjectPath: (projectPath) => set({ projectPath }),

      // HuggingFace
      hfToken: null,
      setHfToken: (hfToken) => set({ hfToken }),

      // Chat Defaults
      defaultModel: "",
      setDefaultModel: (defaultModel) => set({ defaultModel }),
      defaultSystemPrompt: "You are a helpful assistant.",
      setDefaultSystemPrompt: (defaultSystemPrompt) => set({ defaultSystemPrompt }),
      defaultTemperature: 0.7,
      setDefaultTemperature: (defaultTemperature) => set({ defaultTemperature }),
      defaultTopP: 1.0,
      setDefaultTopP: (defaultTopP) => set({ defaultTopP }),
      defaultMaxTokens: 1024,
      setDefaultMaxTokens: (defaultMaxTokens) => set({ defaultMaxTokens }),
    }),
    {
      name: "mlxsmith-settings",
    }
  )
);
