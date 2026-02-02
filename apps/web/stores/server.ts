import { create } from "zustand";

interface ServerState {
  isRunning: boolean;
  port: number;
  isLoading: boolean;
  error: string | null;

  setRunning: (running: boolean) => void;
  setPort: (port: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useServerStore = create<ServerState>((set) => ({
  isRunning: false,
  port: 8080,
  isLoading: false,
  error: null,

  setRunning: (isRunning) => set({ isRunning }),
  setPort: (port) => set({ port }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
